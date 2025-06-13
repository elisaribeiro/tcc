# browser_agent.py
import os
import asyncio
import json
import re
from dotenv import load_dotenv
load_dotenv()

from browser_use import Agent
from langchain_openai import ChatOpenAI

async def run_fomento_search_agent() -> list:
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
    )

    task_description = """
    Você é um assistente especializado em encontrar editais de fomento para instituições de ensino.
    Sua tarefa é navegar na internet para encontrar editais de fomento que estejam ATUALMENTE ABERTOS
    e que sejam direcionados a INSTITUIÇÕES DE ENSINO (universidades, escolas, institutos de pesquisa).

    Comece pesquisando em sites conhecidos de agências de fomento brasileiras ou grandes universidades.
    Exemplos de sites para começar: FAPESP, CNPq, Capes, Finep, ou portais de notícias sobre pesquisa.
    Para cada edital relevante que encontrar:
    - Verifique se está realmente aberto (não encerrado ou futuro).
    - Verifique se é para instituições de ensino.
    - Anote o Título do Edital, a Agência de Fomento, o Prazo Final (se disponível), e a URL direta do edital.
    - Se encontrar um edital relevante, extraia essas informações de forma estruturada.

    AO FINAL DA TAREFA, RETORNE UMA ÚNICA SAÍDA JSON CONTENDO UMA LISTA DE OBJETOS, ONDE CADA OBJETO REPRESENTA UM EDITAL COM AS CHAVES 'title', 'agency', 'deadline', E 'url'.
    """

    agent = Agent(
        task=task_description,
        llm=llm,
        max_actions_per_step=5,
    )

    print("Iniciando a busca online por editais de fomento com o Browser-Use...")
    result_history = await agent.run(max_steps=50)

    final_content_string = ""
    extracted_grants = [] 

    # --- DEBUG: Verificando o objeto final_result ---
    print(f"DEBUG: Tipo de result_history.final_result: {type(result_history.final_result)}")
    if result_history.final_result:
        print(f"DEBUG: Atributos de result_history.final_result: {dir(result_history.final_result)}")
        if callable(result_history.final_result):
            temp_action_result_returned_str = result_history.final_result() # CHAMA O MÉTODO
            print(f"DEBUG: Tipo do resultado de result_history.final_result(): {type(temp_action_result_returned_str)}")
            print(f"DEBUG: Conteúdo de result_history.final_result() (primeiros 200 chars): {temp_action_result_returned_str[:200]}...") # type: ignore
        else:
            print("AVISO DEBUG: result_history.final_result não é um método callable.")
    # --- FIM DEBUG ---

    # *** LÓGICA REVISADA PARA OBTER final_content_string DE FORMA SEGURA ***
    # Inicializa como None para controle preciso
    content_from_final_result_method = None
    if result_history.final_result and callable(result_history.final_result):
        # Chama o método e armazena o resultado
        content_from_final_result_method = result_history.final_result()
    
    # Prioriza o conteúdo do método final_result() se for uma string válida
    if isinstance(content_from_final_result_method, str) and content_from_final_result_method.strip():
        final_content_string = content_from_final_result_method.strip()
        print(f"DEBUG: Conteúdo final do agente obtido via chamada final_result() e é string válida.")
    # Fallback para result_history.extracted_content se o método não retornou string ou estava vazia
    elif result_history.extracted_content and isinstance(result_history.extracted_content, str) and result_history.extracted_content.strip():
        final_content_string = result_history.extracted_content.strip()
        print(f"DEBUG: Conteúdo final do agente obtido via extracted_content (direto do AgentHistoryList) e é string válida.")
    
    if not final_content_string: # Se após todas as tentativas, ainda estiver vazio
        print("AVISO: Não foi possível obter o conteúdo final esperado do agente.")
        return [] # Retorna lista vazia se não conseguir o conteúdo base


    if final_content_string:
        # --- LÓGICA DE PARSING AJUSTADA (com REGEX, como antes) ---
        try:
            parsed_json = json.loads(final_content_string)
            
            # Prioriza a chave "funding_calls" ou "open_calls" se ela existir e for uma lista
            if isinstance(parsed_json, dict):
                if "open_calls" in parsed_json and isinstance(parsed_json["open_calls"], list):
                    extracted_grants = parsed_json["open_calls"]
                    print(f"DEBUG: JSON de editais extraído com sucesso do conteúdo final (chave 'open_calls').")
                elif "funding_calls" in parsed_json and isinstance(parsed_json["funding_calls"], list):
                    extracted_grants = parsed_json["funding_calls"]
                    print(f"DEBUG: JSON de editais extraído com sucesso do conteúdo final (chave 'funding_calls').")
                else:
                    print(f"AVISO: Conteúdo final é JSON de dicionário, mas não tem chaves 'open_calls' ou 'funding_calls'.")
            elif isinstance(parsed_json, list): # Caso o LLM retorne uma lista direta
                extracted_grants = parsed_json
                print(f"DEBUG: JSON de editais extraído com sucesso (lista direta).")
            else:
                print(f"AVISO: Conteúdo final é JSON, mas não no formato esperado (dicionário ou lista).")

        except json.JSONDecodeError:
            # Se não foi JSON direto, tenta encontrar o bloco ```json``` usando REGEX
            print("DEBUG: Conteúdo não é JSON direto. Tentando extrair bloco com regex...")
            json_block_match = re.search(r'```json\s*(.*?)\s*```', final_content_string, re.DOTALL | re.IGNORECASE)

            if json_block_match:
                json_candidate = json_block_match.group(1).strip()
                try:
                    parsed_json = json.loads(json_candidate)
                    if isinstance(parsed_json, dict):
                        if "open_calls" in parsed_json and isinstance(parsed_json["open_calls"], list):
                            extracted_grants = parsed_json["open_calls"]
                            print(f"DEBUG: JSON de editais extraído de bloco ````json``` (chave 'open_calls').")
                        elif "funding_calls" in parsed_json and isinstance(parsed_json["funding_calls"], list):
                            extracted_grants = parsed_json["funding_calls"]
                            print(f"DEBUG: JSON de editais extraído de bloco ````json``` (chave 'funding_calls').")
                        else:
                            print(f"AVISO: Bloco ````json``` encontrado, mas não tem chaves 'open_calls' ou 'funding_calls'.")
                    elif isinstance(parsed_json, list):
                        extracted_grants = parsed_json
                        print(f"DEBUG: JSON de editais extraído de bloco ````json``` (lista direta).")
                    else:
                        print(f"AVISO: Bloco ````json``` encontrado, mas não no formato esperado.")
                except json.JSONDecodeError as e:
                    print(f"AVISO: Bloco ````json``` encontrado, mas o conteúdo não é um JSON válido. Erro: {e}")
            else:
                print(f"AVISO: Conteúdo final não é JSON direto e não contém bloco ````json```. Conteúdo: {final_content_string[:200]}...")
    
    # --- SALVAR A SAÍDA PARA TESTES FUTUROS (CACHE) ---
    if extracted_grants: # Se editais foram extraídos e parseados com sucesso
        cache_file = "cached_online_grants.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(extracted_grants, f, indent=2, ensure_ascii=False)
            print(f"DEBUG: Editais extraídos e parseados salvos em '{cache_file}' para reuso em testes.")
        except Exception as e:
            print(f"AVISO: Falha ao salvar cache de editais: {e}")
    # --- FIM NOVO ---

    return extracted_grants