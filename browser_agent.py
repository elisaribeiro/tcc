# browser_agent.py
import os
import asyncio
import json
import re
from dotenv import load_dotenv
load_dotenv()

from typing import Optional, Dict, Any, List

from langchain_core.utils.utils import secret_from_env
from pydantic import Field, SecretStr
from browser_use import Agent
from langchain_openai import ChatOpenAI

from edital_manager import manage_editals_cache 

# Sua classe ChatOpenRouter personalizada (mantida como está)
class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 **kwargs):
        openai_api_key = (
            openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openai_api_key, # type: ignore
            **kwargs
        )

async def run_fomento_search_agent() -> List[Dict[str, Any]]:
    llm = ChatOpenRouter(
        model_name="google/gemini-2.0-flash-lite-001", # Modelo mantido
        temperature=0.3,
    )

    task_description = """
    Você é um assistente especializado em encontrar editais de fomento para instituições de ensino.
    Sua tarefa é navegar na internet para encontrar editais de fomento que estejam ATUALMENTE ABERTOS
    e que sejam direcionados a INSTITUIÇÕES DE ENSINO (universidades, escolas, institutos de pesquisa).

    Comece pesquisando em sites conhecidos de agências de fomento brasileiras ou grandes universidades.
    Acessar TODOS os seguintes sites: CNPq, Capes, Finep.
    Para cada edital relevante que encontrar:
    - Verifique se está realmente aberto (não encerrado ou futuro).
    - Verifique se é para instituições de ensino.
    - Anote o Título do Edital, a Agência de Fomento, o Prazo Final (se disponível), e a URL direta do edital.
    - Se encontrar um edital relevante, extraia essas informações de forma estruturada.

    É CRÍTICO QUE SUA SAÍDA FINAL SEJA APENAS O JSON. NÃO ADICIONE TEXTO EXPLICATIVO ANTES OU DEPOIS.
    AO FINAL DA TAREFA, RETORNE UMA ÚNICA SAÍDA JSON CONTENDO UMA LISTA DE OBJETOS, ONDE CADA OBJETO REPRESENTA UM EDITAL COM AS CHAVES 'title', 'agency', 'deadline', E 'url'.
    """

    agent = Agent(
        task=task_description,
        llm=llm,
        max_actions_per_step=10, # Aumentado
    )

    print("Iniciando a busca online por editais de fomento com o Browser-Use...")
    result_history = await agent.run(max_steps=50) # Mantenha 50 para testar mais rápido

    final_content_string = ""
    # Esta variável agora é o resultado do melhor JSON encontrado no parsing principal, ou vazio
    parsed_grants_from_content: List[Dict[str, Any]] = [] 

    # --- Lógica para obter final_content_string do AgentHistoryList ---
    # Prioriza o resultado da ação 'done' (final_result)
    if result_history.final_result and callable(result_history.final_result):
        content_from_final_result_method = result_history.final_result() 
        if isinstance(content_from_final_result_method, str) and content_from_final_result_method.strip():
            final_content_string = content_from_final_result_method.strip()
            print(f"DEBUG: Conteúdo final do agente obtido via chamada final_result().")
    
    # Fallback para result_history.extracted_content (se existir e for string)
    if not final_content_string and hasattr(result_history, 'extracted_content') and isinstance(result_history.extracted_content, str) and result_history.extracted_content.strip():
        final_content_string = result_history.extracted_content.strip()
        print(f"DEBUG: Conteúdo final do agente obtido via extracted_content (direto do AgentHistoryList).")
    
    # Se final_content_string está populado, tente parsear
    if final_content_string:
        try:
            parsed_json = json.loads(final_content_string)
            if isinstance(parsed_json, dict):
                if "open_calls" in parsed_json and isinstance(parsed_json["open_calls"], list):
                    parsed_grants_from_content = parsed_json["open_calls"]
                    print(f"DEBUG: JSON de editais extraído com sucesso do conteúdo final (chave 'open_calls').")
                elif "funding_calls" in parsed_json and isinstance(parsed_json["funding_calls"], list):
                    parsed_grants_from_content = parsed_json["funding_calls"]
                    print(f"DEBUG: JSON de editais extraído com sucesso do conteúdo final (chave 'funding_calls').")
                else:
                    print(f"AVISO: Conteúdo final é JSON de dicionário, mas não tem chaves 'open_calls' ou 'funding_calls'.")
            elif isinstance(parsed_json, list):
                parsed_grants_from_content = parsed_json
                print(f"DEBUG: JSON de editais extraído com sucesso (lista direta).")
            else:
                print(f"AVISO: Conteúdo final é JSON, mas não no formato esperado (dicionário ou lista).")

        except json.JSONDecodeError:
            print("DEBUG: Conteúdo final não é JSON direto. Tentando extrair bloco com regex...")
            json_block_match = re.search(r'```json\s*(.*?)\s*```', final_content_string, re.DOTALL | re.IGNORECASE)
            if json_block_match:
                json_candidate = json_block_match.group(1).strip()
                try:
                    parsed_json_from_block = json.loads(json_candidate)
                    if isinstance(parsed_json_from_block, dict):
                        if "open_calls" in parsed_json_from_block and isinstance(parsed_json_from_block["open_calls"], list):
                            parsed_grants_from_content = parsed_json_from_block["open_calls"]
                            print(f"DEBUG: JSON de editais extraído de bloco ````json``` (chave 'open_calls').")
                        elif "funding_calls" in parsed_json_from_block and isinstance(parsed_json_from_block["funding_calls"], list):
                            parsed_grants_from_content = parsed_json_from_block["funding_calls"]
                            print(f"DEBUG: JSON de editais extraído de bloco ````json``` (chave 'funding_calls').")
                        else:
                            print(f"AVISO: Bloco ````json``` encontrado, mas não tem chaves 'open_calls' ou 'funding_calls'.")
                    elif isinstance(parsed_json_from_block, list):
                        parsed_grants_from_content = parsed_json_from_block
                        print(f"DEBUG: JSON de editais extraído de bloco ````json``` (lista direta).")
                    else:
                        print(f"AVISO: Bloco ````json``` encontrado, mas não no formato esperado.")
                except json.JSONDecodeError as e:
                    print(f"AVISO: Bloco ````json``` encontrado, mas o conteúdo não é um JSON válido. Erro: {e}")
            else:
                print(f"AVISO: Conteúdo final não é JSON direto e não contém bloco ````json```. Conteúdo: {final_content_string[:200]}...")
    
    # --- NOVO: Fallback PRINCIPAL: Tentar extrair JSON de *QUALQUER* 'extracted_content' do histórico ---
    # Se parsed_grants_from_content (o resultado do parsing da string final) ainda estiver vazio,
    # itere pelo histórico de ações para encontrar qualquer JSON válido que tenha sido extraído.
    final_extracted_grants: List[Dict[str, Any]] = parsed_grants_from_content # Começa com o que foi encontrado acima

    if not final_extracted_grants and result_history and hasattr(result_history, 'history'):
        print("DEBUG: Nenhum JSON válido da string final. Tentando extrair JSON de ações 'extracted_content' do histórico...")
        for action_record_obj in reversed(result_history.history): # type: ignore
            # AÇÃO MAIS IMPORTANTE: Verificar se o 'action_record_obj' tem um atributo 'action_data'
            # (ou 'result_data', 'model_output' que contenha a informação da ação e seu resultado)
            # Baseado em frameworks de agente, a saída de uma tool call fica no 'action_data' ou 'tool_output'
            
            # Precisamos do tipo exato do objeto action_record_obj e como ele armazena o resultado de ToolOutput.
            # Se não for 'model_output' ou 'extracted_content' direto, pode ser algo como:
            # action_record_obj.raw_output ou action_record_obj.tool_output.extracted_content
            
            # Para o browser-use, as ações são registradas com o output do controller.
            # O INFO [controller] 📄 Extracted from page: ````json { ... } ```` sugere que o JSON está
            # no 'extracted_content' do ActionResult, que é aninhado.
            
            # Tentativa mais robusta de acesso ao JSON dentro de um item do histórico:
            # Tenta acessar extracted_content diretamente ou via action_result
            temp_content_from_extracted = None
            if hasattr(action_record_obj, 'extracted_content') and isinstance(getattr(action_record_obj, 'extracted_content', None), str):
                temp_content_from_extracted = getattr(action_record_obj, 'extracted_content', None)
            elif hasattr(action_record_obj, 'action_result') and getattr(action_record_obj, 'action_result', None) is not None:
                action_result_obj = getattr(action_record_obj, 'action_result', None)
                if hasattr(action_result_obj, 'extracted_content') and isinstance(getattr(action_result_obj, 'extracted_content', None), str):
                    temp_content_from_extracted = getattr(action_result_obj, 'extracted_content', None)

            if temp_content_from_extracted:
                print(f"DEBUG: Tentando parsear EXTRACTED_CONTENT do histórico: {temp_content_from_extracted[:100]}...")
                try:
                    temp_parsed_json_extracted = json.loads(temp_content_from_extracted)
                    if isinstance(temp_parsed_json_extracted, dict) and ("open_calls" in temp_parsed_json_extracted or "funding_calls" in temp_parsed_json_extracted):
                        final_extracted_grants = temp_parsed_json_extracted.get("open_calls") or temp_parsed_json_extracted.get("funding_calls") or []
                        if not isinstance(final_extracted_grants, list):
                            final_extracted_grants = []
                        else:
                            print(f"DEBUG: JSON extraído com sucesso do histórico (EXTRACTED_CONTENT, formato dict).")
                            break
                    elif isinstance(temp_parsed_json_extracted, list):
                        final_extracted_grants = temp_parsed_json_extracted
                        print(f"DEBUG: JSON extraído com sucesso do histórico (EXTRACTED_CONTENT, formato list).")
                        break
                except json.JSONDecodeError:
                    # Tentar regex em EXTRACTED_CONTENT
                    json_block_match_extracted = re.search(r'```json\s*(.*?)\s*```', temp_content_from_extracted, re.DOTALL | re.IGNORECASE)
                    if json_block_match_extracted:
                        json_candidate_extracted = json_block_match_extracted.group(1).strip()
                        try:
                            temp_parsed_json_from_block_extracted = json.loads(json_candidate_extracted)
                            if isinstance(temp_parsed_json_from_block_extracted, dict) and ("open_calls" in temp_parsed_json_from_block_extracted or "funding_calls" in temp_parsed_json_from_block_extracted):
                                final_extracted_grants = temp_parsed_json_from_block_extracted.get("open_calls") or temp_parsed_json_from_block_extracted.get("funding_calls") or []
                                if not isinstance(final_extracted_grants, list):
                                    final_extracted_grants = []
                                else:
                                    print(f"DEBUG: JSON extraído com sucesso do histórico (EXTRACTED_CONTENT, bloco json).")
                                    break
                            elif isinstance(temp_parsed_json_from_block_extracted, list):
                                final_extracted_grants = temp_parsed_json_from_block_extracted
                                print(f"DEBUG: JSON extraído com sucesso do histórico (EXTRACTED_CONTENT, lista direta, bloco json).")
                                break
                        except json.JSONDecodeError:
                            pass
            
            # --- TENTATIVA ANTERIORMENTE FALHA, MAS QUE PODE SER ÚTIL SE A ESTRUTURA MUDAR ---
            # if hasattr(action_record_obj, 'action_result') and action_record_obj.action_result is not None:
            #     actual_action_result = action_record_obj.action_result
            #     if hasattr(actual_action_result, 'extracted_content') and isinstance(actual_action_result.extracted_content, str):
            #         temp_content = actual_action_result.extracted_content
            #         # Lógica de parsing como acima

    if not final_extracted_grants:
        print("AVISO: NENHUM JSON DE EDITAIS VÁLIDO PÔDE SER EXTRAÍDO DO HISTÓRICO DO AGENTE.")
        if hasattr(result_history, 'final_result') and callable(result_history.final_result):
            # Imprime o que o LLM retornou no 'done' se for textual
            print(f"DEBUG: Conteúdo final da ação 'done' do agente (se textual): {result_history.final_result()}")
        else:
            print("DEBUG: Final result não disponível ou não é callable.")

    # --- CHAMA A FUNÇÃO CENTRALIZADA PARA GERENCIAR O CACHE ---
    final_filtered_editals: List[Dict[str, Any]] = manage_editals_cache(final_extracted_grants)

    return final_filtered_editals