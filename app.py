import streamlit as st
import time

# main.py
import os
import asyncio
from dotenv import load_dotenv
import json
from typing import List, Dict, Any # Importar tipos para o histórico de chat

# Importa as funções de alto nível dos seus módulos separados
from browser_agent import run_fomento_search_agent
from indexador_pdf import process_pdfs_into_documents # Usando o nome que você forneceu
from rag import HuggingFaceEmbedding, perguntar_openai, retrieve_documents # Usando o nome que você forneceu
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Importa a função de download
from download_manager import download_pdfs_from_editals_json
from edital_manager import load_cached_grants
import re

# --- Configurações Iniciais ---
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Inicialização Global da Vector Store e Embedding ---
embedding_function = HuggingFaceEmbedding()
vectorstore = Chroma(
    embedding_function=embedding_function,
    persist_directory="chroma"
)

def responde_por_metadados(pergunta: str) -> str | None:
    grants = load_cached_grants()
    # Exemplo: deadline até dezembro de 2025
    deadline_match = re.search(r'(até|antes de|no máximo)\s*(dezembro|12)[/\- ]?2025', pergunta, re.IGNORECASE)
    if deadline_match:
        # Extrai todos os editais com deadline até 31/12/2025
        from datetime import datetime
        def parse_deadline(deadline):
            # Extrai datas do campo deadline (pode ter múltiplas datas)
            datas = re.findall(r'(\d{2}/\d{2}/\d{4})', deadline)
            return [datetime.strptime(d, '%d/%m/%Y') for d in datas]
        limite = datetime(2025, 12, 31)
        resultados = []
        for edital in grants:
            datas = parse_deadline(edital.get('deadline',''))
            if datas and any(data <= limite for data in datas):
                resultados.append(edital)
        if resultados:
            resposta = 'Editais com deadline até dezembro de 2025:\n'
            for e in resultados:
                resposta += f"- {e.get('title','')} (Agência: {e.get('agency','')}, Deadline: {e.get('deadline','')}, URL: {e.get('url','')})\n"
            return resposta
        else:
            return 'Nenhum edital com deadline até dezembro de 2025 encontrado.'
    # Outros filtros podem ser implementados aqui (ex: agência, título, etc)
    return None

def start_qa_session(user_question):
    print("\n--- Inciando sessão de Perguntas e Respostas. Digite 'voltar' para retornar ao menu principal. ---")
    
    chat_history: List[Dict[str, str]] = [] # NOVO: Inicializa o histórico de chat para a sessão

    # Tenta responder por metadados antes de consultar o LLM
    resposta_meta = responde_por_metadados(user_question)
    if resposta_meta:
        return resposta_meta

    # --- Filtro por número de edital na pergunta ---
    edital_num_match = re.search(r'(\d{1,3}/\d{4})', user_question)
    edital_num = edital_num_match.group(1) if edital_num_match else None

    docs = retrieve_documents(user_question, vectorstore)
    print(f"Docs retornados: {len(docs)}")
    # Se houver número de edital, priorize chunks que contenham esse número
    if edital_num:
        docs_prioritarios = [d for d in docs if edital_num in d.page_content or edital_num in d.metadata.get('title','')]
        docs = docs_prioritarios + [d for d in docs if edital_num not in d.page_content and edital_num not in d.metadata.get('title','')]
        print(f"Chunks priorizados para edital {edital_num}: {len(docs_prioritarios)}")

    # Filtro por URL fornecida na pergunta (mantido)
    url_match = re.search(r'https?://\S+', user_question)
    url_prioritaria = url_match.group(0) if url_match else None
    if url_prioritaria:
        docs_prioritarios = [d for d in docs if d.metadata.get('url','') == url_prioritaria]
        docs = docs_prioritarios + [d for d in docs if d.metadata.get('url','') != url_prioritaria]
        print(f"Chunks priorizados para URL {url_prioritaria}: {len(docs_prioritarios)}")

    for d in docs:
        score = getattr(d, 'score', None) or getattr(d, 'similarity_score', None)
        if score is not None:
            print(f"--- DOC (score: {score:.2f}) ---")
        else:
            print("--- DOC ---")
        print(d.page_content[:200])  # Mostra o início do texto de cada doc

    MAX_CHARS = 80000 
    contexto = ""
    if not docs:
        response_content = "Desculpe, não encontrei informações relevantes para sua pergunta nos editais indexados. Por favor, tente indexar mais dados."
    else:
        for doc in docs:
            meta = doc.metadata
            link = meta.get('url', '')
            deadline = meta.get('deadline', '')
            score = getattr(doc, 'score', None) or getattr(doc, 'similarity_score', None)
            contexto += doc.page_content + "\n"
            if score is not None:
                contexto += f"[Similaridade com a pergunta: {score:.2%}]\n"
            if link:
                contexto += f"Link do edital: {link}\n"
            if deadline:
                contexto += f"Prazo (deadline): {deadline}\n"
            contexto += "\n"
        print("Contexto passado para o LLM:")
        print(contexto[:1000])  # Mostra o início do contexto
        try:
            # Passa o histórico de chat para a função perguntar_openai
            response_content = perguntar_openai(user_question, contexto, chat_history=chat_history) 
        except Exception as e:
            response_content = f"Ocorreu um erro ao gerar a resposta: {e}. Por favor, verifique sua chave da API ou o status do serviço do LLM."
    return response_content

# --- CONFIGURAÇÃO DA PÁGINA ---
# Define o título da página, o ícone e o layout.
st.set_page_config(page_title="Meu Chatbot", page_icon="🤖", layout="centered")

# --- FUNÇÃO DE RESPOSTA DO BOT ---
# Esta é uma função de exemplo. Em um caso real, você faria uma chamada
# para uma API de um modelo de linguagem (como a API do Gemini, OpenAI, etc.).
def get_bot_response(user_input):
    """Gera uma resposta simples do bot."""
    # Simula um "pensamento" do bot por um breve momento
    time.sleep(1) 
    return f"Recebi sua mensagem! Você disse: '{user_input}'"

# --- Função Auxiliar para Adicionar Documentos à Vector Store ---
def add_documents_to_vectorstore(documents_to_add: list[Document]):
    if documents_to_add:
        print(f"Adicionando {len(documents_to_add)} documentos à Vector Store...")
        vectorstore.add_documents(documents_to_add)
        print(f"**{len(documents_to_add)}** documentos adicionados e persistidos com sucesso na Vector Store.")
    else:
        print("Nenhum documento para adicionar à Vector Store.")

def chama_browser_use():
    online_grants_data = run_fomento_search_agent()
    if online_grants_data:
        print(f"**{len(online_grants_data)}** editais abertos (incluindo novos e existentes) agora estão no cache.")
        
        # --- CHAMADA CORRIGIDA PARA BAIXAR PDFs ---
        print("\nIniciando automaticamente o download dos PDFs dos editais encontrados...")
        download_dir = "pdfs_baixados"
        # <<< CORREÇÃO AQUI: Receber a lista de caminhos diretamente >>>
        downloaded_pdf_paths: List[str] = download_pdfs_from_editals_json(online_grants_data, download_dir)
        
        print(f"Total de PDFs baixados nesta execução: {len(downloaded_pdf_paths)}")
        # --- FIM CHAMADA ---

        # Processa APENAS os PDFs que foram baixados NESTA execução
        if downloaded_pdf_paths:
            downloaded_pdf_chunks = process_pdfs_into_documents(downloaded_pdf_paths)
            print(f"Total de chunks retornados: {len(downloaded_pdf_chunks)}")
            add_documents_to_vectorstore(downloaded_pdf_chunks)
        else:
            print("Nenhum PDF baixado para indexar a partir dos editais online.")
        
        print("\nEditais online atualizados, baixados (se aplicável) e indexados. Você pode fazer perguntas agora.")
    else:
        print("\nNenhum edital online aberto foi encontrado ou permaneceu no cache após a atualização.")
        print("Não foi possível atualizar o índice com novos editais online.")

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
# O st.session_state é um dicionário que persiste entre os reruns do script.
# Usamos para manter o estado da nossa aplicação.

# Verifica se o histórico de chats já existe no estado da sessão.
if "chat_history" not in st.session_state:
    # Se não existir, inicializa com um dicionário vazio.
    st.session_state.chat_history = {}

# Verifica se um chat ativo está definido.
if "current_chat_id" not in st.session_state:
    # Se não, cria um novo chat ao iniciar a aplicação.
    initial_chat_id = f"chat_{int(time.time())}"
    st.session_state.current_chat_id = initial_chat_id
    # Cada chat é uma lista de mensagens. Começamos com uma mensagem do assistente.
    st.session_state.chat_history[initial_chat_id] = [
        {"role": "assistant", "content": "Olá! Como posso te ajudar hoje?"}
    ]

# --- LÓGICA DA BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    #st.title("Atualizar Editais")
    #if st.button("🔄 Atualizar Editais"):
        # Inicia o processo de atualização dos editais
        #with st.spinner("Atualizando editais..."):
            #asyncio.run(chama_browser_use())
        #st.success("Editais atualizados com sucesso!")

    st.title("Histórico de Conversas")

    # Botão para criar uma nova conversa
    if st.button("➕ Nova Conversa"):
        # Cria um ID único para a nova conversa baseado no tempo atual
        new_chat_id = f"chat_{int(time.time())}"
        # Define este novo ID como o chat atual
        st.session_state.current_chat_id = new_chat_id
        # Inicializa a nova conversa com uma mensagem de boas-vindas
        st.session_state.chat_history[new_chat_id] = [
            {"role": "assistant", "content": "Nova conversa iniciada. Mande sua pergunta!"}
        ]
        # Força o rerun do script para atualizar a interface
        st.rerun()

    st.write("---") # Linha divisória

    # Exibe as conversas anteriores como botões na barra lateral
    # Itera sobre as chaves (IDs dos chats) em ordem reversa (mais recentes primeiro)
    for chat_id in reversed(list(st.session_state.chat_history.keys())):
        # Pega a primeira mensagem do usuário para usar como título do chat
        # Se não houver mensagem do usuário ainda, usa um título padrão.
        messages = st.session_state.chat_history[chat_id]
        first_user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "Nova Conversa...")
        
        # Cria um botão com um trecho da primeira mensagem
        if st.button(first_user_message[:30] + "...", key=f"btn_{chat_id}", use_container_width=True):
            # Se o botão for clicado, define o ID do chat correspondente como o chat atual
            st.session_state.current_chat_id = chat_id
            # Força o rerun para exibir o chat selecionado
            st.rerun()

    #with st.expander("Configurações Avançadas", expanded=False):
       # st.write("API do OpenRouter")
       # api_key = st.text_input("Chave da API", type="password", key="api_key_input")
        #os.environ["OPENROUTER_API_KEY"] = api_key

# --- LÓGICA DA JANELA PRINCIPAL DO CHAT ---

st.title("🤖 Meu Chatbot Pessoal")
st.caption("Um aplicativo de chat simples usando Streamlit")

# Obtém o ID do chat atual
current_chat_id = st.session_state.current_chat_id

# Obtém a lista de mensagens para a conversa atual
current_messages = st.session_state.chat_history[current_chat_id]

# Exibe as mensagens do histórico da conversa atual
# O st.chat_message cria um container de mensagem com avatar
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INPUT DO USUÁRIO ---
# O st.chat_input cria um campo de entrada fixo na parte inferior da tela.
# A variável 'prompt' conterá o texto digitado pelo usuário quando ele pressionar Enter.
if prompt := st.chat_input("Digite sua mensagem aqui..."):
    # 1. Adiciona a mensagem do usuário ao histórico da conversa atual
    current_messages.append({"role": "user", "content": prompt})

    # 2. Exibe a mensagem do usuário na tela imediatamente
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Gera e exibe a resposta do bot
    with st.chat_message("assistant"):
        # Mostra um spinner enquanto o bot "pensa"
        with st.spinner("Pensando..."):
            response = start_qa_session(prompt)
        # Exibe a resposta
        st.markdown(response)

    # 4. Adiciona a resposta do bot ao histórico da conversa
    current_messages.append({"role": "assistant", "content": response})