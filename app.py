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

# --- Configurações Iniciais ---
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Inicialização Global da Vector Store e Embedding ---
embedding_function = HuggingFaceEmbedding()
vectorstore = Chroma(
    embedding_function=embedding_function,
    persist_directory="chroma"
)

def start_qa_session(user_question):
    print("\n--- Inciando sessão de Perguntas e Respostas. Digite 'voltar' para retornar ao menu principal. ---")
    
    chat_history: List[Dict[str, str]] = [] # NOVO: Inicializa o histórico de chat para a sessão

    # user_question = input("\nSua pergunta (ou 'voltar'): ").strip()
    # if user_question.lower() == 'voltar':
    #     print("Retornando ao menu principal.")
    #     break

    docs = retrieve_documents(user_question, vectorstore)

    MAX_CHARS = 80000 
    contexto = ""
    if not docs:
        response_content = "Desculpe, não encontrei informações relevantes para sua pergunta nos editais indexados. Por favor, tente indexar mais dados."
    else:
        for doc in docs:
            if len(contexto) + len(doc.page_content) + 2 <= MAX_CHARS:
                contexto += doc.page_content + "\n\n"
            else:
                print(f"⚠️ Limite de caracteres do contexto atingido. Pulando documentos restantes.")
                break
        
        try:
            # Passa o histórico de chat para a função perguntar_openai
            response_content = perguntar_openai(user_question, contexto, chat_history=chat_history) 
        except Exception as e:
            response_content = f"Ocorreu um erro ao gerar a resposta: {e}. Por favor, verifique sua chave da API ou o status do serviço do LLM."
    
    # # Adiciona a resposta do assistente ao histórico de chat
    # chat_history.append({"role": "assistant", "content": response_content})
    return response_content
    # print("\n📌 Resposta do modelo:\n", response_content)



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

async def chama_browser_use():
    online_grants_data = await run_fomento_search_agent()
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
    st.title("Atualizar Editais")
    if st.button("🔄 Atualizar Editais"):
        # Inicia o processo de atualização dos editais
        with st.spinner("Atualizando editais..."):
            asyncio.run(chama_browser_use())
        st.success("Editais atualizados com sucesso!")

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

    with st.expander("Configurações Avançadas", expanded=False):
        st.write("API do OpenRouter")
        api_key = st.text_input("Chave da API", type="password", key="api_key_input")
        os.environ["OPENROUTER_API_KEY"] = api_key

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