# main.py
import os
import asyncio
from dotenv import load_dotenv
import json

# Importa as funções de alto nível dos seus módulos separados
from browser_agent import run_fomento_search_agent
from indexador_pdf import process_pdfs_into_documents
from rag import HuggingFaceEmbedding, perguntar_openai, retrieve_documents
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Importa a função de download
from download_manager import download_pdfs_from_editals_json

# --- Configurações Iniciais ---
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Inicialização Global da Vector Store e Embedding ---
print("Iniciando sistema de RAG (global)...")
embedding_function = HuggingFaceEmbedding()
vectorstore = Chroma(
    embedding_function=embedding_function,
    persist_directory="chroma"
)
print("✅ Sistema de RAG e Vector Store carregados globalmente!")

# --- Função Auxiliar para Adicionar Documentos à Vector Store ---
def add_documents_to_vectorstore(documents_to_add: list[Document]):
    if documents_to_add:
        print(f"Adicionando {len(documents_to_add)} documentos à Vector Store...")
        vectorstore.add_documents(documents_to_add)
        print(f"**{len(documents_to_add)}** documentos adicionados e persistidos com sucesso na Vector Store.")
    else:
        print("Nenhum documento para adicionar à Vector Store.")

# --- Função para Iniciar a Sessão de Perguntas e Respostas ---
async def start_qa_session():
    print("\n--- Inciando sessão de Perguntas e Respostas. Digite 'voltar' para retornar ao menu principal. ---")
    while True:
        user_question = input("\nSua pergunta (ou 'voltar'): ").strip()
        if user_question.lower() == 'voltar':
            print("Retornando ao menu principal.")
            break

        print(f"Buscando resposta para: '{user_question}'...")
        
        docs = retrieve_documents(user_question, vectorstore)

        MAX_CHARS = 25000 
        contexto = ""
        if not docs:
            response = "Desculpe, não encontrei informações relevantes para sua pergunta nos editais indexados. Por favor, tente indexar mais dados."
        else:
            for doc in docs:
                if len(contexto) + len(doc.page_content) + 2 <= MAX_CHARS:
                    contexto += doc.page_content + "\n\n"
                else:
                    print(f"⚠️ Limite de caracteres do contexto atingido. Pulando documentos restantes.")
                    break
            
            try:
                # perguntar_openai agora já usa o cliente e modelo OpenRouter configurados internamente
                response = perguntar_openai(user_question, contexto)
            except Exception as e:
                response = f"Ocorreu um erro ao gerar a resposta: {e}. Por favor, verifique sua chave da API ou o status do serviço do LLM."
        
        print("\n📌 Resposta do modelo:\n", response)

# --- FUNÇÃO PRINCIPAL DE ORQUESTRAÇÃO ---
async def main_orchestrator():
    print("\n--- Bem-vindo ao Sistema Híbrido de Consulta de Editais (Terminal) ---")

    # Apenas verifica a chave do OpenRouter, pois é a única usada agora
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️ ERRO: A variável de ambiente OPENROUTER_API_KEY não está configurada no seu arquivo .env.")
        print("Por favor, configure sua chave do OpenRouter para continuar.")
        return

    while True:
        print("\nSelecione uma opção:")
        print("1. Atualizar Editais Online (Browser-Use) e Fazer Perguntas")
        print("2. Apenas Fazer Perguntas (usar dados já no índice)")
        print("0. Sair")

        choice = input("Digite o número da sua opção: ").strip()

        if choice == '1':
            print("\nIniciando busca e atualização de editais online...")
            online_grants_data = await run_fomento_search_agent()
            
            if online_grants_data:
                print(f"**{len(online_grants_data)}** editais abertos (incluindo novos e existentes) agora estão no cache.")
                
                print("\nIniciando automaticamente o download dos PDFs dos editais encontrados...")
                download_dir = "pdfs_baixados"
                downloaded_count = download_pdfs_from_editals_json(online_grants_data, download_dir)
                print(f"Total de PDFs baixados: {downloaded_count}")

                if downloaded_count > 0:
                    downloaded_pdf_chunks = process_pdfs_into_documents(download_dir)
                    add_documents_to_vectorstore(downloaded_pdf_chunks)
                else:
                    print("Nenhum PDF baixado para indexar a partir dos editais online.")
                
                print("\nEditais online atualizados, baixados (se aplicável) e indexados.")
                await start_qa_session()
            else:
                print("\nNenhum edital online aberto foi encontrado ou permaneceu no cache após a atualização.")
                print("Não foi possível atualizar o índice com novos editais online.")

        elif choice == '2':
            print("\nCarregando dados da Vector Store para perguntas...")
            await start_qa_session()

        elif choice == '0':
            print("Saindo do programa. Adeus!")
            break

        else:
            print("Opção inválida. Por favor, digite 1, 2 ou 0.")

if __name__ == "__main__":
    asyncio.run(main_orchestrator())