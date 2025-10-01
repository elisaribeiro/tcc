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
print("Iniciando sistema de RAG (global)...")
embedding_function = HuggingFaceEmbedding()
vectorstore = Chroma(
    embedding_function=embedding_function,
    persist_directory="chroma"
)
print("✅ Sistema de RAG e Vector Store carregados globalmente!")

# # --- Função Auxiliar para Adicionar Documentos à Vector Store ---
# def add_documents_to_vectorstore(documents_to_add: list[Document]):
#     if documents_to_add:
#         print(f"Adicionando {len(documents_to_add)} documentos à Vector Store...")
#         vectorstore.add_documents(documents_to_add)
#         print(f"**{len(documents_to_add)}** documentos adicionados e persistidos com sucesso na Vector Store.")
#     else:
#         print("Nenhum documento para adicionar à Vector Store.")

# # --- Função para Iniciar a Sessão de Perguntas e Respostas (AJUSTADA) ---
# async def start_qa_session():
#     print("\n--- Inciando sessão de Perguntas e Respostas. Digite 'voltar' para retornar ao menu principal. ---")
    
#     chat_history: List[Dict[str, str]] = [] # NOVO: Inicializa o histórico de chat para a sessão

#     while True:
#         user_question = input("\nSua pergunta (ou 'voltar'): ").strip()
#         if user_question.lower() == 'voltar':
#             print("Retornando ao menu principal.")
#             break

#         # Adiciona a pergunta do usuário ao histórico de chat
#         chat_history.append({"role": "user", "content": user_question})
        
#         print(f"Buscando resposta para: '{user_question}'...")
        
#         docs = retrieve_documents(user_question, vectorstore)

#         MAX_CHARS = 80000 
#         contexto = ""
#         if not docs:
#             response_content = "Desculpe, não encontrei informações relevantes para sua pergunta nos editais indexados. Por favor, tente indexar mais dados."
#         else:
#             for doc in docs:
#                 if len(contexto) + len(doc.page_content) + 2 <= MAX_CHARS:
#                     contexto += doc.page_content + "\n\n"
#                 else:
#                     print(f"⚠️ Limite de caracteres do contexto atingido. Pulando documentos restantes.")
#                     break
            
#             try:
#                 # Passa o histórico de chat para a função perguntar_openai
#                 response_content = perguntar_openai(user_question, contexto, chat_history=chat_history) 
#             except Exception as e:
#                 response_content = f"Ocorreu um erro ao gerar a resposta: {e}. Por favor, verifique sua chave da API ou o status do serviço do LLM."
        
#         # Adiciona a resposta do assistente ao histórico de chat
#         chat_history.append({"role": "assistant", "content": response_content})
#         return response_content
#         # print("\n📌 Resposta do modelo:\n", response_content)

# # --- FUNÇÃO PRINCIPAL DE ORQUESTRAÇÃO ---
# async def main_orchestrator():
#     print("\n--- Bem-vindo ao Sistema Híbrido de Consulta de Editais (Terminal) ---")

#     if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
#         print("⚠️ ERRO: Nenhuma chave de API (OPENAI_API_KEY ou OPENROUTER_API_KEY) está configurada no seu arquivo .env.")
#         print("Por favor, configure uma das chaves para continuar.")
#         return

#     while True:
#         print("\nSelecione uma opção:")
#         print("1. Atualizar Editais Online (Browser-Use) e Fazer Perguntas")
#         print("2. Apenas Fazer Perguntas (usar dados já no índice)")
#         print("0. Sair")

#         choice = input("Digite o número da sua opção: ").strip()

#         if choice == '1': # Atualizar Editais Online (Browser-Use) e Fazer Perguntas
#             print("\nIniciando busca e atualização de editais online...")
#             online_grants_data = await run_fomento_search_agent()

#             if online_grants_data:
#                 print(f"**{len(online_grants_data)}** editais abertos (incluindo novos e existentes) agora estão no cache.")
                
#                 # --- CHAMADA CORRIGIDA PARA BAIXAR PDFs ---
#                 print("\nIniciando automaticamente o download dos PDFs dos editais encontrados...")
#                 download_dir = "pdfs_baixados"
#                 # <<< CORREÇÃO AQUI: Receber a lista de caminhos diretamente >>>
#                 downloaded_pdf_paths: List[str] = download_pdfs_from_editals_json(online_grants_data, download_dir)
                
#                 print(f"Total de PDFs baixados nesta execução: {len(downloaded_pdf_paths)}")
#                 # --- FIM CHAMADA ---

#                 # Processa APENAS os PDFs que foram baixados NESTA execução
#                 if downloaded_pdf_paths:
#                     downloaded_pdf_chunks = process_pdfs_into_documents(downloaded_pdf_paths)
#                     add_documents_to_vectorstore(downloaded_pdf_chunks)
#                 else:
#                     print("Nenhum PDF baixado para indexar a partir dos editais online.")
                
#                 print("\nEditais online atualizados, baixados (se aplicável) e indexados. Você pode fazer perguntas agora.")
#                 await start_qa_session()
#             else:
#                 print("\nNenhum edital online aberto foi encontrado ou permaneceu no cache após a atualização.")
#                 print("Não foi possível atualizar o índice com novos editais online.")

#         elif choice == '2':
#             print("\nCarregando dados da Vector Store para perguntas...")
#             await start_qa_session()

#         elif choice == '0':
#             print("Saindo do programa. Adeus!")
#             break

#         else:
#             print("Opção inválida. Por favor, digite 1, 2 ou 0.")

# if __name__ == "__main__":
#     asyncio.run(main_orchestrator())