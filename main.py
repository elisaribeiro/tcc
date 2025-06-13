# main_orchestrator.py
import os
import asyncio
from dotenv import load_dotenv
import json # Para lidar com a saída JSON do agente

# Importa as funções e classes dos seus módulos separados
from browser_agent import run_fomento_search_agent # Função para buscar online
from indexador_pdf import process_pdfs_into_documents # Função para processar PDFs
from rag import HuggingFaceEmbedding, perguntar_openai, retrieve_documents # Funções do seu sistema RAG
from langchain_chroma import Chroma # Classe Chroma
from langchain_core.documents import Document # Classe Document

# --- Configurações Iniciais ---
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Inicialização Global da Vector Store e Embedding ---
# A função de embedding é inicializada uma única vez
embedding_function = HuggingFaceEmbedding()

# A Vector Store é inicializada ou carregada. Ela será a mesma instância
# para onde os PDFs e os editais online serão adicionados.
vectorstore = Chroma(
    embedding_function=embedding_function,
    persist_directory="chroma" # O diretório onde seus embeddings serão salvos/carregados
)
print("Vector Store ChromaDB inicializada globalmente para orquestrador.")

# --- Função Auxiliar para Adicionar Documentos à Vector Store ---
def add_documents_to_vectorstore(documents_to_add: list[Document]):
    """
    Adiciona uma lista de documentos à vector store e persiste as mudanças.
    """
    if documents_to_add:
        print(f"Adicionando {len(documents_to_add)} documentos à Vector Store...")
        vectorstore.add_documents(documents_to_add)
        print("Documentos adicionados e persistidos com sucesso na Vector Store.")
    else:
        print("Nenhum documento para adicionar à Vector Store.")

# --- Função para Iniciar o Chatbot (Perguntas e Respostas) ---
async def start_chatbot_session():
    print("\n--- Inciando sessão de Perguntas e Respostas. Digite 'voltar' para retornar ao menu principal. ---")
    while True:
        user_question = input("\nSua pergunta (ou 'voltar'): ").strip()
        if user_question.lower() == 'voltar':
            print("Retornando ao menu principal.")
            break

        print(f"Buscando resposta para: '{user_question}'...")
        
        docs = retrieve_documents(user_question, vectorstore) # Usa a vectorstore global

        MAX_CHARS = 25000 
        contexto = ""
        if not docs:
            print("Não foram encontrados documentos relevantes para a sua pergunta no momento. Tente indexar mais dados ou reformule a pergunta.")
            continue # Pula para a próxima iteração do loop de perguntas

        for doc in docs:
            if len(contexto) + len(doc.page_content) + 2 <= MAX_CHARS:
                contexto += doc.page_content + "\n\n"
            else:
                print(f"⚠️ Limite de caracteres do contexto atingido. Pulando documentos restantes.")
                break
        
        print("\n📌 Resposta do modelo:")
        try:
            response = perguntar_openai(user_question, contexto)
            print(response)
        except Exception as e:
            print(f"Ocorreu um erro ao gerar a resposta: {e}")
            print("Por favor, verifique sua chave da API ou o status do serviço do LLM.")


# --- FUNÇÃO PRINCIPAL DE ORQUESTRAÇÃO ---
async def main_orchestrator():
    print("\n--- Bem-vindo ao Sistema Híbrido de Consulta de Editais ---")

    while True:
        print("\nSelecione uma opção:")
        print("1. Importar PDFs e fazer perguntas sobre eles")
        print("2. Pesquisar editais em aberto na internet (usando Browser-Use)")
        print("3. Iniciar sessão de Perguntas e Respostas (com dados já indexados)")
        print("4. Carregar editais online do cache e indexar (modo de teste)") # NOVA OPÇÃO
        print("0. Sair")

        choice = input("Digite o número da sua opção: ").strip()

        if choice == '1':
            pdf_dir = input("Digite o caminho do diretório com os PDFs (ex: /Users/seu_usuario/documentos/pdfs): ").strip()
            if os.path.isdir(pdf_dir):
                pdf_documents = process_pdfs_into_documents(pdf_dir)
                add_documents_to_vectorstore(pdf_documents)
                await start_chatbot_session() # Inicia a sessão de perguntas após indexar
            else:
                print("Caminho do diretório inválido. Por favor, tente novamente.")

        elif choice == '2':
            # Executa o Browser-Use
            online_grants_data = await run_fomento_search_agent()
            
            if online_grants_data: # Verifica se a lista não está vazia após a extração
                print("\n--- Editais Online Encontrados e Preparando para Indexação ---")
                online_grant_documents = []
                for grant in online_grants_data:
                    content = (f"Edital: {grant.get('title', 'N/A')}. Agência: {grant.get('agency', 'N/A')}. "
                               f"Prazo: {grant.get('deadline', 'N/A')}. URL: {grant.get('url', 'N/A')}.")
                    online_grant_documents.append(Document(page_content=content, metadata={
                        "source": grant.get('url', 'Online Search'),
                        "title": grant.get('title', 'N/A'),
                        "agency": grant.get('agency', 'N/A'),
                        "deadline": grant.get('deadline', 'N/A'),
                        "type": "edital_online"
                    }))
                
                add_documents_to_vectorstore(online_grant_documents)
                await start_chatbot_session() # Inicia a sessão de perguntas após indexar
            else:
                print("\nNenhum edital online foi extraído para adicionar à Vector Store.")

        elif choice == '3':
            await start_chatbot_session()

        elif choice == '4': # IMPLEMENTAÇÃO DA NOVA OPÇÃO: Carregar do cache
            cache_file = "cached_online_grants.json"
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        online_grants_data = json.load(f)
                    print(f"\n--- Carregando editais online do cache: '{cache_file}' ---")
                    
                    if online_grants_data: # Verifica se a lista carregada do cache não está vazia
                        print("Preparando editais do cache para indexação...")
                        online_grant_documents = []
                        for grant in online_grants_data:
                            content = (f"Edital: {grant.get('title', 'N/A')}. Agência: {grant.get('agency', 'N/A')}. "
                                       f"Prazo: {grant.get('deadline', 'N/A')}. URL: {grant.get('url', 'N/A')}.")
                            online_grant_documents.append(Document(page_content=content, metadata={
                                "source": grant.get('url', 'Online Search (Cached)'),
                                "title": grant.get('title', 'N/A'),
                                "agency": grant.get('agency', 'N/A'),
                                "deadline": grant.get('deadline', 'N/A'),
                                "type": "edital_online"
                            }))
                        add_documents_to_vectorstore(online_grant_documents)
                        await start_chatbot_session()
                    else:
                        print("O arquivo de cache está vazio ou não contém editais válidos.")
                except json.JSONDecodeError:
                    print(f"Erro: O arquivo '{cache_file}' não é um JSON válido. Por favor, remova-o ou gere um novo.")
                except Exception as e:
                    print(f"Ocorreu um erro ao ler o cache: {e}")
            else:
                print(f"Arquivo de cache '{cache_file}' não encontrado. Use a opção '2' primeiro para gerar um cache.")

        elif choice == '0':
            print("Saindo do programa. Adeus!")
            break

        else:
            print("Opção inválida. Por favor, digite 1, 2, 3, 4 ou 0.")


if __name__ == "__main__":
    asyncio.run(main_orchestrator())