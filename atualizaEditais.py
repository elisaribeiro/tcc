from browser_agent import run_fomento_search_agent
from indexador_pdf import process_pdfs_into_documents
from download_manager import download_pdfs_from_editals_json
from rag import HuggingFaceEmbedding
from langchain_chroma import Chroma
import os

def main():
    print("Iniciando atualização dos editais...")
    online_grants_data = run_fomento_search_agent()
    if online_grants_data:
        print(f"**{len(online_grants_data)}** editais abertos (incluindo novos e existentes) agora estão no cache.")
        print("\nIniciando automaticamente o download dos PDFs dos editais encontrados...")
        download_dir = "pdfs_baixados"
        downloaded_pdf_paths = download_pdfs_from_editals_json(online_grants_data, download_dir)
        print(f"Total de PDFs baixados nesta execução: {len(downloaded_pdf_paths)}")
        if downloaded_pdf_paths:
            downloaded_pdf_chunks = process_pdfs_into_documents(downloaded_pdf_paths)
            print(f"Total de chunks retornados: {len(downloaded_pdf_chunks)}")
            for i, chunk in enumerate(downloaded_pdf_chunks):
                print(f"Chunk {i} (source: {chunk.metadata.get('source')}, page: {chunk.metadata.get('page')}, type: {chunk.metadata.get('type')}):\n{chunk.page_content[:300]}\n{'-'*60}")
            # Adiciona os chunks ao Chroma
            embedding_function = HuggingFaceEmbedding()
            vectorstore = Chroma(
                embedding_function=embedding_function,
                persist_directory="chroma"
            )
            if downloaded_pdf_chunks:
                print(f"Adicionando {len(downloaded_pdf_chunks)} documentos à Vector Store...")
                vectorstore.add_documents(downloaded_pdf_chunks)
                print(f"Documentos adicionados e persistidos com sucesso na Vector Store.")
        else:
            print("Nenhum PDF baixado para indexar a partir dos editais online.")
        print("\nEditais online atualizados, baixados (se aplicável) e indexados. Você pode fazer perguntas agora.")
    else:
        print("\nNenhum edital online aberto foi encontrado ou permaneceu no cache após a atualização.")
        print("Não foi possível atualizar o índice com novos editais online.")
    print("Atualização dos editais concluída.")

if __name__ == "__main__":
    main()
