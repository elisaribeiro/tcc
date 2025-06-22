# pdf_indexer.py (apenas a função process_pdfs_into_documents)
import os
import pdfplumber
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import io
from typing import Union, List

# Adicional para Streamlit UploadedFile
from streamlit.runtime.uploaded_file_manager import UploadedFile # Importar o tipo UploadedFile se ainda estiver no app.py

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ... (safe_str_strip, clean_cell permanecem as mesmas) ...

# --- FUNÇÃO process_pdfs_into_documents AJUSTADA ---
def process_pdfs_into_documents(pdf_sources: Union[str, List[Union[io.BytesIO, UploadedFile]]]) -> List[Document]:
    all_documents_for_indexing: List[Document] = []
    
    files_to_process = []

    if isinstance(pdf_sources, str):
        print(f"\n--- Iniciando processamento de PDFs do diretório: {pdf_sources} ---")
        pdf_paths = [os.path.join(pdf_sources, f) for f in os.listdir(pdf_sources) if f.endswith(".pdf")]
        for path in pdf_paths:
            files_to_process.append((os.path.basename(path), open(path, 'rb')))
        
    elif isinstance(pdf_sources, list):
        print("\n--- Iniciando processamento de PDFs carregados via upload ---")
        for uploaded_file_obj in pdf_sources:
            file_name = getattr(uploaded_file_obj, 'name', f"uploaded_file_{len(files_to_process)}.pdf")
            # Certifique-se de que getvalue() é chamado apenas uma vez se for UploadedFile
            if isinstance(uploaded_file_obj, UploadedFile):
                file_obj_bytesio = io.BytesIO(uploaded_file_obj.getvalue())
            else: # Já é BytesIO
                file_obj_bytesio = uploaded_file_obj
            
            files_to_process.append((file_name, file_obj_bytesio))
    else:
        raise ValueError("pdf_sources deve ser um caminho de diretório (str) ou uma lista de objetos UploadedFile/BytesIO.")

    for file_name, file_obj in files_to_process:
        try:
            with pdfplumber.open(file_obj) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()

                    structured_cronograma_content = ""
                    found_main_cronograma_table = False
                    
                    for table_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue
                        
                        headers_row = table[0]
                        cleaned_headers = [
                            cleaned_cell for h in headers_row if h is not None and (cleaned_cell := str(h).replace("\n", " ").replace("\r", " ").strip())
                        ]
                        
                        if "ETAPAS" in cleaned_headers and "DATAS" in cleaned_headers:
                            found_main_cronograma_table = True
                            structured_cronograma_content += "CRONOGRAMA DE EVENTOS E DATAS IMPORTANTES:\n"
                            
                            for row_idx, row in enumerate(table):
                                if row_idx == 0:
                                    continue
                                if len(row) < 2 or row[0] is None or row[1] is None:
                                    continue

                                item = str(row[0]).replace("\n", " ").replace("\r", " ").strip()
                                value = str(row[1]).replace("\n", " ").replace("\r", " ").strip()
                                
                                structured_cronograma_content += f"O evento '{item}' tem a data ou período de {value}.\n"
                    
                    # Crie um ID único para cada chunk para evitar duplicatas no Chroma
                    # Combina nome do arquivo, número da página e tipo de chunk para um ID robusto
                    chunk_id_base = f"{file_name.replace('.', '_')}_page_{i+1}"
                    
                    if found_main_cronograma_table and structured_cronograma_content:
                        all_documents_for_indexing.append(Document(
                            page_content=structured_cronograma_content,
                            metadata={"source": file_name, "page": i + 1, "type": "cronograma_principal"},
                            id=f"{chunk_id_base}_cronograma" # Adiciona um ID único
                        ))
                    elif text and not found_main_cronograma_table:
                        # Para texto geral, divida em chunks menores e dê IDs únicos para cada um
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # Split menor para texto geral
                        text_chunks = text_splitter.split_text(text)
                        for chunk_idx, txt_chunk in enumerate(text_chunks):
                            all_documents_for_indexing.append(Document(
                                page_content=txt_chunk,
                                metadata={"source": file_name, "page": i + 1, "type": "page_text", "chunk_idx": chunk_idx},
                                id=f"{chunk_id_base}_text_{chunk_idx}" # Adiciona um ID único para cada sub-chunk
                            ))
            
            print(f"PDF '{file_name}' processado.")
        except Exception as e:
            print(f"Erro ao processar o arquivo '{file_name}': {e}")
            continue
        finally:
            if isinstance(pdf_sources, str):
                file_obj.close()


    # O splitting já foi feito dentro do loop para texto geral
    # Apenas retorna all_documents_for_indexing que já contém todos os chunks
    # Comentamos esta linha pois chunks já foi criado acima
    # chunks: List[Document] = text_splitter_general.split_documents(all_documents_for_indexing)

    print(f"✅ PDFs processados. Gerados {len(all_documents_for_indexing)} chunks.")
    return all_documents_for_indexing # Retorna a lista já chunkada e com IDs