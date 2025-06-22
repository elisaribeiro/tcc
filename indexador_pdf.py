# pdf_indexer.py
import os
import pdfplumber
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import io
# >>> MUDANÇA AQUI: Importar Sequence e Union da typing <<<
from typing import Union, List, Sequence 

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def safe_str_strip(value):
    if value is None:
        return ""
    return str(value).strip()

def clean_cell(cell):
    if cell is None:
        return ""
    return str(cell).replace("\n", " ").replace("\r", " ").strip()

# --- FUNÇÃO process_pdfs_into_documents AJUSTADA ---
# >>> MUDANÇA AQUI: Tipo de pdf_sources para Union[str, Sequence[Union[str, io.BytesIO]]] <<<
def process_pdfs_into_documents(pdf_sources: Union[str, Sequence[Union[str, io.BytesIO]]]) -> List[Document]:
    """
    Processa arquivos PDF de um diretório OU uma lista de caminhos de arquivo OU uma lista de objetos BytesIO.
    Extrai texto e tabelas (cronogramas), e retorna uma lista de objetos Document para indexação.

    Args:
        pdf_sources: Um caminho de diretório (str) ou uma sequência de caminhos de arquivo (Sequence[str])
                     ou uma sequência de objetos BytesIO (Sequence[io.BytesIO]).
    """
    all_documents_for_indexing: List[Document] = []
    
    files_to_process = [] # Lista de tuplas (nome_do_arquivo, objeto_arquivo_BytesIO)

    if isinstance(pdf_sources, str): # Se for um caminho de diretório (str)
        print(f"\n--- Iniciando processamento de PDFs do diretório: {pdf_sources} ---")
        pdf_paths = [os.path.join(pdf_sources, f) for f in os.listdir(pdf_sources) if f.endswith(".pdf")]
        for path in pdf_paths:
            files_to_process.append((os.path.basename(path), open(path, 'rb'))) # Abre o arquivo do disco
        
    elif isinstance(pdf_sources, (list, tuple)): # Se for uma lista/tupla (Sequence)
        print("\n--- Iniciando processamento de PDFs da lista fornecida ---")
        for item in pdf_sources:
            if isinstance(item, str): # Se o item é um CAMINHO DE ARQUIVO (str)
                files_to_process.append((os.path.basename(item), open(item, 'rb')))
            elif isinstance(item, io.BytesIO): # Se o item é um BytesIO
                file_name = getattr(item, 'name', f"uploaded_file_{len(files_to_process)}.pdf")
                files_to_process.append((file_name, item))
            else:
                raise TypeError(f"Tipo de item de PDF não suportado na lista: {type(item)}")
    else:
        raise ValueError("pdf_sources deve ser um caminho de diretório (str) ou uma lista/tupla de caminhos/BytesIO.")

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
                    chunk_id_base = f"{file_name.replace('.', '_')}_page_{i+1}"
                    
                    if found_main_cronograma_table and structured_cronograma_content:
                        all_documents_for_indexing.append(Document(
                            page_content=structured_cronograma_content,
                            metadata={"source": file_name, "page": i + 1, "type": "cronograma_principal"},
                            id=f"{chunk_id_base}_cronograma"
                        ))
                    elif text and not found_main_cronograma_table:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        text_chunks = text_splitter.split_text(text)
                        for chunk_idx, txt_chunk in enumerate(text_chunks):
                            all_documents_for_indexing.append(Document(
                                page_content=txt_chunk,
                                metadata={"source": file_name, "page": i + 1, "type": "page_text", "chunk_idx": chunk_idx},
                                id=f"{chunk_id_base}_text_{chunk_idx}"
                            ))
            
            print(f"PDF '{file_name}' processado.")
        except Exception as e:
            print(f"Erro ao processar o arquivo '{file_name}': {e}")
            continue
        finally:
            # Apenas feche o file_obj se ele foi aberto por nós (do diretório local ou de um caminho str)
            # Objetos BytesIO passados como lista não precisam ser fechados aqui.
            if isinstance(pdf_sources, str) or (isinstance(pdf_sources, (list, tuple)) and all(isinstance(item, str) for item in pdf_sources)):
                file_obj.close()


    text_splitter_general = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks: List[Document] = text_splitter_general.split_documents(all_documents_for_indexing)

    print(f"✅ PDFs processados. Gerados {len(chunks)} chunks.")
    return chunks