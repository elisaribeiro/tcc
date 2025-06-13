# pdf_indexer.py
import os
import pdfplumber
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# A classe HuggingFaceEmbedding será importada de 'embedding.py' pelo orquestrador
# e a instância do embedding será passada para esta função se ela for criar/adicionar Chroma.
# No entanto, como o Chroma será inicializado no orquestrador, esta função só precisa retornar os Documents.

def safe_str_strip(value):
    if value is None:
        return ""
    return str(value).strip()

def clean_cell(cell):
    if cell is None:
        return ""
    return str(cell).replace("\n", " ").replace("\r", " ").strip()

def process_pdfs_into_documents(pdf_dir_path: str) -> list[Document]:
    """
    Processa arquivos PDF de um diretório, extrai texto e tabelas (cronogramas),
    e retorna uma lista de objetos Document para indexação.
    """
    print(f"\n--- Iniciando processamento de PDFs do diretório: {pdf_dir_path} ---")
    pdf_paths = [os.path.join(pdf_dir_path, f) for f in os.listdir(pdf_dir_path) if f.endswith(".pdf")]

    all_documents_for_indexing = []

    for pdf_path in pdf_paths:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()

                    structured_cronograma_content = ""
                    found_main_cronograma_table = False
                    
                    for table_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue
                        
                        headers_row = table[0]
                        cleaned_headers = [clean_cell(h) for h in headers_row if h is not None]
                        
                        if "ETAPAS" in cleaned_headers and "DATAS" in cleaned_headers:
                            found_main_cronograma_table = True
                            structured_cronograma_content += "CRONOGRAMA DE EVENTOS E DATAS IMPORTANTES:\n"
                            
                            for row_idx, row in enumerate(table):
                                if row_idx == 0:
                                    continue
                                if len(row) < 2 or not row[0] or not row[1]:
                                    continue

                                item = clean_cell(row[0])
                                value = clean_cell(row[1])
                                
                                structured_cronograma_content += f"O evento '{item}' tem a data ou período de {value}.\n"
                    
                    if found_main_cronograma_table and structured_cronograma_content:
                        all_documents_for_indexing.append(Document(page_content=structured_cronograma_content, metadata={"source": pdf_path, "page": i + 1, "type": "cronograma_principal"}))
                    elif text and not found_main_cronograma_table:
                        all_documents_for_indexing.append(Document(page_content=text, metadata={"source": pdf_path, "page": i + 1, "type": "page_text"}))
        except Exception as e:
            print(f"Erro ao processar PDF {pdf_path}: {e}")
            continue

    text_splitter_general = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = text_splitter_general.split_documents(all_documents_for_indexing)

    print(f"✅ PDFs processados. Gerados {len(chunks)} chunks.")
    return chunks

# Removido o bloco if __name__ == "__main__": original