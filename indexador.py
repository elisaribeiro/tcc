import os
from langchain_chroma import Chroma
from embedding import HuggingFaceEmbedding
import pdfplumber
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbedding()

def safe_str_strip(value):
    if value is None:
        return ""
    return str(value).strip()

def clean_cell(cell):
    if cell is None:
        return ""
    return str(cell).replace("\n", " ").replace("\r", " ").strip()

pdf_dir = r"/Users/elisaribeiro/tcc/pdfs"
pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

all_documents_for_indexing = []

for pdf_path in pdf_paths:
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
                else:
                    # Para outras tabelas não identificadas como cronograma principal
                    # Você pode adicionar essas linhas em um Document de tipo 'tabela_geral'
                    # ou ignorá-las para manter o índice mais focado.
                    # Por simplicidade, este exemplo as ignora se não forem o cronograma principal.
                    pass # Se você quiser indexar tabelas não-cronograma, adicione a lógica aqui.


            if found_main_cronograma_table and structured_cronograma_content:
                all_documents_for_indexing.append(Document(page_content=structured_cronograma_content, metadata={"source": pdf_path, "page": i + 1, "type": "cronograma_principal"}))
            
            elif text and not found_main_cronograma_table:
                all_documents_for_indexing.append(Document(page_content=text, metadata={"source": pdf_path, "page": i + 1, "type": "page_text"}))


print("Iniciando indexador...")

text_splitter_general = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

chunks = text_splitter_general.split_documents(all_documents_for_indexing)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="chroma"
)

print(f"✅ Banco vetorial criado com {len(chunks)} chunks.")