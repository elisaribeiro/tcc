# indexador.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from embedding import HuggingFaceEmbedding

pdf_dir = r"C:\TCC\LLM\pdfs"
pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

all_pages = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    all_pages.extend(loader.load())

print("Iniciando indexador...")
embedding = HuggingFaceEmbedding()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
chunks = text_splitter.split_documents(all_pages)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="chroma"
)

print(f"✅ Banco vetorial criado com {len(chunks)} chunks.")