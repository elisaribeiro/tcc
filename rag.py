# rag_core.py
import os
import torch
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document # Importar Document para tipagem
from transformers import AutoTokenizer, AutoModel

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Sua Classe HuggingFaceEmbedding (MANTIDA NO embedding.py) ---
# Importe-a de 'embedding'
from embedding import HuggingFaceEmbedding

# --- Configuração do Cliente OpenAI ---
# Garanta que OPENAI_API_KEY está no seu .env
# OU GITHUB_API_KEY se for o caso, mas use OPENAI_API_KEY para ChatOpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Usando OPENAI_API_KEY
model_name_rag = "gpt-4o"

# --- Função de Perguntar ao Modelo ---
def perguntar_openai(pergunta: str, contexto: str) -> str:
    """
    Envia a pergunta e o contexto para o modelo OpenAI e retorna a resposta.
    """
    prompt = (f"Com base nos dados abaixo, responda com exatidão à seguinte pergunta: '{pergunta}'. "
            "Se a pergunta for sobre uma data de evento, procure pela data correspondente à etapa mencionada e forneça a data diretamente. "
            "Se a data for um período, forneça o período completo. Não adicione informações que não estão no contexto.\n\n"
            f"{contexto}\n\nPergunta: {pergunta}")
    
    response = client.chat.completions.create(
        model=model_name_rag,
        messages=[
            {"role": "system", "content": "Você é um assistente útil e preciso. Responda apenas com informações contidas no contexto fornecido."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content or ""

# --- Lógica de Recuperação Híbrida ---
def retrieve_documents(pergunta: str, vectorstore_instance: Chroma) -> list[Document]:
    """
    Recupera documentos da vector store usando estratégia híbrida.
    Recebe a instância da vectorstore como argumento.
    """
    # Busca 1: Documentos mais similares à pergunta geral
    docs_similar = vectorstore_instance.similarity_search(pergunta, k=15)

    # Busca 2: Documentos específicos de cronograma
    docs_cronograma_principal = vectorstore_instance.similarity_search(
        "CRONOGRAMA DE EVENTOS E DATAS IMPORTANTES: " + pergunta,
        k=5,
        filter={"type": "cronograma_principal"}
    )

    # Combina e deduplica os documentos recuperados
    all_docs = []
    seen_content = set()

    for doc in docs_cronograma_principal:
        if doc.page_content not in seen_content:
            all_docs.append(doc)
            seen_content.add(doc.page_content)

    for doc in docs_similar:
        if doc.page_content not in seen_content:
            all_docs.append(doc)
            seen_content.add(doc.page_content)

    return all_docs

# Removido o bloco if __name__ == "__main__": original