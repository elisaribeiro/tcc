# rag_core.py
import os
import torch
import numpy as np
from openai import OpenAI # Cliente da OpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModel

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Sua Classe HuggingFaceEmbedding (MANTIDA NO embedding.py) ---
from embedding import HuggingFaceEmbedding

# --- Configuração do Cliente OpenAI e Modelo para RAG ---
# ESTAS VARIÁVEIS AGORA SERÃO DEFINIDAS DENTRO DA FUNÇÃO perguntar_openai,
# ou você pode definir um cliente global aqui para OpenRouter.
# Para manter a flexibilidade e garantir que a chave esteja sempre atualizada,
# vamos inicializar o cliente DENTRO de perguntar_openai.

# --- Função de Perguntar ao Modelo (AJUSTADA PARA OPENROUTER) ---
def perguntar_openai(pergunta: str, contexto: str) -> str:
    """
    Envia a pergunta e o contexto para o modelo OpenRouter e retorna a resposta.
    O cliente OpenAI é configurado para usar o OpenRouter.
    """
    # Inicializa o cliente OpenAI para usar OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", # Endpoint do OpenRouter
        api_key=os.getenv("OPENROUTER_API_KEY"), # Usa a chave do OpenRouter
    )

    # O mesmo modelo que você usa no browser_agent
    model_name_rag = "google/gemini-2.0-flash-lite-001" 

    # Opcional: Adicionar cabeçalhos extra_headers para OpenRouter (para rankings)
    extra_headers_config = {
        "HTTP-Referer": "https://seu-app-de-editais.com", # Substitua pela URL do seu app
        "X-Title": "Chatbot de Editais TCC", # Substitua pelo nome do seu app
    }
    
    prompt = (f"Com base nos dados abaixo, responda com exatidão à seguinte pergunta: '{pergunta}'. "
            "Se a pergunta for sobre uma data de evento, procure pela data correspondente à etapa mencionada e forneça a data diretamente. "
            "Se a data for um período, forneça o período completo. Não adicione informações que não estão no contexto.\n\n"
            f"{contexto}\n\nPergunta: {pergunta}")
    
    response = client.chat.completions.create(
        model=model_name_rag,
        messages=[
            {"role": "system", "content": "Você é um assistente útil e preciso. Responda apenas com informações contidas no contexto fornecido."},
            {"role": "user", "content": prompt}
        ],
        extra_headers=extra_headers_config, # Adiciona os cabeçalhos extras
    )
    # Garante que o retorno seja uma string, mesmo que vazio
    return response.choices[0].message.content if response.choices[0].message.content else ""

# --- Lógica de Recuperação Híbrida ---
def retrieve_documents(pergunta: str, vectorstore_instance: Chroma) -> list[Document]:
    """
    Recupera documentos da vector store usando estratégia híbrida.
    Recebe a instância da vectorstore como argumento.
    """
    docs_similar = vectorstore_instance.similarity_search(pergunta, k=15)
    docs_cronograma_principal = vectorstore_instance.similarity_search(
        "CRONOGRAMA DE EVENTOS E DATAS IMPORTANTES: " + pergunta,
        k=5,
        filter={"type": "cronograma_principal"}
    )

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