# rag.py
import os
import torch
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document # Importar Document para tipagem
from transformers import AutoTokenizer, AutoModel

# Importar tipos específicos para as mensagens do OpenAI API
from typing import List, Dict, Any
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Sua Classe HuggingFaceEmbedding (MANTIDA NO embedding.py) ---
# Certifique-se de que 'embedding.py' existe e contém a classe HuggingFaceEmbedding
from embedding import HuggingFaceEmbedding 

# --- Configuração do Cliente OpenAI e Modelo para RAG ---
from typing import Optional

def perguntar_openai(pergunta: str, contexto: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str: # Adicionado chat_history
    """
    Envia a pergunta e o contexto para o modelo OpenRouter e retorna a resposta.
    Agora inclui o histórico de chat para contexto.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    model_name_rag = "google/gemini-2.0-flash-lite-001" 

    extra_headers_config = {
        "X-Title": "Chatbot de Editais TCC",
    }
    
    # Construção das mensagens para a API com tipagem mais explícita
    messages_for_llm: List[ChatCompletionMessageParam] = [
        # Usar os tipos específicos para o role="system"
        ChatCompletionSystemMessageParam(
            role="system",
            content="Você é um assistente útil e preciso. Responda apenas com informações contidas no contexto fornecido. Quando solicitado a listar editais, **LISTE CADA EDITAL INDIVIDUALMENTE, SEJA EXAUSTIVO e inclua TÍTULO, AGÊNCIA, PRAZO FINAL e URL para CADA edital relevante**. Se a URL não for um link direto para o PDF, forneça a URL da página do edital."
        ),
    ]

    # Adicionar histórico de chat, se houver
    if chat_history:
        for msg in chat_history:
            # Excluir a mensagem inicial do assistente para não duplicar instruções
            # Isso é útil se você tiver uma mensagem de boas-vindas fixa no início da sessão
            if msg["role"] == "assistant" and msg["content"].startswith("Olá! Como posso te ajudar"):
                continue
            
            # Converter mensagens do histórico para tipos específicos
            if msg["role"] == "user":
                messages_for_llm.append(ChatCompletionUserMessageParam(role="user", content=msg["content"]))
            elif msg["role"] == "assistant":
                messages_for_llm.append(ChatCompletionAssistantMessageParam(role="assistant", content=msg["content"]))
            # Outros roles (system, tool) não viriam do histórico de chat do usuário
    
    # Adicionar a pergunta atual e o contexto recuperado
    messages_for_llm.append(
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Contexto para a pergunta: {contexto}\n\nPergunta: {pergunta}"
        )
    )
    
    response = client.chat.completions.create(
        model=model_name_rag,
        messages=messages_for_llm, # Passa as mensagens construídas
        extra_headers=extra_headers_config,
    )
    return response.choices[0].message.content if response.choices[0].message.content else ""

# --- Lógica de Recuperação Híbrida (MANTIDA COMO ESTÁ, apenas para referência) ---
def retrieve_documents(pergunta: str, vectorstore_instance: Chroma) -> list[Document]:
    """
    Recupera documentos da vector store usando estratégia híbrida.
    Recebe a instância da vectorstore como argumento.
    """
    # Aumentar k para recuperar mais documentos (ex: de 15 para 30 ou 50)
    docs_similar = vectorstore_instance.similarity_search(pergunta, k=100) 

    # Manter k para cronograma, mas também pode aumentar se necessário
    docs_cronograma_principal = vectorstore_instance.similarity_search(
        "CRONOGRAMA DE EVENTOS E DATAS IMPORTANTES: " + pergunta,
        k=20, # Aumentado para 10
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