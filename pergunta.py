# perguntar.py
import os
from langchain_chroma import Chroma
from embedding import HuggingFaceEmbedding
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key="",
    base_url="https://models.github.ai/inference"
)
model_name = "openai/gpt-4.1-mini"

embedding = HuggingFaceEmbedding()

vectorstore = Chroma(
    embedding_function=embedding,
    persist_directory="chroma"
)

def perguntar_openai(pergunta, contexto):
    prompt = f"Responda com base nos documentos abaixo:\n{contexto}\nPergunta: {pergunta}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Você é um assistente útil."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

pergunta = "Em forma de tabela Markdown, informe o público alvo dos editais"
docs = vectorstore.similarity_search(pergunta, k=50)
print(f"Documentos encontrados: {len(docs)}")

MAX_CHARS = 12000  # Limite seguro para o modelo gpt-4.1-mini (~8k tokens)

contexto = ""
for doc in docs:
    if len(contexto) + len(doc.page_content) > MAX_CHARS:
        break
    contexto += doc.page_content + "\n\n"

resposta = perguntar_openai(pergunta, contexto)

print("Resposta:\n", resposta)