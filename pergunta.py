import os
from langchain_chroma import Chroma
from embedding import HuggingFaceEmbedding
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Carrega a chave da API de uma variável de ambiente por segurança
client = OpenAI(
    api_key=os.getenv("GITHUB_API_KEY"), # Garanta que GITHUB_API_KEY está no seu arquivo .env
    base_url="https://models.github.ai/inference"
)
model_name = "openai/gpt-4o"

embedding = HuggingFaceEmbedding()

vectorstore = Chroma(
    embedding_function=embedding,
    persist_directory="chroma"
)

def perguntar_openai(pergunta, contexto):
    """
    Envia a pergunta e o contexto para o modelo OpenAI e retorna a resposta.
    """
    prompt = (f"Com base nos dados abaixo, responda com exatidão à seguinte pergunta: '{pergunta}'. "
            "Se a pergunta for sobre uma data de evento, procure pela data correspondente à etapa mencionada e forneça a data diretamente. "
            "Se a data for um período, forneça o período completo. Não adicione informações que não estão no contexto.\n\n"
            f"{contexto}\n\nPergunta: {pergunta}") # Repete a pergunta no final para reforçar
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Você é um assistente útil e preciso. Responda apenas com informações contidas no contexto fornecido."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Pergunta que será feita ao modelo
pergunta = "Qual é a data do Lançamento da Chamada Pública FAPESC"

# ESTRATÉGIA DE RECUPERAÇÃO HÍBRIDA:
# Busca documentos de duas formas e os combina para aumentar a chance de incluir o chunk do cronograma.

# Busca 1: Documentos mais similares à pergunta geral
docs_similar = vectorstore.similarity_search(pergunta, k=15) # Recupera os 15 documentos mais similares

# Busca 2: Documentos específicos de cronograma
# Utiliza uma query mais direcionada e filtra pelo tipo de metadado 'cronograma_principal'
docs_cronograma_principal = vectorstore.similarity_search(
    "CRONOGRAMA DE EVENTOS E DATAS IMPORTANTES: " + pergunta,
    k=5, # Pega os 5 mais relevantes desta busca específica
    filter={"type": "cronograma_principal"} # Filtra para garantir que seja o chunk do cronograma
)

# Combina e deduplica os documentos recuperados
all_docs = []
seen_content = set() # Usado para evitar adicionar o mesmo chunk mais de uma vez

# Adiciona primeiro os documentos de cronograma principal (dando prioridade a eles)
for doc in docs_cronograma_principal:
    if doc.page_content not in seen_content:
        all_docs.append(doc)
        seen_content.add(doc.page_content)

# Adiciona os documentos gerais, evitando duplicatas
for doc in docs_similar:
    if doc.page_content not in seen_content:
        all_docs.append(doc)
        seen_content.add(doc.page_content)

# A lista 'docs' agora contém os documentos combinados, priorizando os de cronograma
docs = all_docs
print(f"Documentos encontrados (após combinação e deduplicação): {len(docs)}")

# Limite de caracteres para o contexto enviado ao modelo (aproximadamente 8k tokens)
MAX_CHARS = 25000 

contexto = ""
for i, doc in enumerate(docs):
    # Adiciona o chunk ao contexto se houver espaço
    # +2 para o '\n\n' que é adicionado entre os chunks
    if len(contexto) + len(doc.page_content) + 2 <= MAX_CHARS: 
        contexto += doc.page_content + "\n\n"
        # Não é mais necessário imprimir o status de inclusão de cada documento aqui
    else:
        # Se o limite de caracteres for atingido, para de adicionar documentos
        print(f"⚠️ Limite de caracteres do contexto atingido. Pulando documentos restantes.")
        break

print("\n📌 Resposta do modelo:\n")
# Chama a função para perguntar ao modelo e imprime a resposta
print(perguntar_openai(pergunta, contexto))