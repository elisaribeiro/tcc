from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class HuggingFaceEmbedding(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", device=None):
        print("Iniciando embedding...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        # Detecta GPU automaticamente
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)

    def embed_documents(self, texts):
        print(">>> embed_documents recebeu:", type(texts), len(texts) if hasattr(texts, "__len__") else "??")
        """
        Gera embeddings para uma lista de textos (um embedding por item da lista).
        """
        if isinstance(texts, str):  # caso passe só uma string por engano
            texts = [texts]

        embeddings = []
        batch_size = 32  # ajuste conforme memória

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                model_output = self.model(**encoded_input)

                # Mean pooling
                embeddings_batch = self.mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )

                # Normalizar (para similaridade coseno funcionar melhor)
                embeddings_batch = torch.nn.functional.normalize(embeddings_batch, p=2, dim=1)

                # Converte para lista de listas de floats
                embeddings_batch = embeddings_batch.cpu().numpy().tolist()
                embeddings.extend(embeddings_batch)

        print("Total de embeddings gerados:", len(embeddings))
        return embeddings

    def embed_query(self, text):
        """
        Gera embedding para uma única query.
        """
        return self.embed_documents([text])[0]

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Faz mean pooling sobre a saída do modelo, considerando a máscara de atenção.
        """
        token_embeddings = model_output.last_hidden_state  # (batch, seq_len, hidden_dim)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
