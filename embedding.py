from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class HuggingFaceEmbedding(Embeddings):
    print("Iniciando embedding...")
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def embed_documents(self, texts):
        embeddings = []
        batch_size = 32  # ajusta se quiser

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
                if torch.cuda.is_available():
                    encoded_input = {k: v.to("cuda") for k, v in encoded_input.items()}

                model_output = self.model(**encoded_input)
                # Mean pooling dos tokens para gerar embedding
                embeddings_batch = self.mean_pooling(model_output, encoded_input['attention_mask'])
                embeddings_batch = embeddings_batch.cpu().numpy()
                embeddings.extend(embeddings_batch)

        print("Total de embeddings gerados:", len(embeddings))
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
