from fastapi import FastAPI
from qdrant_client import QdrantClient
from model import tokenizer, client
from sentence_transformers import SentenceTransformer

class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.model = tokenizer
        self.qdrant_client = QdrantClient("http://localhost:6333")


def search(self, text: str):
        vector = self.model.encode(text).tolist()

        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            limit=5 
        )

        payloads = [hit.payload for hit in search_result]
        return payloads


app = FastAPI()

neural_searcher = NeuralSearcher(collection_name='startups')

@app.get("/api/search")
def search_startup(q: str):
    return {
        "result": neural_searcher.search(text=q)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.0", port=8000)
