from fastapi import FastAPI

from neural_searcher import NeuralSearcher

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.model = chat_completion
        self.qdrant_client = QdrantClient("http://localhost:6333")

app = FastAPI()

neural_searcher = NeuralSearcher(collection_name='startups')

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

@app.get("/api/search")
def search_startup(q: str):
    return {
        "result": neural_searcher.search(text=q)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
