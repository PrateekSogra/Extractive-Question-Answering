import torch
from datasets import load_dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm.auto import tqdm
from typing import List
import pandas as pd
import openai
from openai import OpenAI

df = pd.read_csv('/content/drive/MyDrive/BigBasketProducts.csv')
tokenizer = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

descriptions = df['description'].astype(str).to_list()

client = QdrantClient(":memory:")

collection_name = "BigBasketProducts"

batch_size = 32  

collections = client.get_collections()

if collection_name not in [c.name for c in collections.collections]:
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
        ),
    )
collections = client.get_collections()


for index in tqdm(range(0, len(df), batch_size)):
    i_end = min(index + batch_size, len(df))
    batch = df.iloc[index:i_end]  
    emb = tokenizer.encode(batch["description"].tolist()).tolist() 
    meta = batch.to_dict(orient="records") 
    ids = list(range(index, i_end)) 

    client.upsert(
        collection_name=collection_name,
        points=models.Batch(ids=ids, vectors=emb, payloads=meta),
    )

collection_vector_count = client.get_collection(collection_name=collection_name).vectors_count
print(f"Vector count in collection: {collection_vector_count}")
assert collection_vector_count == len(df)

def get_relevant_details(question: str, top_k: int) -> List[str]:
    encoded_query = tokenizer.encode(question).tolist() 

    result = client.search(
        collection_name=collection_name,
        query_vector=encoded_query,
        limit=top_k,
    ) 
    return result


openai.api_key = OPENAI-API-KEY

response = openai.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt="Write a tagline for an ice cream shop."
)
