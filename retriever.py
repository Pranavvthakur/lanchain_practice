import os
from pymilvus import Collection
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

def get_query_embedding(query: str):
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small" 
    )
    return response.data[0].embedding

def search_collection(collection: Collection, query_vector, top_k=10):
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }

    collection.load()
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )

    reranked = sorted(results[0], key=lambda x: x.distance)
    return [res.entity.get("text") for res in reranked]
