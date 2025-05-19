from milvus_client import connect_milvus, load_collection
from retriever import get_query_embedding, search_collection
from rag_chain import get_answer

# Step 1: Connect to Milvus
connect_milvus()
collection = load_collection()

    # Step 2: Define the query
query = "How many sibilings did the seagull have?"
    
    # Step 3: Get query vector
query_vector = get_query_embedding(query)
    
    # Step 4: Search Milvus and get top chunks
top_chunks = search_collection(collection, query_vector, top_k=10)
context = "\n\n".join(top_chunks)

    # Step 5: Run RAG pipeline
answer = get_answer(context, query)

print("Final Answer:\n", answer)