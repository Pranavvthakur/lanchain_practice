# from sentence_transformers import SentenceTransformer
# from pymilvus import Collection, connections, CollectionSchema
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# import os
# import numpy as np

# # ------------------ Load OpenAI API Key ------------------ #
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

# # ------------------ Initialize Milvus ------------------ #
# connections.connect("default", host="localhost", port="19530")
# schema = CollectionSchema(fields, description="PDF Chunk Embeddings")
# collection = Collection("pdf_chunks")
# collection.load()

# # ------------------ Encode Question ------------------ #
# query = "how many brother and sissters seagull having?"
# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# query_embedding = embedder.encode(query).tolist()

# # ------------------ Search in Milvus ------------------ #
# search_params = {
#     "metric_type": "L2",
#     "params": {"nprobe": 10}
# }
# search_result = collection.search(
#     data=[query_embedding],
#     anns_field="embedding",         # Replace with your vector field name
#     param=search_params,
#     limit=10,
#     output_fields=["text"]          # Replace with your text field name
# )
# question = query

# # ------------------ Rerank by Distance ------------------ #
# results = search_result[0]
# reranked = sorted(results, key=lambda x: x.distance)[:3]  # lowest distance = most similar
# top_chunks = [hit.entity.get("text") for hit in reranked]

# # ------------------ Create Prompt ------------------ #
# context = "\n\n".join(top_chunks)
# template = PromptTemplate.from_template(
#     """You are a helpful assistant.

# Use the following context to answer the query. Only return the specific answer, be brief, and do not repeat the context. Give answer if only the answer is avaiilable in context else say "Sorry, answer for this question is not present in context".

# Context:
# {context}

# Question: {question}
# Answer:"""
# )

# # ------------------ Get Final Answer ------------------ #
# llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
# output_parser = StrOutputParser()

# prompt = template.format(context=context, question=query)
# response = llm.invoke(prompt)
# answer = response.content

# print("\n--- Answer ---")
# print(answer)
# print(question)
# print(context)


import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    """You are a strict assistant.

Only use the context below to answer the question. If the answer is not explicitly mentioned in the context, reply exactly with:
"Sorry, the answer is not available in the context."

Do not make up or infer answers. Do not guess. Do not use prior knowledge.

Context:
{context}

Question: {question}

Answer:"""
)

llm = ChatOpenAI(model="gpt-4", temperature=0.0)
parser = StrOutputParser()
rag_chain = template | llm | parser

def get_answer(context, question):
    return rag_chain.invoke({"context": context, "question": question})

