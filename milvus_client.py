import os
from typing import List
from PyPDF2 import PdfReader
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from openai import OpenAI
from dotenv import load_dotenv

# ENV & OPENAI SETUP
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MILVUS CONNECTION
def connect_milvus():
    connections.connect("default", host="localhost", port="19530")


# MILVUS COLLECTION
def create_collection(collection_name="pdf_chunks", dim=1536):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
    ]
    schema = CollectionSchema(fields, description="PDF Chunk Embeddings")
    collection = Collection(collection_name, schema)
    return collection


def create_index(collection):
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)


def load_collection(collection_name="pdf_chunks"):
    return Collection(collection_name)


def insert_data(collection, embeddings, texts):
    collection.insert([embeddings, texts])
    collection.flush()


# PDF HANDLING & CHUNKING 
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def split_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# EMBEDDING + INSERT WORKFLOW
from pymilvus import utility
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def process_pdf_and_store(pdf_path: str, collection_name="pdf_chunks", dim=1536):
    # Connect to Milvus
    connect_milvus()

    # Always drop the existing collection (if exists)
    if utility.has_collection(collection_name):
        print(f" Dropping existing collection '{collection_name}'...")
        collection = load_collection(collection_name)
        collection.drop()
        print(f" Dropped collection '{collection_name}'")

    # Create a new collection with the specified dimension
    print(f" Creating new collection '{collection_name}' with dim={dim}...")
    collection = create_collection(collection_name, dim=dim)
    create_index(collection)

    # Process PDF
    print(f" Reading PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)

    # Generate embeddings
    print(f" Generating embeddings using OpenAI text-embedding-3-small")
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )
    embeddings = [item.embedding for item in response.data]

    # Insert data into Milvus
    print(" Inserting into Milvus...")
    insert_data(collection, embeddings, chunks)
    print(" Done inserting into Milvus")

