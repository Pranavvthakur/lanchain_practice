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

