from typing import List
from langchain_classic.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


class RAGChain:
    def __init__(self, retriever, llm: ChatOllama):
        self.retriever = retriever
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """
You are a FastAPI documentation assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer (with citations):
"""
        )

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents"""
        return self.retriever.hybrid_search(query)

    def generate(self, query: str, docs: List[Document]) -> str:
        """Generate answer from retrieved docs"""
        context = "\n\n".join(doc.page_content for doc in docs)
        chain = self.prompt | self.llm
        return chain.invoke({"context": context, "question": query})

    def answer(self, query: str) -> str:
        """End-to-end RAG"""
        docs = self.retrieve(query)
        return self.generate(query, docs)
