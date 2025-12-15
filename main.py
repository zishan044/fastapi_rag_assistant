from pathlib import Path

from ingestion.ingest import IngestionPipleline
from retriever.retriever import RetrievalPipeline

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    """Format retrieved docs for the prompt"""
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    docs_path = Path("./rag_docs")
    vectorstore_path = Path("./vectorstore")

    # Ingestion
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    ingestion = IngestionPipleline(dir_path=docs_path, loader=None, splitter=splitter)
    documents = ingestion.chunk_documents()

    # Embeddings & VectorStore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever = RetrievalPipeline(embedding_model=embeddings, vectorstore_path=vectorstore_path)

    if retriever.vectorstore_exists():
        retriever.load_vectorstore()
    else:
        retriever.build_vectorstore(documents)

    # Create BM25 retriever for hybrid search
    retriever.create_bm25_retriever(documents)

    # Get LCEL-compatible retriever
    doc_retriever = retriever.as_retriever(k=4)

    # LLM
    llm = ChatOllama(model="mistral", temperature=0)

    # RAG Prompt - uses {context} and {input} for LCEL
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a FastAPI documentation assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context, reply "I don't know."
Cite sources where possible.

Context: {context}"""),
        ("human", "{input}")
    ])

    # LLM chain
    llm_chain = rag_prompt | llm | StrOutputParser()

    # Modern LCEL RAG chain - replaces deprecated create_*_chain functions
    rag_chain = (
        {"context": doc_retriever | format_docs, "input": RunnablePassthrough()}
        | llm_chain
    )

    # Test query
    query = "How do I use dependencies in FastAPI?"
    print(f"\n🔍 Query: {query}")
    
    result = rag_chain.invoke({"input": query})

    print("\n=== ANSWER ===")
    print(result)


if __name__ == "__main__":
    main()