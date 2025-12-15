from pathlib import Path

from ingestion.ingest import IngestionPipleline
from retriever.retriever import RetrievalPipeline
from chains.chains import RAGChain

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings


def main():
    docs_path = Path("./rag_docs")
    vectorstore_path = Path("./vectorstore")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    ingestion = IngestionPipleline(
        dir_path=docs_path,
        loader=None,
        splitter=splitter,
    )
    documents = ingestion.chunk_documents()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever = RetrievalPipeline(
        embedding_model=embeddings,
        vectorstore_path=vectorstore_path,
    )
    if retriever.vectorstore_exists():
        retriever.load_vectorstore()
    else:
        retriever.build_vectorstore(documents)

    retriever.create_bm25_retriever(documents)

    llm = ChatOllama(model="mistral", temperature=0)
    rag = RAGChain(retriever=retriever, llm=llm)

    query = "How do I use dependencies in FastAPI?"
    answer = rag.answer(query)
    print(answer)


if __name__ == "__main__":
    main()
