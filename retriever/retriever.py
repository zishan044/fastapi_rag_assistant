import faiss
from typing import List, Dict
from langchain_classic.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from pathlib import Path
from langchain_core.runnables import RunnableLambda


class RetrievalPipeline:
    def __init__(self, embedding_model: OllamaEmbeddings, vectorstore_path: str | Path):
        self.embedding_model = embedding_model
        self.vectorstore_path = Path(vectorstore_path)
        self.vectorstore: FAISS | None = None
        self.bm25_retriever: BM25Retriever | None = None

    def build_vectorstore(self, documents: List[Document]):
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model,
        )
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.vectorstore_path)

    def load_vectorstore(self):
        self.vectorstore = FAISS.load_local(
            self.vectorstore_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True,
        )

    def create_bm25_retriever(self, documents: List[Document]):
        def simple_tokenize(text: str):
            return text.split()

        texts = [doc.page_content for doc in documents]
        self.bm25_retriever = BM25Retriever.from_texts(
            texts=texts,
            k=2,
            preprocess_func=simple_tokenize,
        )

    def _convert_to_lc_document(self, doc: Document):
        from langchain_core.documents import Document as LCDocument
        return LCDocument(
            page_content=doc.page_content,
            metadata=doc.metadata
        )

    def as_retriever(self, k: int = 4, search_type: str = "hybrid"):
        """RunnableLambda - 100% LCEL compatible, no Pydantic issues!"""
        
        def retrieve(query: str):
            """Core retrieval logic - captures self, k, search_type"""
            if search_type == "dense" and self.vectorstore:
                docs = self.vectorstore.similarity_search(query, k=k)
            elif search_type == "sparse" and self.bm25_retriever:
                docs = self.bm25_retriever.invoke(query)[:k]
                docs = [self._convert_to_lc_document(doc) for doc in docs]
            else:  # hybrid
                docs = self.hybrid_search(query, k=k)
                docs = [self._convert_to_lc_document(doc) for doc in docs]
            return docs
        
        return RunnableLambda(retrieve)

    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        results: Dict[str, Document] = {}
        
        # Dense retrieval
        if self.vectorstore:
            try:
                dense_docs = self.vectorstore.similarity_search(query, k=k)
                for doc in dense_docs:
                    key = doc.metadata.get("file_name") or doc.metadata.get("source", f"doc_{id(doc)}")
                    results[key] = doc
            except Exception:
                pass

        # Sparse retrieval
        if self.bm25_retriever:
            try:
                sparse_docs = self.bm25_retriever.invoke(query)
                for doc in sparse_docs[:k]:
                    key = doc.metadata.get("file_name") or doc.metadata.get("source", f"doc_{id(doc)}")
                    results[key] = doc
            except Exception:
                pass

        return list(results.values())[:k]

    def vectorstore_exists(self) -> bool:
        return (self.vectorstore_path / "index.faiss").exists()