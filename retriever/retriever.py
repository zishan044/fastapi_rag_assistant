import faiss
from typing import List, Dict
from langchain_classic.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
import pickle
from pathlib import Path

class RetrievalPipeline:
    def __init__(self, embedding_model: OllamaEmbeddings, vectorstore_path: str | Path):
        """
        Initialize the retrieval pipeline with an embedding model and path to store/load FAISS index.
        """
        self.embedding_model = embedding_model
        self.vectorstore_path = Path(vectorstore_path)
        self.vectorstore: FAISS | None = None
        self.bm25_retriever: BM25Retriever | None = None

    def build_vectorstore(self, documents: List[Document]):
        """
        Create FAISS vectorstore from chunked documents and save it to disk.
        """
        embedding_dim = 768
        index = faiss.IndexFlatL2(embedding_dim)

        vector_store = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        vector_store.add_documents(documents=documents)
        self.vectorstore = vector_store

        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.vectorstore_path / "faiss.index"))
        with open(self.vectorstore_path / "docstore.pkl", "wb") as f:
            pickle.dump(vector_store.docstore, f)

    def load_vectorstore(self):
        """
        Load FAISS vectorstore from disk.
        """
        index_file = self.vectorstore_path / "faiss.index"
        docstore_file = self.vectorstore_path / "docstore.pkl"

        if not index_file.exists() or not docstore_file.exists():
            raise FileNotFoundError("Vectorstore files not found. Build vectorstore first.")

        index = faiss.read_index(str(index_file))
        with open(docstore_file, "rb") as f:
            docstore = pickle.load(f)

        self.vectorstore = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id={},
        )

    def create_bm25_retriever(self, documents: List[Document]):
        """
        Create a BM25 retriever for sparse retrieval using a simple tokenizer (no NLTK).
        """
        def simple_tokenize(text: str):
            return text.split()  # simple whitespace tokenizer

        texts = [doc.page_content for doc in documents]

        from langchain_community.retrievers import BM25Retriever

        self.bm25_retriever = BM25Retriever.from_texts(
            texts=texts,
            k=2,
            preprocess_func=simple_tokenize,
        )



    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform hybrid retrieval: combine BM25 and FAISS results.
        """
        results: Dict[str, Document] = {}

        if self.vectorstore:
            dense_docs = self.vectorstore.similarity_search(query, k=k)
            for doc in dense_docs:
                results[doc.metadata.get("file_name", str(doc))] = doc

        if self.bm25_retriever:
            sparse_docs = self.bm25_retriever.invoke(query)
            for doc in sparse_docs:
                results[doc.metadata.get("file_name", str(doc))] = doc

        return list(results.values())[:k]
