from pathlib import Path
from typing import List
from langchain_classic.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class IngestionPipleline:
    def __init__(self, dir_path: str | Path, loader: TextLoader, splitter: RecursiveCharacterTextSplitter):
        self.dir_path = Path(dir_path)
        self.loader = loader
        self.splitter = splitter

    def load_documents(self) -> List[Document]:
        docs = []

        for md_file in self.dir_path.rglob("*.md"):

            loader = TextLoader(str(md_file))
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata.update({
                    "file_name": md_file.name,
                    "topic": md_file.parent.name,
                })
                docs.append(doc)
        return docs

    def chunk_documents(self) -> List[Document]:
        docs = self.load_documents()
        splits = self.splitter.split_documents(docs)
        return splits
