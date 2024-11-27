import os
from typing import Iterator, Union
from tempfile import TemporaryDirectory

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from docling.document_converter import DocumentConverter


class DoclingPDFLoader(BaseLoader):
    def __init__(self, file_path: Union[str, list[str]]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)

class DB:
    def __init__(self, path: str, embed_model: str,milvus_uri:str):
        """
        Args:
            path (str): Either a single PDF file path or a folder path containing multiple PDFs.
            embed_model (str): HuggingFace embedding model name.
        """
        self.path = path
        self.embed_model = embed_model
        self.milvus_uri = milvus_uri 
        self.vectorstore = None
        self.embedding= HuggingFaceEmbeddings(model_name=self.embed_model)

    def process_documents(self):
        # Check if the Milvus database directory exists
        if os.path.exists(self.milvus_uri):
            print("Loading existing vectorstore...")
            self.vectorstore = self._load_existing_vectorstore()
            return

        print("Creating new vectorstore...")

        # Load documents
        loader = DoclingPDFLoader(file_path=self._get_files(self.path))
        documents = list(loader.lazy_load())

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        splits = []
        for doc in documents:
            # Wrap each split in an LCDocument with metadata
            chunks = text_splitter.split_text(doc.page_content)
            splits.extend(
                [LCDocument(page_content=chunk, metadata=doc.metadata) for chunk in chunks]
            )

        # Store in Milvus vector database
        self.vectorstore = Milvus.from_documents(
            splits,
            self.embedding,
            connection_args={"uri": f"{self.milvus_uri}"},
            drop_old=True,
        )
        print("Vectorstore created and stored.")

    def _load_existing_vectorstore(self):
        """
        Load an existing Milvus vectorstore using the same connection arguments.

        Returns:
            Milvus: The loaded vectorstore.
        """
        print(f"Loading vectorstore from {self.milvus_uri}...")
        return Milvus(
            self.embedding,
            connection_args={"uri": f"{self.milvus_uri}"}
        )

    def _get_files(self, path: str) -> list[str]:
        """
        Determines whether the path is a single file or a folder and returns the relevant PDF file(s).

        Args:
            path (str): A file or folder path.

        Returns:
            list[str]: A list of PDF file paths.
        """
        if os.path.isfile(path) and path.endswith(".pdf"):
            return [path]  # Single PDF file
        elif os.path.isdir(path):
            return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pdf")]
        else:
            raise ValueError(f"Invalid path: {path}. Must be a .pdf file or a directory containing .pdf files.")

    def get_vectorstore(self):
        return self.vectorstore
