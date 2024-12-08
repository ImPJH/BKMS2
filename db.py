import os
from typing import Iterator, Union,Optional,List
from tempfile import TemporaryDirectory

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from docling.document_converter import DocumentConverter

import re
# class DoclingPDFLoader(BaseLoader):
#     def __init__(self, file_path: Union[str, list[str]]) -> None:
#         self._file_paths = file_path if isinstance(file_path, list) else [file_path]
#         self._converter = DocumentConverter()

#     def lazy_load(self) -> Iterator[LCDocument]:
#         for source in self._file_paths:
#             dl_doc = self._converter.convert(source).document
#             text = dl_doc.export_to_markdown()
#             yield LCDocument(page_content=text)
class DoclingPDFLoader(BaseLoader):
    def __init__(self, file_path: Union[str, list[str]]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()

            # 불필요한 텍스트 제거
            clean_text = self.clean_text(text)
            yield LCDocument(page_content=clean_text, metadata={"source": source})

    @staticmethod
    def clean_text(text: str) -> str:
        # HTML 주석, 특수 문자, 불필요한 레이아웃 제거
        import re
        text = re.sub(r"<!--.*?-->", "", text)  # HTML 주석 제거
        text = re.sub(r"[|]+", "", text)  # 파이프(|) 제거
        text = re.sub(r"\s+", " ", text)  # 공백 정리
        return text.strip()


class DB:
    def __init__(self, path: str, embed_model: str, milvus_uri: str, dir_list: Optional[List[str]] = None):
        """
        Initialize the class with paths and embedding model details.

        Args:
            path (str): Either a single PDF file path or a folder path containing multiple PDFs.
            embed_model (str): HuggingFace embedding model name.
            milvus_uri (str): URI for the Milvus vector database.
            dir_list (Optional[List[str]]): A list of directory paths, or None if no directories are provided.
        """
        self.path = path
        self.embed_model = embed_model
        self.milvus_uri = milvus_uri
        self.vectorstore = None
        self.embedding = HuggingFaceEmbeddings(model_name=self.embed_model)
        self.dir_list = dir_list or None

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
            #ssg ; Extract phone_type from file name (e.g., 폴더 안에 'samsung.pdf' -> phone_type = 'samsung')
            phone_type = os.path.basename(doc.metadata.get("source", "")).replace(".pdf", "").lower()
            doc.metadata["phone_type"] = phone_type
            print(f"process_documents: save phone type as {phone_type}")
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
        
    def process_multiple_directories(self):
        """
        Process all files from the provided directories and add them to the same vectorstore.

        This method combines files from multiple directories (if specified in `self.dir_list`)
        and processes them into a single Milvus vectorstore.

        Returns:
            None
        """
        if self.dir_list is None or len(self.dir_list) == 0:
            raise ValueError("No directories specified in `dir_list` to process.")

        # Check if the Milvus database directory exists
        if os.path.exists(self.milvus_uri):
            print("Loading existing vectorstore...")
            self.vectorstore = self._load_existing_vectorstore()
            return

        print("Creating new vectorstore...")

        # Collect files from all directories
        all_files = []
        for directory in self.dir_list:
            all_files.extend(self._load_files_from_directories())

        # Ensure all files are unique
        all_files = list(set(all_files))

        print(f"Total files to process: {len(all_files)}")

        # Load documents
        loader = DoclingPDFLoader(file_path=all_files)
        documents = list(loader.lazy_load())

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        splits = []
        for doc in documents:
            #ssg; Extract phone_type from file name (e.g., 'samsung.pdf' -> phone_type = 'samsung')
            phone_type = os.path.basename(doc.metadata.get("source", "")).replace(".pdf", "").lower()
            print(f"process_multiple_directories: save phone type as {phone_type}")
            doc.metadata["phone_type"] = phone_type
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

    def _load_files_from_directories(self) -> list[str]:
        """
        Read all files from the specified directories.

        Args:
            directories (list[str]): A list of folder paths to search for files.

        Returns:
            list[str]: A list of file paths found in the given directories.
        """
        
        all_files = []
        for directory in self.dir_list:
            print(f"Loading files from {directory}")
            for root, _, files in os.walk(directory):
                for file in files:
                    all_files.append(os.path.join(root, file))
                    print(f"{file} added")
        return all_files

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
