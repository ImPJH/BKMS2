from typing import Iterable
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus import Milvus
from model import LLMModel


class RAGPipeline:
    def __init__(self, vectorstore, llm):
        """
        Initializes the RAG pipeline.

        Args:
            vectorstore: The Milvus vectorstore for retrieving documents.
            llm: The HuggingFace LLM model for generating responses.
        """
        self.vectorstore = vectorstore
        self.llm = llm

    @staticmethod
    def format_docs(docs: Iterable[LCDocument]) -> str:
        """
        Formats documents into a string for prompt context.

        Args:
            docs: Iterable of LCDocument objects.

        Returns:
            A formatted string of document content.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def build_pipeline(self):
        """
        Builds the RAG chain pipeline.

        Returns:
            The RAG pipeline.
        """
        # Convert vectorstore to retriever
        retriever = self.vectorstore.as_retriever()

        # Define the prompt template
        prompt = PromptTemplate.from_template(
            "Context information is below.\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\n"
            "Query: {question}\n"
            "Answer:\n"
        )

        # Create the RAG pipeline
        rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    def run(self, query: str) -> str:
        """
        Runs the RAG pipeline for a given query.

        Args:
            query: The input question to the RAG pipeline.

        Returns:
            The response from the RAG pipeline.
        """
        pipeline = self.build_pipeline()
        return pipeline.invoke(query)