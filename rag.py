from typing import Iterable
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus import Milvus
from model import LLMModel

import json
import numpy as np
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import asyncio


def retrieve_cache(json_file):
    try:
        with open(json_file, "r") as file:
            cache = json.load(file)
    except FileNotFoundError:
        cache = {"questions": [], "embeddings": [], "answers": [], "response_text": []}

    return cache

class semantic_cache:
    def __init__(self, json_file="cache_file.json", thresold=0.95, max_response=100, eviction_policy="FIFO"):
        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)
        self.thresold = thresold
        self.max_response = max_response
        self.eviction_policy = eviction_policy

    def evict(self):
        """Evicts an item from the cache based on the eviction policy."""
        if self.eviction_policy and len(self.cache["embeddings"]) > self.max_response:
            for _ in range((len(self.cache["embeddings"]) - self.max_response)):
                if self.eviction_policy == "FIFO":
                    self.cache["embeddings"].pop(0)
                    self.cache["response_text"].pop(0)

    def ask(self,emb_query,filters):
        total_key = "None"
        if filters:
            for k, v in filters.items():
                total_key += k + v

        if total_key not in self.cache["embeddings"]:
            self.cache["embeddings"][total_key] = []
            self.cache["response_text"][total_key] = []
            
        np_cache = np.array(self.cache["embeddings"][total_key])
        np_query = np.array(emb_query)
        if len(np_cache) > 0:
            cos_sims = np.dot(np_cache,np_query.T).T/(np.linalg.norm(np_cache,2,axis=1)*np.linalg.norm(np_query,2))

            for i , sim in enumerate(cos_sims):
                if sim >= self.thresold:
                    return self.cache["response_text"][total_key][i]
    
        return None
    
    def store_cache(self):
        with open(self.json_file, "w") as file:
            json.dump(self.cache, file)
    
    def append_result(self,emb_query,res,filters):
        total_key = "None"
        if filters:
            for k, v in filters.items():
                total_key += k + v

        self.cache["embeddings"][total_key].append(emb_query)
        self.cache["response_text"][total_key].append(res)
        self.evict()
        self.store_cache()
    

class RAGPipeline:
    def __init__(self, vectorstore, llm, config_path):
        """
        Initializes the RAG pipeline.

        Args:
            vectorstore: The Milvus vectorstore for retrieving documents.
            llm: The HuggingFace LLM model for generating responses.
        """
        self.vectorstore = vectorstore
        self.llm = llm

        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        self.embedding = HuggingFaceEmbeddings(model_name=config["embed_model"])
        self.cache = semantic_cache(json_file="cache_file.json")

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

    # def build_pipeline(self):
    #     """
    #     Builds the RAG chain pipeline.

    #     Returns:
    #         The RAG pipeline.
    #     """
    #     # Convert vectorstore to retriever
    #     retriever = self.vectorstore.as_retriever()

    #     # Define the prompt template
    #     prompt = PromptTemplate.from_template(
    #         "Context information is below.\n"
    #         "---------------------\n"
    #         "{context}\n"
    #         "---------------------\n"
    #         "Given the context information and not prior knowledge, answer the query.\n"
    #         "Query: {question}\n"
    #         "Answer:\n"
    #     )

    #     # Create the RAG pipeline
    #     rag_chain = (
    #         {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
    #         | prompt
    #         | self.llm
    #         | StrOutputParser()
    #     )

    #     return rag_chain

    # def run(self, query: str) -> str:
    #     """
    #     Runs the RAG pipeline for a given query.

    #     Args:
    #         query: The input question to the RAG pipeline.

    #     Returns:
    #         The response from the RAG pipeline.
    #     """
    #     pipeline = self.build_pipeline()
    #     return pipeline.invoke(query)


    #ssg;
    def build_pipeline(self, filters=None):
        """
        Builds the RAG chain pipeline with optional filters.

        Args:
            filters: Optional filters to apply during document retrieval.

        Returns:
            The RAG pipeline.
        """
        # Retrieve documents from vectorstore
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

        def apply_filters(docs):
            if filters is None:
                return docs
            # Filter documents manually based on metadata
            return [
                doc for doc in docs
                if all(doc.metadata.get(key) == value for key, value in filters.items())
            ]

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

        # Create the RAG pipeline with manual filter application
        rag_chain = (
            {"context": retriever | apply_filters | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain


    def run(self, query: str, filters=None) -> str:
        """
        Runs the RAG pipeline for a given query with optional filters.

        Args:
            query: The input question to the RAG pipeline.
            filters: Optional filters to apply during document retrieval.

        Returns:
            The response from the RAG pipeline.
        """
        # Build pipeline
        pipeline = self.build_pipeline(filters=filters)

        # Log filter conditions
        if filters:
            print(f"Applying filters: {filters}")

        # Execute pipeline
        embedded_query = self.embedding.embed_query(query)
        cache_response = self.cache.ask(embedded_query, filters)
        if cache_response:
            print("Segment cache used!")
            return "You already asked this question! (This answer is attained by segment cache.)\n\n\n"+cache_response
        


        response = pipeline.invoke(query)

        # Log filtered results (if debugging)
        if filters:
            results = self.vectorstore.similarity_search(query, k=10)
            filtered_results = [
                doc for doc in results
                if all(doc.metadata.get(key) == value for key, value in filters.items())
            ]
            print(f"Filtered results: {len(filtered_results)} documents matched filters.")
            for doc in filtered_results:
                print(f"Metadata: {doc.metadata}")

        self.cache.append_result(embedded_query,response,filters)
        return response
