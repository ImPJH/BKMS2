import os
import json
from langchain_huggingface import HuggingFaceEndpoint


class LLMModel:
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)

        # Extract API Key and Model ID
        self.api_key = self.config.get("hf_api_key")
        self.model_id = self.config.get("hf_llm_model_id")

        # Ensure environment variable is set
        os.environ["HF_API_KEY"] = self.api_key

        # Initialize the HuggingFaceEndpoint
        self.llm = HuggingFaceEndpoint(
            repo_id=self.model_id,
            huggingfacehub_api_token=self.api_key,
        )

    def get_llm(self):
        return self.llm