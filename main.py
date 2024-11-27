import json
from tempfile import TemporaryDirectory
from model import LLMModel
from db import DB  
from rag import RAGPipeline

# Example usage
if __name__ == "__main__":
    import json

    # Load configuration
    config_path = "config.json"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    # Initialize vectorstore and LLM
    db = DB(
        path=config["folder_path"],
        embed_model=config["embed_model"],
        milvus_uri= config["milvus_uri"]
    )
    db.process_documents()
    vectorstore = db.get_vectorstore()

    llm_model = LLMModel(config_path=config_path)
    llm = llm_model.get_llm()

    # Initialize and run the RAG pipeline
    rag_pipeline = RAGPipeline(vectorstore=vectorstore, llm=llm)

    #run rag inference
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting the program.")
            break
        response = rag_pipeline.run(user_query)
        print(f"Response: {response}")