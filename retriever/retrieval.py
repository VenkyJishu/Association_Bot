from utils.model_loader import ModelLoader
from dotenv import load_dotenv
import os
from typing import List
from langchain_core.documents import Document
from config.config_loader import load_config
from langchain_astradb import AstraDBVectorStore
import re
from datetime import datetime


class Retriever():
    def __init__(self):
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.config = load_config()

        self.vstore = None
        self.retriever = None

    def _load_env_variables(self):
        
        load_dotenv()
        
        required_vars = ["GEMINI_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        self.google_api_key = os.getenv("GEMINI_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    def extract_month_and_flat(self,query: str):
        # Regex to extract flat number (e.g., "A-107")
        flat_number = re.search(r'([A-Za-z]-\d+)', query)
        # Regex to extract month (e.g., "May")
        month = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)', query)
        
        # If month or flat number is found, return them
        if flat_number and month:
            return month.group(0), flat_number.group(0)
        else:
            return None, None  # Return None if not found

    def load_retriever(self):
        if not self.vstore:
            collection_name = self.config['astra_db']['collection_name']
            print("load embeddings")
        
            self.vstore = AstraDBVectorStore(
                                    api_endpoint=self.db_api_endpoint,
                                    token=self.db_application_token,
                                    collection_name=collection_name,
                                    namespace=self.db_keyspace,
                                    embedding=self.model_loader.load_embeddings(),
                                )
        if not self.retriever:
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3
            retriever = self.vstore.as_retriever(search_kwargs={"k": 20
                                                                }
                                                )
            
            print("Retriever loaded successfully.")
            return retriever
        
    def call_retriever(self,query:str) -> List[Document]:
        # Extract month and flat number from the query
        month, flat = self.extract_month_and_flat(query)
        retriever = self.load_retriever()
            # If month and flat are found, apply them to the filter
        filter_kwargs = {}
        if month:
            filter_kwargs["month"] = month
        if flat:
            filter_kwargs["flat"] = flat
        
        output = retriever.invoke(query,filter=filter_kwargs)
        return output
    
if __name__=='__main__':
    retriever_obj = Retriever()
    retriever_obj.load_retriever()
    # Check how many documents were ingested in total
    
    # ✅ Get the raw collection object
    collection_name = retriever_obj.config['astra_db']['collection_name']
    #raw_collection = retriever_obj.vstore._client.get_collection(collection_name)

    # ✅ Count documents safely
    #print("📦 Total docs in Astra DB:", raw_collection.count_documents())

    user_query = "Who paid maintenance recently, list recent 5  "
    results = retriever_obj.call_retriever(user_query)
    print(f"🔎 Retrieved {len(results)} documents")

    for idx, doc in enumerate(results, 1):
        print(f"Result {idx}: {doc.page_content[:1000]}\nMetadata: {doc.metadata}\n")