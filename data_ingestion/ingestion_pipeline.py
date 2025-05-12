import os
import sys
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
import logging
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

from utils.model_loader import ModelLoader
from config.config_loader import load_config
from typing import List
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore

logging.basicConfig(level=logging.INFO)

class DataIngestion():
    def __init__(self):
        self.model_loader = ModelLoader()
        self.config = load_config()
        self._load_env_variables()
        self.file_path = self._get_file_path()
        self.data = self._load_file()

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
    
    def _get_file_path(self):
        """
        Get path to the  file located inside 'dataset' folder.
        """
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, 'dataset', 'Association_Last_6_Months_Statement.pdf')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f" file not found at: {file_path}")

        return file_path

    def _load_file(self):
        if not  os.path.exists(self.file_path):
            raise FileNotFoundError(f"File does not exist at {self.file_path}")


        reader = PdfReader(self.file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def transform_data(self) -> List[Document]:
        """
        Split bank statement text into documents suitable for embedding.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
        chunks =  splitter.split_text(text=self.data)
        
        # Creating documents with metadata (if needed)
        documents = []
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={"source": "bank_statement"})
            documents.append(doc)

        print(f"Transformed {len(documents)} documents.")
        return documents
    
    def create_vector_store(self,documents : List[Document]):
        """
        Store documents into AstraDB vector store.
        """
        collection_name = self.config["astra_db"]["collection_name"]
        vstore = AstraDBVectorStore(
            embedding=self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace,
        )

        inserted_ids = vstore.add_documents(documents)
        print(f"Successfully inserted {len(inserted_ids)} documents into AstraDB.")
        return vstore, inserted_ids
    
    def run_pipeline(self):
        """
        Run the full data ingestion pipeline: transform data and store into vector DB.
        """
        documents = self.transform_data()
        vstore, inserted_ids = self.create_vector_store(documents)

        # Optionally do a quick search
        query = "What were the major expenses last month?"
        results = vstore.similarity_search(query)

        print(f"\nSample search results for query: '{query}'")
        for res in results:
            print(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")

if __name__ == "__main__":
    di = DataIngestion()
    di.run_pipeline()