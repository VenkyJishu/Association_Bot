import os
import re
import logging
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import List,Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from datetime import datetime


from utils.model_loader import ModelLoader
from config.config_loader import load_config

logging.basicConfig(level=logging.INFO)

class DataIngestion:
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
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, 'dataset', 'Association_Last_6_Months_Statement.pdf')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")
        return file_path

    def _load_file(self):
        reader = PdfReader(self.file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def extract_metadata(self, text: str):
        # Match month (case insensitive)
        # Match dd/mm/yyyy pattern (e.g., 01/11/2024)
        date_matches = re.findall(r'\b\d{2}/\d{2}/\d{4}\b', text)
        month = None
        year = None
        flat = None

        # Match year (just one valid year)
        #year_match = re.search(r'\b(20[2-3]\d)\b', text)  # Accepts 2020-2039 only
        # Match flat number (e.g., A-107 or A107)
        flat_match = re.search(r'\b([A-Za-z]-?\d{2,4})\b', text)

        if date_matches:
            try:
                # Use the earliest date as representative
                parsed_date = datetime.strptime(date_matches[0], "%d/%m/%Y")
                month = parsed_date.strftime("%B")  # e.g., "November"
                year = str(parsed_date.year)
            except Exception as e:
                print(f" Date parsing error: {e}")
        
        #year = year_match.group(0) if year_match else None
        flat = flat_match.group(0).upper() if flat_match else None

        return month, year, flat

    @staticmethod
    def custom_chunk_split(text:str) -> List[str]:
        chunks = re.split(r'(?=\b\d{2}/\d{2}/\d{4}\b)',text)
        return [chunk for chunk in chunks if chunk.strip()]
    
    @staticmethod
    def extract_all_metadata(text:str) -> List[Tuple[str,str,str]]:
        date_matches = re.findall(r'\b\d{2}/\d{2}/\d{4}\b', text)
        flat_matches = re.findall(r'\b([A-Za-z]-?\d{2,4})\b', text)

        metadata_list = []
        for date_val,flat in zip(date_matches,flat_matches):
            try:
                parsed_date = datetime.strptime(date_val, "%d/%m/%Y")
                month = parsed_date.strftime("%B")
                year = str(parsed_date.year)
                flat = flat.upper()

                metadata_list.append((month,year,flat))
            except Exception as e:
                print(f"Error while parsing date {date_val} and error is {e}")
                continue
        return metadata_list
    
    def transform_data(self) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                  chunk_overlap=100,
                                                  separators=["\n\n", "\n", ".", " "]
                                                  )
        chunks = splitter.split_text(text=self.data)
        #chunks = self.custom_chunk_split(self.data)

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---\n{chunk[:500]}...\n")



        documents = []
        for i, chunk in enumerate(chunks):
            metadata_details = self.extract_all_metadata(chunk)
            #if isinstance(year, list):
             #   print(f"⚠️ Unexpected list type in year: {year} (chunk index: {i})")
            
            # If no metadata found, still store chunk
            if not metadata_details:
                doc = Document(
                    page_content=chunk,
                    metadata={"source": "bank_statement"}
                )
                documents.append(doc)
                continue
            
            # Create one document per flat/month/year match
            for month,year,flat in metadata_details:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": "bank_statement",
                        "month": month,
                        "year": year,
                        "flat": flat
                    }
                )
                if i < 5:
                    print(f"[Sample {i+1}] Metadata: {doc.metadata}")
                documents.append(doc)

        print(f"Transformed {len(documents)} documents.")
        return documents

    
    def create_vector_store(self, documents: List[Document]):
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
    def get_loaded_unique_months_year_details(self,vstore):
        result = vstore.similarity_search(" ",k=1000)
        months = set()
        years = set()
        
        for doc in result:
            if doc.metadata.get("month"):
                months.add(doc.metadata["month"])
            if doc.metadata.get("year"):
                years.add(doc.metadata["year"])
        
        print(f"\n Unique Months loaded {months}")
        print(f"\n Unique years loaded {years}")

    def run_pipeline(self):
        documents = self.transform_data()
        vstore, inserted_ids = self.create_vector_store(documents)

        self.get_loaded_unique_months_year_details(vstore=vstore)

        # Optional: sample test
        query = "What were the major expenses last month?"
        results = vstore.similarity_search(query)
        print(f"\n🔍 Sample search results for: '{query}'")
        for res in results:
            print(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")

if __name__ == "__main__":
    test_text = """
                01/11/2024    BY TRANSFER-.../A107-TRANSFER
                01/11/2024    BY TRANSFER-.../B303-TRANSFER
                Invalid date A-999
                """
    di = DataIngestion()
    #res = di.extract_all_metadata(test_text) # For testing sample data
    #print(res)
    di.run_pipeline()