import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config_loader import load_config



class ModelLoader():
    def __init__(self):
        load_dotenv()
        self.config = load_config()
        self._validate_env()
        self.google_api_key = os.getenv("GEMINI_API_KEY")
    
    def _validate_env(self):

        req_vars = ['GEMINI_API_KEY']
        missing_var = [var for var in req_vars if not os.getenv(var) ]

        if missing_var:
            raise EnvironmentError(f"Missing environment variables {missing_var}")
    
    def load_embeddings(self):
        print(f"venky self.google_api_key is {self.google_api_key}")
        embeddings = GoogleGenerativeAIEmbeddings(model=self.config['embedding_model']['model_name'],api_key=self.google_api_key)
        return embeddings
    
    def load_llm(self):

        llm = ChatGoogleGenerativeAI(model=self.config['llm']['chat_model'],api_key=self.google_api_key)
        return llm


if __name__ == "__main__":
    ml = ModelLoader()
    ml.load_embeddings()
