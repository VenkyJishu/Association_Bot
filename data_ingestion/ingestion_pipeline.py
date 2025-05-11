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

logging.basicConfig(level=logging.INFO)

class DataIngestion():
    def __init__(self):
        load_dotenv()
        os.environ["GOOGLE_API_KEY"]  = os.getenv("GEMINI_API_KEY")

        #genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.llm_model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

        
    def embed_model(self,text):
        result = self.model.embed_documents([text])

        return result
    
    def load_file(self,file_path="dataset/Association_Last_6_Months_Statement.pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def split_text(self,text):
        splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
        return splitter.split_text(text=text)
    
    def create_vector_store(self,chunks):
        #embeddings = self.embed_model()
        vector_store = FAISS.from_texts(chunks,embedding=self.model)
        vector_store.save_local("faiss_index") 
        return vector_store
    
    def generate_answer(self,query,chunks,k=5):
        #query_embedding = self.embed_model(query)
        query_embedding = np.array(self.embed_model(query))


        # Perform a search in the FAISS index
        D, I = vs.index.search(query_embedding.reshape(1, -1), k)

        # Get the top-k relevant chunks
        retrieved_chunks = [chunks[i] for i in I[0]]
        context = "\n".join(retrieved_chunks)
        prompt = f"""
                    You are a helpful assistant. Use the following context to answer the question.

                    Context:
                    {context}

                    Question:
                    {query}

                    Answer:
                    """
        response = self.llm_model.invoke(prompt)
        return response.text

if __name__ == "__main__":
    di = DataIngestion()
    #r = di.embed_model("What is the meaning of life")
    
    text = di.load_file()
    chunks = di.split_text(text)
    vs = di.create_vector_store(chunks)
    print(f"Vector store contains {vs.index.ntotal} embeddings.")

    query = "What were the major expenses in the last 6 months?"
    answer = di.generate_answer(query, chunks)

    print("\nGenerated Answer:\n", answer)
