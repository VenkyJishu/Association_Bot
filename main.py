from dotenv import load_dotenv
from utils.model_loader import ModelLoader
from retriever.retrieval import Retriever
from config.config_loader import load_config
from data_ingestion.ingestion_pipeline import DataIngestion

from fastapi import FastAPI,Request,Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from prompt_library.prompt import PROMPT_TEMPLATE
from google.api_core.exceptions import ResourceExhausted
import random
import time

load_dotenv()



retriever_obj = Retriever()
model_loader = ModelLoader()

def invoke_chain(query:str,retries: int = 3, backoff_factor: float = 1.5):
    docs = retriever_obj.call_retriever(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE['Association_bot'])
    llm = model_loader.load_llm()

    chain = (        
          prompt
        | llm
        | StrOutputParser()
    )

    input_values = {"context": context, "question": query}
    
    print("🔍 Context being passed to LLM:\n")
    print(context[:2000])  # print up to 2k chars for readability

    for attempt in range(retries):
        try:

            return chain.invoke(input_values)
        except ResourceExhausted as e:
            if attempt == retries - 1:
                raise e
            wait = backoff_factor ** attempt + random.uniform(0, 0.5)
            print(f"[WARN] Quota hit. Retrying in {wait:.2f} seconds...")
            time.sleep(wait)



query = "Did A-107 flat paid mmaintenance in May 2025 "
response = invoke_chain(query=query)
print("\n LLM Response:\n", response)