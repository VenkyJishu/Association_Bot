from dotenv import load_dotenv
from utils.model_loader import ModelLoader
from retriever.retrieval import Retriever
from config.config_loader import load_config
from data_ingestion.ingestion_pipeline import DataIngestion

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from prompt_library.prompt import PROMPT_TEMPLATE
from google.api_core.exceptions import ResourceExhausted
import random
import time
import logging


# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Components
retriever_obj = Retriever()
model_loader = ModelLoader()
chat_history = []

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE['Admin_bot'])
llm = model_loader.load_llm()
chain = prompt | llm | StrOutputParser()

GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "how are you"]

def is_greeting(query: str) -> bool:
    return query.strip().lower() in GREETINGS

def retrieve_context(query: str) -> str:
    docs = retriever_obj.call_retriever(query)
    if not docs:
        return ""
    return "\n\n".join([doc.page_content for doc in docs])

def invoke_chain(query: str, retries: int = 3, backoff_factor: float = 1.5) -> str:
    context = retrieve_context(query=query)
    if not context:
        return "No relevant information found for this query."

    # Include last 3 turns of chat history
    full_history_text = "\n".join([
        f"You: {msg['user']}\nAgent: {msg['bot']}" for msg in chat_history[-3:]
    ])

    logger.info("Context being passed to LLM \n")
    logger.info(context[:1000])  # Truncated for logging

    input_values = {
        "context": context,
        "chat_history": full_history_text,
        "question": query
    }

    for attempt in range(retries):
        try:
            return chain.invoke(input_values)
        except ResourceExhausted as e:
            if attempt == retries - 1:
                raise e
            wait = backoff_factor ** attempt + random.uniform(0, 0.5)
            logger.warning(f"[WARN] Quota hit. Retrying in {wait:.2f} seconds...")
            time.sleep(wait)

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": None})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, query: str = Form(...)):
    global chat_history

    if is_greeting(query):
        response = "Hello! How can I help you regarding your apartment or maintenance queries?"
    else:
        response = invoke_chain(query=query)

    # Save to history
    chat_history.append({"user": query, "bot": response})

    return templates.TemplateResponse("index.html", {
        "request": request,
        "response": response,
        "query": query,
        "chat_history": chat_history
    })
