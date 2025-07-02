from fastapi import FastAPI
from pydantic import BaseModel
import os

# NUOVO IMPORT: Aggiungi questa riga
from fastapi.middleware.cors import CORSMiddleware

# Importiamo le stesse librerie di LangChain di prima
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Carichiamo la chiave API dal file .env
load_dotenv()

# --- Funzione di Setup ---
def setup_qa_chain():
    percorso_docs = './docs'
    documenti_caricati = []
    
    for file in os.listdir(percorso_docs):
        if file.endswith('.pdf'):
            percorso_file = os.path.join(percorso_docs, file)
            loader = PyPDFLoader(percorso_file)
            documenti_caricati.extend(loader.load())
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    testi_divisi = text_splitter.split_documents(documenti_caricati)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=testi_divisi, embedding=embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

# --- Avvio dell'API ---
app = FastAPI(
    title="API Chatbot Sindacale",
    description="Un'API per interrogare documenti di un sindacato."
)

# --- NUOVA CONFIGURAZIONE CORS ---
# Aggiungi queste righe. Permettono al tuo browser di parlare con l'API.
origins = ["*"]  # In produzione, dovresti essere più specifico (es. "http://tuo-sito.com")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------------

qa_chain = setup_qa_chain()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    risposta = qa_chain.invoke({"query": query.question})
    return {"answer": risposta["result"]}

@app.get("/")
def read_root():
    return {"Status": "API del Chatbot Attiva"}