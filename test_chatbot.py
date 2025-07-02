import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Carica la chiave API dal file .env che abbiamo creato
load_dotenv() 

# Controlliamo subito se la chiave API è stata caricata correttamente
if not os.getenv("OPENAI_API_KEY"):
    st.error("Chiave API di OpenAI non trovata! Assicurati di aver creato il file .env e inserito la chiave correttamente.")
    st.stop()

# Questa funzione carica i PDF, li divide e li "memorizza"
# Usiamo @st.cache_resource per non ripetere l'operazione ogni volta che fai una domanda
@st.cache_resource
def prepara_database_documenti():
    percorso_docs = './docs'

    # Controlliamo se la cartella 'docs' esiste
    if not os.path.exists(percorso_docs):
        return None, "La cartella 'docs' non è stata trovata."

    documenti_caricati = []
    for file in os.listdir(percorso_docs):
        if file.endswith('.pdf'):
            percorso_file = os.path.join(percorso_docs, file)
            loader = PyPDFLoader(percorso_file)
            documenti_caricati.extend(loader.load())

    if not documenti_caricati:
        return None, "Nessun file PDF trovato nella cartella 'docs'."

    # Dividiamo i testi in pezzi più piccoli
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    testi_divisi = text_splitter.split_documents(documenti_caricati)

    # Creiamo gli "embeddings" (rappresentazioni numeriche) e li salviamo in ChromaDB
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=testi_divisi, embedding=embeddings)

    st.success(f"Database creato con successo da {len(documenti_caricati)} pagine di documenti!")
    return vectorstore, None

# --- Inizia l'interfaccia della nostra pagina web di test ---
st.title("🧪 Prototipo Chatbot Sindacale")
st.write("Fai una domanda basata sui documenti che hai caricato nella cartella 'docs'.")

vectorstore, errore = prepara_database_documenti()

# Se c'è stato un errore durante la preparazione, lo mostriamo e fermiamo l'app
if errore:
    st.error(errore)
else:
    # Se tutto è andato bene, creiamo la logica di domanda e risposta
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Creiamo il campo di testo per la domanda dell'utente
    domanda_utente = st.text_input("Scrivi qui la tua domanda:")

    if domanda_utente:
        with st.spinner("Sto cercando la risposta nei documenti..."):
            try:
                risposta = qa_chain.invoke({"query": domanda_utente})
                st.write("### Risposta:")
                st.write(risposta["result"])
            except Exception as e:
                st.error(f"Si è verificato un errore: {e}")