import streamlit as st
import chromadb
import os
import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- AGGIORNAMENTO PERCORSI MODULI ---
# Aggiungiamo la cartella 'rag_pipeline' al percorso di ricerca di Python
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_folder_path = os.path.join(current_dir, "..", "rag_pipeline")
sys.path.append(rag_folder_path)

try:
    from rag_pipeline import setup_rag_chain 
except ImportError:
    st.error(f"❌ Impossibile trovare 'rag_pipeline.py' in: {rag_folder_path}")
    st.stop()

# --- CONFIGURAZIONE PERCORSI DATABASE ---
# Usiamo un percorso relativo che risale fino alla cartella root 'ARIS'
DB_PATH = os.path.join(current_dir, "..", "..", "vector_db", "chroma_locale_700") 
COLLECTION_NAME = "manuali_fanuc_es1"

@st.cache_resource
def init_retriever():
    """Inizializza il database vettoriale."""
    if not os.path.exists(DB_PATH):
        st.error(f"❌ Database non trovato in: {os.path.abspath(DB_PATH)}. Verifica la struttura delle cartelle.")
        st.stop()
        
    client = chromadb.PersistentClient(path=DB_PATH)
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    lc_chroma = Chroma(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embedding_function=embedder
    )
    return lc_chroma.as_retriever(search_kwargs={"k": 3})

# --- INTERFACCIA STREAMLIT ---
st.set_page_config(page_title="ARIS - Assistente Fanuc", page_icon="🤖", layout="wide")

# --- SIDEBAR DI CONFIGURAZIONE ---
st.sidebar.title("⚙️ Configurazione Sistema")
st.sidebar.markdown("---")

scelta_env = st.sidebar.radio(
    "Seleziona Motore LLM:",
    ["locale", "cloud"],
    index=0,
    help="Locale usa LM Studio sul tuo PC. Cloud usa OpenRouter (richiede API Key)."
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Stato Sistema:**
- DB: `{os.path.basename(DB_PATH)}`
- Chunk Size: `700 token`
- Retriever: `Ibrido (Semantico + BM25)`
""")

# --- LOGICA DI AVVIO ---
st.title("🤖 ARIS: Assistente per Manutenzione Industriale")
st.caption("Supporto tecnico basato su RAG per Robot Fanuc R-30iB Mate")

# 1. Inizializza il database
retriever = init_retriever()

# 2. Configura la catena
try:
    rag_chain = setup_rag_chain(retriever, env=scelta_env)
except Exception as e:
    st.error(f"Errore nella configurazione della pipeline: {e}")
    st.stop()

# --- GESTIONE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Chiedi aiuto su un errore o una procedura (es: SRVO-004)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        with st.spinner(f"L'IA ({scelta_env}) sta consultando i manuali..."):
            try:
                full_response = rag_chain.invoke(prompt)
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                msg_errore = f"❌ Errore durante la generazione ({scelta_env}). "
                if scelta_env == "locale":
                    msg_errore += "Assicurati che LM Studio abbia il Server avviato su porta 1234."
                st.error(f"{msg_errore}\n\nDettagli: {e}")