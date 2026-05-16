import streamlit as st
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
rag_folder_path = os.path.join(current_dir, "..", "rag_pipeline")
sys.path.append(rag_folder_path)

try:
    from rag_pipeline_hybrid import setup_rag_chain, get_db_path, get_embeddings, build_hybrid_retriever
except ImportError:
    st.error(f"❌ Impossibile trovare 'rag_pipeline_hybrid.py' in: {rag_folder_path}")
    st.stop()

from langchain_chroma import Chroma

COLLECTION_NAME = "langchain"

PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "..", ".."))

@st.cache_resource
def init_retriever(env: str, chunk_size: int):
    """Inizializza il retriever ibrido (BM25 + Vector Search) per l'env scelto."""
    db_path = os.path.join(PROJECT_ROOT, get_db_path(env, chunk_size))

    if not os.path.exists(db_path):
        st.error(f"❌ Database non trovato in: {db_path}")
        st.stop()

    embedder = get_embeddings(env)
    lc_chroma = Chroma(
        persist_directory=db_path,
        collection_name=COLLECTION_NAME,
        embedding_function=embedder
    )
    return build_hybrid_retriever(lc_chroma, k=3)


@st.cache_resource
def init_rag_chain(env: str, chunk_size: int):
    """Inizializza l'intera chain RAG (retriever + LLM).
    Cached per evitare di ricreare l'LLM ad ogni rerun della sidebar.
    Si ricalcola solo se cambiano env o chunk_size.
    """
    retriever = init_retriever(env, chunk_size)
    return setup_rag_chain(retriever, env=env)


st.set_page_config(page_title="ARIS - Assistente Fanuc", page_icon="🤖", layout="wide")

st.sidebar.title("⚙️ Configurazione Sistema")
st.sidebar.markdown("---")

scelta_env = st.sidebar.radio(
    "Seleziona Motore LLM:",
    ["locale", "cloud"],
    index=0,
    help="Locale usa LM Studio sul tuo PC. Cloud usa OpenRouter (richiede API Key nel .env)."
)

scelta_chunk_size = st.sidebar.selectbox(
    "Dimensione Chunk:",
    [300, 700, 1000],
    index=1,
    help="Deve corrispondere alla dimensione usata durante l'ingestion nel Vector DB."
)

st.sidebar.markdown("---")
db_label = get_db_path(scelta_env, scelta_chunk_size)
embedding_label = "bge-m3 (1024d, locale)" if scelta_env == "locale" else "text-embedding-3-small (1536d, OpenRouter)"
st.sidebar.info(f"""
**Stato Sistema:**
- DB: `{db_label}`
- Collection: `{COLLECTION_NAME}`
- Chunk Size: `{scelta_chunk_size} token`
- Embedding: `{embedding_label}`
- Retriever: `Ibrido BM25 + Vector Search (50/50)`
""")

st.title("🤖  ARIS: Assistente per Manutenzione Industriale")
st.caption("Supporto tecnico basato su RAG per Robot Fanuc R-30iB Mate")

# Inizializza retriever + chain (entrambi cached per env+chunk_size)
try:
    rag_chain = init_rag_chain(scelta_env, scelta_chunk_size)
except Exception as e:
    st.error(f"Errore nella configurazione della pipeline: {e}")
    st.stop()

# --- GESTIONE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.bottom:
    audio_value = st.audio_input("🎤 Registra messaggio vocale", label_visibility="collapsed")
    prompt = st.chat_input("Chiedi aiuto su un errore o una procedura (es: SRVO-004)...")


if audio_value is not None and st.session_state.get("last_audio_value") != audio_value:
    st.session_state["last_audio_value"] = audio_value
    with st.spinner("Sto trascrivendo l'audio..."):
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(audio_value) as source:
                audio_data = r.record(source)
            testo_vocale = r.recognize_google(audio_data, language="it-IT")
            prompt = testo_vocale  # Sovrascrive il prompt con la trascrizione
        except sr.UnknownValueError:
            st.error("❌ Non ho capito l'audio. Riprova.")
        except Exception as e:
            st.error(f"❌ Errore nel riconoscimento vocale: {e}")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        with st.spinner(f"L'IA ({scelta_env}) sta consultando i manuali..."):
            try:
                # Costruisce la cronologia degli ultimi 3 scambi (6 messaggi)
                # escludendo il messaggio appena aggiunto
                history_msgs = st.session_state.messages[:-1][-6:]
                history_str = "\n".join(
                    f"{m['role'].upper()}: {m['content']}"
                    for m in history_msgs
                ) if history_msgs else "(nessuna conversazione precedente)"

                full_response = rag_chain.invoke({
                    "question": prompt,
                    "history":  history_str,
                })
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                msg_errore = f"❌ Errore durante la generazione ({scelta_env}). "
                if scelta_env == "locale":
                    msg_errore += "Assicurati che LM Studio abbia il Server avviato su porta 1234."
                else:
                    msg_errore += "Verifica che OPENAI_API_KEY sia configurata nel file .env."
                st.error(f"{msg_errore}\n\nDettagli: {e}")