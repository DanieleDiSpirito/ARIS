import streamlit as st
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
rag_folder_path = os.path.join(current_dir, "..", "rag_pipeline")
sys.path.append(rag_folder_path)

try:
    import rag_pipeline_hybrid
    import rag_pipeline_rerank
    import rag_pipeline_graph
    from rag_pipeline_hybrid import get_db_path, get_embeddings
except ImportError as e:
    st.error(f"❌ Impossibile caricare i moduli RAG: {e}")
    st.stop()

from langchain_chroma import Chroma

COLLECTION_NAME = "langchain"

PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "..", ".."))

@st.cache_resource
def init_rag_chain(env: str, chunk_size: int, rag_type: str):
    """Inizializza l'intera chain RAG (retriever + LLM).
    Cached per evitare di ricreare l'LLM ed il Reranker ad ogni rerun.
    """
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
    
    if rag_type == "puro":
        retriever = lc_chroma.as_retriever(search_kwargs={"k": 3})
        return rag_pipeline_hybrid.setup_rag_chain(retriever, env=env)
    elif rag_type == "ibrido":
        retriever = rag_pipeline_hybrid.build_hybrid_retriever(lc_chroma, k=3)
        return rag_pipeline_hybrid.setup_rag_chain(retriever, env=env)
    elif rag_type == "rerank":
        ensemble = rag_pipeline_rerank.build_hybrid_retriever(lc_chroma, k=6)
        retriever = rag_pipeline_rerank.HybridRerankRetriever(ensemble, top_n=3)
        return rag_pipeline_rerank.setup_rag_chain(retriever, env=env)
    elif rag_type == "graph":
        retriever = rag_pipeline_graph.build_graph_retriever(lc_chroma, k_base=3, top_n=4)
        return rag_pipeline_graph.setup_rag_chain(retriever, env=env)
    else:
        st.error(f"❌ Algoritmo RAG non riconosciuto: {rag_type}")
        st.stop()


st.set_page_config(page_title="ARIS - Assistente Fanuc", page_icon="🤖", layout="wide")

st.sidebar.title("⚙️ Configurazione Sistema")
st.sidebar.markdown("---")

scelta_env = st.sidebar.radio(
    "Seleziona Motore LLM:",
    ["cloud", "locale"],
    index=0,
    help="Cloud usa OpenRouter (richiede API Key nel .env). Locale usa LM Studio sul tuo PC."
)

scelta_rag_type = st.sidebar.radio(
    "Algoritmo RAG:",
    ["puro", "ibrido", "rerank", "graph"],
    index=2,
    help="Puro = Solo Vector Search. Ibrido = BM25 + Vector Search. Rerank = Ibrido + Cross-Encoder Reranking. Graph = GraphRAG leggero."
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
retriever_label = {
    "puro": "Solo Vector Search (k=3)",
    "ibrido": "Ibrido BM25 + Vector Search (k=3)",
    "rerank": "Ibrido + Rerank Cross-Encoder (k=12 -> top 3)",
    "graph": "GraphRAG leggero (k=3 base -> k=4 espanso via Grafo)"
}[scelta_rag_type]

st.sidebar.info(f"""
**Stato Sistema:**
- DB: `{db_label}`
- Collection: `{COLLECTION_NAME}`
- Chunk Size: `{scelta_chunk_size} token`
- Embedding: `{embedding_label}`
- Retriever: `{retriever_label}`
""")

st.title("🤖  ARIS: Assistente per Manutenzione Industriale")
st.caption("Supporto tecnico basato su RAG per Robot Fanuc R-30iB Mate")

# Inizializza retriever + chain (entrambi cached per env+chunk_size+rag_type)
try:
    rag_chain = init_rag_chain(scelta_env, scelta_chunk_size, scelta_rag_type)
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
        with st.spinner(f"L'IA ({scelta_env}) sta consultando i manuali..."):
            try:
                # Costruisce la cronologia degli ultimi 3 scambi (6 messaggi)
                # escludendo il messaggio appena aggiunto
                history_msgs = st.session_state.messages[:-1][-6:]
                history_str = "\n".join(
                    f"{m['role'].upper()}: {m['content']}"
                    for m in history_msgs
                ) if history_msgs else "(nessuna conversazione precedente)"

                response_stream = rag_chain.stream({
                    "question": prompt,
                    "history":  history_str,
                })
                
                # Estrae il primo chunk per innescare il caricamento ed il retrieval all'interno dello spinner
                try:
                    first_chunk = next(response_stream)
                except StopIteration:
                    first_chunk = ""
            except Exception as e:
                msg_errore = f"❌ Errore durante la generazione ({scelta_env}). "
                if scelta_env == "locale":
                    msg_errore += "Assicurati che LM Studio abbia il Server avviato su porta 1234."
                else:
                    msg_errore += "Verifica che OPENAI_API_KEY sia configurata nel file .env."
                st.error(f"{msg_errore}\n\nDettagli: {e}")
                st.stop()

        def stream_generator():
            if first_chunk:
                yield first_chunk
            for chunk in response_stream:
                yield chunk

        full_response = st.write_stream(stream_generator())
        st.session_state.messages.append({"role": "assistant", "content": full_response})