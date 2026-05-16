import os
import argparse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

COLLECTION_NAME = "langchain"

# Mappa env+chunk_size → cartella del vector DB
def get_db_path(env: str, chunk_size: int) -> str:
    """Restituisce il percorso corretto del Vector DB in base all'ambiente e alla dimensione dei chunk."""
    return os.path.join("vector_db", f"chroma_{env}_{chunk_size}")


def get_embeddings(env: str):
    """Seleziona il modello di embedding coerente con quello usato durante l'ingestion.

    - locale → BAAI/bge-m3  (1024 dimensioni, HuggingFace, locale)
    - cloud  → text-embedding-3-small  (1536 dimensioni, OpenAI)
    """
    if env == "locale":
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif env == "cloud":
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"env non valido: '{env}'. Scegli tra 'locale' o 'cloud'.")


def build_prompt():
    """Costruisce il prompt con System+Human separati.
    Gestisce tre situazioni:
      1. Messaggi di chat generica (saluti, ringraziamenti, follow-up)
      2. Riferimenti alla conversazione precedente
      3. Domande tecniche sui manuali Fanuc
    """
    system_template = """Sei ARIS, un assistente tecnico esperto per robot Fanuc, amichevole e disponibile.

Hai accesso a:
- Un "Contesto tecnico recuperato" dai manuali Fanuc
- Una "Cronologia conversazione" con i messaggi precedenti

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGOLA 1 — MESSAGGI DI CHAT GENERICA
Se il messaggio dell'operatore è un saluto (es. "ciao", "buongiorno", "grazie"), 
una presentazione, o una domanda non tecnica (es. "cosa sai fare?", "chi sei?"),
rispondi in modo naturale e cordiale SENZA usare il contesto tecnico.
Presentati brevemente come assistente tecnico per robot Fanuc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGOLA 2 — RIFERIMENTI ALLA CONVERSAZIONE PRECEDENTE
Se il messaggio fa riferimento a uno scambio precedente (es. "quella risposta non va bene",
"puoi approfondire?", "e per l'asse 2?", "come dicevi prima..."), usa la Cronologia 
conversazione per capire il contesto, poi rispondi o correggi di conseguenza.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGOLA 3 — DOMANDE TECNICHE SUI MANUALI
Se la domanda è tecnica (allarmi, procedure, specifiche hardware, connessioni):
- Rispondi ESCLUSIVAMENTE usando il "Contesto tecnico recuperato".
- Il contesto contiene tabelle con "|": analizza riga per riga per trovare la corrispondenza.
- Solo se sei ASSOLUTAMENTE CERTO che il dato non esista nella documentazione, scrivi:
  "La documentazione disponibile non contiene informazioni sufficienti. Si consiglia di consultare un tecnico qualificato."
- Non inventare mai procedure, codici errore, valori tecnici o bypass di sicurezza.

FORMATTAZIONE per domande tecniche:
- ALLARME / ERRORE / GUASTO → elenco numerato: 1. Significato 2. Possibili cause 3. Controlli 4. Azioni 5. Fonte documentale.
- SPECIFICHE / COMPONENTI → risposta discorsiva.

IMPORTANTE: Devi SEMPRE includere la fonte alla fine di QUALSIASI risposta tecnica. La pagina DEVE essere specificata sempre.
Scrivi ESATTAMENTE in questo formato su una nuova riga alla fine:
"Fonte documentale: [nome_file.pdf] (Pagina: [numero_pagina])"
"""

    human_template = """Cronologia conversazione (ultimi scambi):
{history}

Contesto tecnico recuperato:
{context}

Messaggio dell'operatore:
{question}"""

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])


def format_docs_with_sources(docs):
    if not docs:
        return "Nessun dato trovato nel contesto."

    print(f"\n--- 🔎 CHUNK INVIATI ALL'LLM — Ibrido BM25+Vector ({len(docs)} totali) ---")
    formatted_chunks = []
    for i, doc in enumerate(docs):
        file_name = doc.metadata.get("file_name", "Documento Sconosciuto")
        page = doc.metadata.get("original_source_page", doc.metadata.get("page", "N/A"))

        print(f"  [{i+1}] {file_name} | Pag: {page} | Testo: {doc.page_content[:100].replace(chr(10), ' ')}...")

        chunk_str = f"--- INIZIO FONTE: {file_name} (Pagina: {page}) ---\n{doc.page_content}\n--- FINE FONTE ---\n"
        formatted_chunks.append(chunk_str)

    print("---------------------------------------------------\n")
    return "\n".join(formatted_chunks)


def setup_rag_chain(retriever, env="locale"):
    """Configura la pipeline RAG collegando retriever, prompt e LLM.

    La chain accetta un dict: {"question": str, "history": str}
    - question : domanda corrente dell'operatore
    - history  : ultimi scambi formattati come stringa (può essere vuota)
    """
    if env == "locale":
        print(f"🤖 LLM: Locale (server su localhost:1234)")
        local_model = os.getenv("LOCAL_LLM_MODEL", None)
        base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
        llm = ChatOpenAI(
            model=local_model,
            base_url=base_url,
            api_key="lm-studio",
            temperature=0.1
        )
    elif env == "cloud":
        print("☁️ LLM: Cloud (OpenRouter)")
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("❌ ERRORE: Variabile OPENAI_API_KEY non trovata nel file .env")
        llm = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1
        )
    else:
        raise ValueError("Il parametro env deve essere 'locale' o 'cloud'")

    prompt = build_prompt()

    # Estrae la sola domanda dal dict di input per il retriever
    get_question = RunnableLambda(lambda x: x["question"] if isinstance(x, dict) else x)
    get_history  = RunnableLambda(lambda x: x.get("history", "") if isinstance(x, dict) else "")

    # Pipeline RAG: retriever usa solo {question}, il prompt riceve context + question + history
    rag_chain = (
        {
            "context":  get_question | retriever | format_docs_with_sources,
            "question": get_question,
            "history":  get_history,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def answer_question(rag_chain, question):
    """Invia la domanda alla catena e gestisce eventuali errori di connessione."""
    try:
        return rag_chain.invoke(question)
    except Exception as e:
        return f"❌ Errore di connessione o generazione. Dettagli: {str(e)}"


def build_hybrid_retriever(db: Chroma, k: int = 3) -> EnsembleRetriever:
    """Costruisce un retriever ibrido combinando BM25 (keyword) e Chroma (semantic).

    Args:
        db: istanza Chroma già caricata.
        k: numero di documenti da recuperare per ciascun retriever.

    Returns:
        EnsembleRetriever con peso 50% BM25 + 50% Vector Search.
    """
    # 1. Retriever denso (Vector Search)
    chroma_retriever = db.as_retriever(search_kwargs={"k": k})

    # 2. Retriever sparso (BM25 / Keyword Search)
    # Carichiamo tutti i documenti dal Chroma DB per inizializzare BM25.
    # NOTA: per dataset molto grandi, in produzione si userebbe un indice BM25
    # persistito separatamente (es. Elasticsearch/OpenSearch).
    all_data = db.get()
    if not all_data or not all_data.get("documents"):
        raise RuntimeError("❌ Nessun documento trovato nel database vettoriale. Impossibile costruire BM25.")

    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
        if text
    ]
    print(f"✅ BM25 inizializzato su {len(docs)} chunk...")
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    # 3. Ensemble: combina i due retriever con pesi uguali
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]   # 50% keyword + 50% semantic
    )
    return ensemble


def main():
    parser = argparse.ArgumentParser(description="Testa la RAG Pipeline Ibrida (BM25 + Vector Search) dal terminale")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale",
                        help="Scegli tra 'locale' (LM Studio) o 'cloud' (OpenRouter)")
    parser.add_argument("--chunk_size", type=int, default=700,
                        help="Dimensione dei chunk usata durante l'ingestion (default: 700). "
                             "Determina quale cartella del Vector DB viene caricata.")
    parser.add_argument("--query", type=str, default="Cosa significa l'allarme SRVO-004?",
                        help="La domanda da porre al sistema")
    args = parser.parse_args()

    db_path = get_db_path(args.env, args.chunk_size)

    print(f"🔄 Avvio test pipeline RAG (Ibrido) — env: {args.env.upper()} | chunk_size: {args.chunk_size}")

    if not os.path.exists(db_path):
        print(f"❌ ERRORE: Database vettoriale non trovato in '{db_path}'.")
        print("   👉 Assicurati di eseguire lo script dalla root del progetto (ARIS/).")
        return

    print(f"🗄️ Caricamento Vector DB da: {db_path}")
    embeddings = get_embeddings(args.env)
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # Costruisce il retriever ibrido BM25 + Vector
    print("🤝 Costruzione Retriever Ibrido (50% BM25 + 50% Vector Search)...")
    try:
        retriever = build_hybrid_retriever(db, k=3)
    except RuntimeError as e:
        print(e)
        return

    try:
        chain = setup_rag_chain(retriever, env=args.env)
    except ValueError as e:
        print(e)
        return

    print(f"\n🗣️ Domanda: {args.query}")
    print("⏳ Generazione risposta in corso...\n")

    risposta = answer_question(chain, args.query)

    print("================ RISPOSTA ================")
    print(risposta)
    print("==========================================")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    os.chdir(project_root)

    main()
