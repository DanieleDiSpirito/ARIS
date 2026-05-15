import os
import argparse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

COLLECTION_NAME = "langchain"

# Mappa env+chunk_size → cartella del vector DB
def get_db_path(env: str, chunk_size: int) -> str:
    return os.path.join("vector_db", f"chroma_{env}_{chunk_size}")


def get_embeddings(env: str):
    """
    Seleziona il modello di embedding coerente con quello usato durante l'ingestion:
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
    - System: regole rigide di sicurezza, istruzioni per leggere le tabelle, regole di formattazione.
    - Human: contesto recuperato + domanda dell'operatore.
    """
    system_template = """Sei un assistente tecnico esperto per robot Fanuc.
Devi rispondere ESCLUSIVAMENTE usando il "Contesto tecnico recuperato".

ATTENZIONE ALLA LETTURA DEI DATI:
Il contesto spesso contiene tabelle formattate con il carattere "|".
Se l'operatore chiede le specifiche di un codice (es. A05B-...), analizza riga per riga queste tabelle per trovare la corrispondenza.
Se trovi il dato, estrailo e rendilo discorsivo.
Solo se, dopo aver letto attentamente tutte le righe e le tabelle, sei ASSOLUTAMENTE CERTO che il dato non esista, scrivi testualmente:
"La documentazione disponibile non contiene informazioni sufficienti per indicare una procedura sicura. Si consiglia di consultare un tecnico qualificato."

Non inventare mai procedure, codici errore, valori tecnici o bypass di sicurezza.

REGOLE DI FORMATTAZIONE:
- Se la domanda è su un ALLARME, ERRORE o GUASTO:
  Rispondi usando un elenco numerato: 1. Significato, 2. Possibili cause, 3. Controlli, 4. Azioni, 5. Fonte documentale.
- Per DOMANDE SU SPECIFICHE o COMPONENTI (es. schede, codici parte):
  NON usare l'elenco dei guasti. Scrivi una risposta discorsiva chiara con le caratteristiche richieste.
  Aggiungi sempre alla fine, su una nuova riga: "Fonte documentale: [Nome File e Pagina]".
"""

    human_template = """Contesto tecnico recuperato:
{context}

Domanda dell'operatore:
{question}"""

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])


def format_docs_with_sources(docs):
    """Formatta i chunk recuperati per il prompt e li stampa a schermo per debug."""
    if not docs:
        return "Nessun dato trovato nel contesto."

    print(f"\n--- 🔎 CHUNK INVIATI ALL'LLM ({len(docs)} totali) ---")
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
    """Configura la pipeline RAG collegando retriever, prompt e LLM."""
    if env == "locale":
        print("🤖 LLM: Locale (LM Studio su localhost:1234)")
        llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            temperature=0.0  # Zero creatività per massima precisione tecnica
        )
    elif env == "cloud":
        print("☁️ LLM: Cloud (OpenRouter)")
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("❌ ERRORE: Variabile OPENAI_API_KEY non trovata nel file .env")
        llm = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0
        )
    else:
        raise ValueError("Il parametro env deve essere 'locale' o 'cloud'")

    prompt = build_prompt()

    # Pipeline RAG: retriever → formattazione → prompt → LLM → output
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def answer_question(rag_chain, question):
    try:
        return rag_chain.invoke(question)
    except Exception as e:
        return f"❌ Errore di connessione o generazione. Dettagli: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Testa la RAG Pipeline (Puro Vector Search) dal terminale")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale",
                        help="Scegli tra 'locale' (LM Studio) o 'cloud' (OpenRouter)")
    parser.add_argument("--chunk_size", type=int, default=700,
                        help="Dimensione dei chunk usata durante l'ingestion (default: 700). "
                             "Determina quale cartella del Vector DB viene caricata.")
    parser.add_argument("--query", type=str, default="Cosa significa l'allarme SRVO-004?",
                        help="La domanda da porre al sistema")
    args = parser.parse_args()

    db_path = get_db_path(args.env, args.chunk_size)

    print(f"🔄 Avvio test pipeline RAG (Puro) — env: {args.env.upper()} | chunk_size: {args.chunk_size}")

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
    retriever = db.as_retriever(search_kwargs={"k": 3})

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