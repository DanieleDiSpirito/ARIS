import os
import argparse
from dotenv import load_dotenv

# Librerie LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- CONFIGURAZIONE PERCORSO ASSOLUTO ---
DB_PATH = r"C:\Users\vince\Documents\GitHub\ARIS\vector_db\chroma_locale_700"

def build_prompt():
    """Costruisce il prompt istruendo il modello a leggere le tabelle."""
    
    system_template = """Sei un assistente tecnico esperto per robot Fanuc.
Devi rispondere ESCLUSIVAMENTE usando il "Contesto tecnico recuperato".

ATTENZIONE ALLA LETTURA DEI DATI:
Il contesto spesso contiene tabelle formattate con il carattere "|". 
Se l'operatore chiede le specifiche di un codice (es. A05B-...), analizza riga per riga queste tabelle per trovare la corrispondenza. 
Se trovi il dato, estrailo e rendilo discorsivo. 
Solo se, dopo aver letto attentamente tutte le righe e le tabelle, sei ASSOLUTAMENTE CERTO che il dato non esista, scrivi testualmente: "La documentazione disponibile non contiene informazioni sufficienti."

REGOLE DI FORMATTAZIONE:
- Se la domanda è su un ALLARME, ERRORE o GUASTO: 
  Rispondi usando un elenco numerato: 1. Significato, 2. Possibili cause, 3. Controlli, 4. Azioni, 5. Fonte documentale.
- Per DOMANDE SU SPECIFICHE o COMPONENTI (es. schede): 
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
    """Formatta i chunk recuperati includendo metadati chiari."""
    if not docs:
        return "Nessun dato trovato nel contesto."
    
    formatted_chunks = []
    for doc in docs:
        file_name = doc.metadata.get("file_name", "Documento Sconosciuto")
        page = doc.metadata.get("original_source_page", doc.metadata.get("page", "N/A"))
        chunk_str = f"--- INIZIO FONTE: {file_name} (Pagina: {page}) ---\n{doc.page_content}\n--- FINE FONTE ---\n"
        formatted_chunks.append(chunk_str)
        
    return "\n".join(formatted_chunks)

def setup_rag_chain(retriever, env="locale"):
    """Configura la pipeline RAG collegando LLM e Prompt."""
    if env == "locale":
        print("🤖 LLM: Locale (LM Studio su localhost:1234)")
        llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            temperature=0.0 # Zero creatività per massima precisione tecnica
        )
    elif env == "cloud":
        print("☁️ LLM: Cloud (OpenRouter)")
        llm = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0
        )
    else:
        raise ValueError("Il parametro env deve essere 'locale' o 'cloud'")

    prompt = build_prompt()

    # Pipeline RAG
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    parser = argparse.ArgumentParser(description="Testa la RAG Pipeline")
    parser.add_argument("--env", choices=["locale", "cloud"], default="locale")
    parser.add_argument("--query", type=str, default="Cosa significa l'allarme SRVO-004?")
    args = parser.parse_args()

    print(f"🔄 Avvio test pipeline RAG")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ ERRORE: Percorso database non trovato: {DB_PATH}")
        return

    print(f"🗄️ Caricamento Vector DB...")
    
    # Torniamo a usare BGE-M3 (1024 dimensioni)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    db = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings, 
        collection_name="langchain"  # Il nome usato dal tuo script di ingestion
    )

    # 👇 DEFINIZIONE DEL RETRIEVER AGGIUNTA QUI 👇
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # --- DEBUG RETRIEVAL (Indispensabile per verificare Chroma) ---
    print(f"🔎 Ricerca nel manuale per: {args.query}")
    docs = retriever.invoke(args.query)
    print(f"✅ Chunk recuperati dal database: {len(docs)}")
    for i, d in enumerate(docs):
        # Stampa i primi 80 caratteri di ogni chunk per verifica visiva
        print(f"   [{i+1}] Fonte: {d.metadata.get('file_name')} | Testo: {d.page_content[:80].replace('\n', ' ')}...")

    # Generazione Risposta
    chain = setup_rag_chain(retriever, env=args.env)
    
    print("\n⏳ Generazione risposta LLM in corso...\n")
    risposta = chain.invoke(args.query)
    
    print("================ RISPOSTA ================")
    print(risposta)
    print("==========================================")

if __name__ == "__main__":
    main()