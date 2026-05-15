import os
import argparse
from dotenv import load_dotenv

# Usiamo la stessa libreria sia per il cloud che per LM Studio!
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Se vuoi testarlo da terminale, ci serve Chroma per il retriever fittizio
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def build_prompt():
    """Costruisce il prompt imponendo le rigide regole di sicurezza industriale."""
    template = """Sei un assistente tecnico esperto per la manutenzione di macchinari industriali, in particolare per i robot serie Fanuc.
Rispondi SOLO usando le informazioni contenute nel contesto fornito. 
Se il contesto non contiene informazioni sufficienti o non è pertinente alla domanda, devi dire chiaramente: 
"La documentazione disponibile non contiene informazioni sufficienti per indicare una procedura sicura. Si consiglia di consultare un tecnico qualificato."

Non inventare mai procedure, codici errore, valori tecnici o bypass di sicurezza.
Quando possibile, fornisci una risposta strutturata in:
1. Significato del problema
2. Possibili cause
3. Controlli consigliati
4. Azioni successive
5. Fonte documentale (Elenca Nome File e Pagina estratti dal contesto)

Contesto tecnico recuperato:
{context}

Domanda dell'operatore:
{question}
"""
    return ChatPromptTemplate.from_template(template)

def format_docs_with_sources(docs):
    """Formatta i chunk recuperati per mostrare il testo e i metadati al modello."""
    formatted_chunks = []
    for doc in docs:
        file_name = doc.metadata.get("file_name", "Documento Sconosciuto")
        # Usa original_source_page se disponibile, altrimenti la page fisica
        page = doc.metadata.get("original_source_page", doc.metadata.get("page", "N/A"))
        
        chunk_str = f"--- INIZIO FONTE: {file_name} (Pagina: {page}) ---\n{doc.page_content}\n--- FINE FONTE ---\n"
        formatted_chunks.append(chunk_str)
        
    return "\n".join(formatted_chunks)

def setup_rag_chain(retriever, env="locale"):
    """
    Collega il retriever, il prompt e il modello LLM.
    Lo switch 'env' determina se usare il server locale (LM Studio) o il Cloud (OpenRouter).
    """
    if env == "locale":
        print("🤖 Inizializzazione LLM: Locale (LM Studio su localhost:1234)")
        llm = ChatOpenAI(
            base_url="http://localhost:1234/v1", # L'indirizzo del server locale di LM Studio
            api_key="lm-studio",                 # Chiave fittizia richiesta dalla libreria
            temperature=0.0                      # Nessuna "creatività" per evitare allucinazioni
        )
    elif env == "cloud":
        print("☁️ Inizializzazione LLM: Cloud (OpenRouter)")
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("❌ ERRORE: Variabile OPENAI_API_KEY non trovata nel file .env")
        llm = ChatOpenAI(
            model="openai/gpt-3.5-turbo", # Sostituisci con il modello OpenRouter che preferisci
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0
        )
    else:
        raise ValueError("Il parametro env deve essere 'locale' o 'cloud'")

    prompt = build_prompt()

    # Costruzione della pipeline RAG
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
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

# =====================================================================
# BLOCO MAIN PER TEST DA TERMINALE
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Testa la RAG Pipeline dal terminale")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale", 
                        help="Scegli tra 'locale' (LM Studio) o 'cloud' (OpenRouter)")
    parser.add_argument("--query", type=str, default="Cosa significa l'allarme SRVO-004?", 
                        help="La domanda da porre al sistema")
    args = parser.parse_args()

    print(f"🔄 Avvio test pipeline RAG in modalità: {args.env.upper()}")
    
    # 1. Inizializziamo un retriever rapido per il test (punta al tuo DB a 700 token)
    db_path = os.path.join("vector_db", "chroma_locale_700") # Adatta il percorso se necessario
    if not os.path.exists(db_path):
        print(f"❌ Database vettoriale non trovato in {db_path}. Esegui il test dalla cartella principale.")
        return

    print("🗄️ Caricamento Vector DB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = Chroma(persist_directory=db_path, embedding_function=embeddings, collection_name="manuali_fanuc_es1")
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 2. Configura la catena
    try:
        chain = setup_rag_chain(retriever, env=args.env)
    except ValueError as e:
        print(e)
        return

    # 3. Esegue la domanda
    print(f"\n🗣️ Domanda: {args.query}")
    print("⏳ Generazione risposta in corso...\n")
    
    risposta = answer_question(chain, args.query)
    
    print("================ RISPOSTA ================")
    print(risposta)
    print("==========================================")

if __name__ == "__main__":
    # Assicura che l'esecuzione avvenga dalla root del progetto (ARIS)
    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")
        
    main()
    