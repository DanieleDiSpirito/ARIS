import os
import argparse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def build_prompt():
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
    """Formatta i chunk recuperati e li STAMPA a schermo per debug."""
    print(f"\n--- 🔎 CHUNK RECUPERATI DAL VECTOR DB (PURO RAG) ---")
    print(f"Numero di chunk recuperati: {len(docs)}")
    
    formatted_chunks = []
    for i, doc in enumerate(docs):
        file_name = doc.metadata.get("file_name", "Documento Sconosciuto")
        page = doc.metadata.get("original_source_page", doc.metadata.get("page", "N/A"))
        
        # Stampa nel terminale per vedere i chunk che arrivano!
        print(f"\n[Chunk {i+1}] Da: {file_name} - Pag: {page}")
        # Stampiamo i primi 200 caratteri per capire cosa c'è dentro
        print(f"Testo: {doc.page_content[:200]}...") 
        
        chunk_str = f"--- INIZIO FONTE: {file_name} (Pagina: {page}) ---\n{doc.page_content}\n--- FINE FONTE ---\n"
        formatted_chunks.append(chunk_str)
        
    print("----------------------------------------------------\n")
    return "\n".join(formatted_chunks)

def setup_rag_chain(retriever, env="locale"):
    if env == "locale":
        print("🤖 Inizializzazione LLM: Locale (LM Studio su localhost:1234)")
        llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            temperature=0.0
        )
    elif env == "cloud":
        print("☁️ Inizializzazione LLM: Cloud (OpenRouter)")
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
    parser = argparse.ArgumentParser(description="Testa la RAG Pipeline (Puro) dal terminale")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale", 
                        help="Scegli tra 'locale' (LM Studio) o 'cloud' (OpenRouter)")
    parser.add_argument("--query", type=str, default="Cosa significa l'allarme SRVO-004?", 
                        help="La domanda da porre al sistema")
    args = parser.parse_args()

    print(f"🔄 Avvio test pipeline RAG PURO in modalità: {args.env.upper()}")
    
    db_path = os.path.join("vector_db", "chroma_locale_700")
    if not os.path.exists(db_path):
        print(f"❌ Database vettoriale non trovato in {db_path}. Esegui il test dalla cartella principale.")
        return

    print("🗄️ Caricamento Vector DB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = Chroma(persist_directory=db_path, embedding_function=embeddings, collection_name="langchain")
    
    # Questo è RAG PURO (Solo Dense Vector Search)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    try:
        chain = setup_rag_chain(retriever, env=args.env)
    except ValueError as e:
        print(e)
        return

    print(f"\n🗣️ Domanda: {args.query}")
    print("⏳ Recupero chunk e generazione risposta in corso...\n")
    
    risposta = answer_question(chain, args.query)
    
    print("================ RISPOSTA ================")
    print(risposta)
    print("==========================================")

if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == "rag_pipeline":
        os.chdir("../..")
    main()
