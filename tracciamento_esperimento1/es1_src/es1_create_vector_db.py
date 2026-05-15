import json
import os
import argparse
import shutil
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()  

def create_documents_from_json(json_path):
    """Carica il JSON dei chunk e lo converte in oggetti Document di LangChain."""
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    documents = []
    for chunk in chunks:
        # Estraiamo il testo puro che verrà vettorizzato
        text = chunk.pop("text")
        
        # Tutto il resto (chunk_id, original_source_page, file_name, ecc.) 
        # diventa metadato utile per il filtering e il tracciamento
        metadata = chunk
        
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
        
    return documents

def main():
    parser = argparse.ArgumentParser(description="Genera gli embeddings e popola ChromaDB per l'Esperimento 1")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale", 
                        help="Scegli tra 'locale' (BGE-M3) o 'cloud' (OpenAI)")
    parser.add_argument("--chunk_size", type=int, choices=[300, 700, 1000], default=700, 
                        help="Dimensione in token dei chunk da vettorizzare")
    args = parser.parse_args()
    
    # MODIFICA PERCORSI: Punta alla cartella es1_chunks
    file_path = os.path.join("..", "es1_chunks", f"dataset_chunks_{args.env}_{args.chunk_size}.json")
    
    if not os.path.exists(file_path):
        print(f"❌ File non trovato: {file_path}")
        print("Assicurati di aver prima eseguito: python es1_chunking.py")
        return
        
    print(f"📄 Caricamento chunk da: {file_path}")
    docs = create_documents_from_json(file_path)
    print(f"✅ Trovati {len(docs)} frammenti di testo pronti per l'embedding.")
    
    # 2. Configura il Modello di Embedding (Cloud o Locale)
    if args.env == "locale":
        print("🧠 Inizializzazione modello locale: BAAI/bge-m3...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # MODIFICA PERCORSI: Salva in es1_vector_db
        persist_dir = os.path.join("..", "es1_vector_db", f"chroma_locale_{args.chunk_size}")
    else:
        print("☁️ Inizializzazione modello cloud: text-embedding-3-small...")
        if "OPENAI_API_KEY" not in os.environ:
            print("❌ ERRORE: Per usare il cloud devi impostare la variabile d'ambiente OPENAI_API_KEY.")
            return
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )
        # MODIFICA PERCORSI: Salva in es1_vector_db
        persist_dir = os.path.join("..", "es1_vector_db", f"chroma_cloud_{args.chunk_size}")

    # 3. Costruisci il Vector Database (ChromaDB)
    print(f"🗄️ Generazione embeddings e salvataggio DB in: {persist_dir}")
    print("⏳ L'operazione potrebbe richiedere alcuni minuti (CPU-bound)...")
    
    # Svuota la directory se il database esiste già per evitare duplicati nell'esperimento
    if os.path.exists(persist_dir):
        print(f"⚠️ Un database precedente esiste in {persist_dir}. Verrà sovrascritto per pulizia.")
        shutil.rmtree(persist_dir)
        
    os.makedirs(persist_dir, exist_ok=True)
    
    # Chroma calcola gli embedding e popola la collezione
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="manuali_fanuc_es1" # Nome collezione specifico per l'esperimento
    )
    
    print(f"🎉 Operazione completata! Vector DB Esperimento 1 salvato in '{persist_dir}'.")
    
    # 4. Micro-test di ricerca per verificare il tracciamento metadati
    print("\n--- 🧪 TEST TRACCIAMENTO METADATI ---")
    query_test = "SRVO-004 alarm"
    results = db.similarity_search(query_test, k=1)
    if results:
        meta = results[0].metadata
        print(f"Query test: '{query_test}'")
        print(f"Pagina Fisica: {meta.get('page')}")
        print(f"Pagina Originale (Tracciata): {meta.get('original_source_page')}")
        print(f"Testo (estratto): {results[0].page_content[:100]}...")

if __name__ == "__main__":
    # Eseguire stando dentro la cartella es1_src
    main()