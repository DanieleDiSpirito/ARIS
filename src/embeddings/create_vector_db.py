import json
import os
import sys
import argparse
import shutil
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv()  

def create_documents_from_json(json_path):
    """Carica il JSON dei chunk e lo converte in oggetti Document di LangChain."""
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    documents = []
    for chunk in chunks:
        # Estraiamo il testo puro che verrà vettorizzato
        text = chunk.pop("text")
        
        # Tutto il resto (chunk_id, file_name, title, char_count, has_table, ecc.) 
        # diventa "metadato" utile per il filtering nel vector DB
        metadata = chunk
        
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
        
    return documents

def main():
    parser = argparse.ArgumentParser(description="Genera gli embeddings e popola ChromaDB")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale", 
                        help="Scegli tra 'locale' (BGE-M3) o 'cloud' (OpenAI)")
    parser.add_argument("--chunk_size", type=int, choices=[300, 700, 1000], default=700, 
                        help="Dimensione in token dei chunk da vettorizzare")
    parser.add_argument(
        "--metodo", "-m", 
        type=str, 
        choices=["euristico", "docling", "llamaparse", "qwen", "pdf4llm", "mineru"],
        default="docling",
        help="Metodo di estrazione da elaborare (euristico, docling, llamaparse, qwen, pdf4llm, mineru)."
    )
    args = parser.parse_args()
    
    file_path = os.path.join("data", "chunks", args.metodo, f"dataset_chunks_{args.env}_{args.chunk_size}.json")
    if not os.path.exists(file_path):
        print(f"❌ File non trovato: {file_path}")
        print("Assicurati di aver prima eseguito: python src/chunking/chunking.py")
        return
        
    print(f"📄 Caricamento chunk da: {file_path}")
    docs = create_documents_from_json(file_path)
    print(f"✅ Trovati {len(docs)} frammenti di testo pronti per l'embedding.")
    
    if args.env == "locale":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Inizializzazione modello locale: BAAI/bge-m3 su dispositivo '{device}'...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        persist_dir = os.path.join("vector_db", f"chroma_{args.metodo}_locale_{args.chunk_size}")
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
        persist_dir = os.path.join("vector_db", f"chroma_{args.metodo}_cloud_{args.chunk_size}")

    print(f"🗄️ Generazione embeddings e salvataggio DB in: {persist_dir}")
    print("⏳ L'operazione potrebbe richiedere alcuni minuti...")
    
    if os.path.exists(persist_dir):
        print("⚠️ Un database precedente esiste già in questa cartella. Verrà sovrascritto.")
        shutil.rmtree(persist_dir)
        
    os.makedirs(persist_dir, exist_ok=True)
    
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    # test di ricerca rapido per verificare la funzionalità
    print(f"🎉 Operazione completata! Vector DB salvato con successo in '{persist_dir}'.")
    print("Test rapido del DB:")
    
    query_test = "errore o allarme"
    results = db.similarity_search_with_score(query_test, k=1)
    if results:
        best_doc, score = results[0]
        print(f"\n🔍 Risultato di test per '{query_test}':")
        print(f"Titolo: {best_doc.metadata.get('title')}")
        print(f"Testo: {best_doc.page_content[:100]}...")
        print(f"Distanza: {score:.4f}")

if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == "embeddings":
        os.chdir("../..")
    elif os.path.basename(os.getcwd()) == "src":
        os.chdir("..")
        
    main()
