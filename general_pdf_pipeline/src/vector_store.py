"""
vector_store.py
===============
Script generico per il caricamento dei chunk, la creazione degli embeddings
e il popolamento del database vettoriale ChromaDB in locale.
"""

import os
import sys
import json
import argparse
import shutil
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Configura console Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def load_documents_from_json(json_path: str) -> List[Document]:
    """Carica i chunk da un file JSON e li mappa in oggetti Document di LangChain."""
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    documents = []
    for chunk in chunks:
        text = chunk.pop("text")
        # I campi rimanenti diventano metadati del chunk
        metadata = chunk
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
        
    return documents


def main():
    parser = argparse.ArgumentParser(description="Generatore di Vector DB generico")
    parser.add_argument("--size", type=int, default=500, help="Dimensione dei chunk da caricare")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale",
                        help="Scegli tra 'locale' (BGE-M3) o 'cloud' (OpenAI)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks_dir = os.path.join(base_dir, "output_data", "chunks")
    persist_dir = os.path.join(base_dir, "vector_db", f"chroma_general_{args.env}_{args.size}")

    if not os.path.exists(chunks_dir):
        print(f"❌ Cartella chunk '{chunks_dir}' non trovata.")
        sys.exit(1)

    # Identifica tutti i file di chunk corrispondenti all'ambiente e alla taglia scelti
    pattern = f"_chunks_{args.env}_{args.size}.json"
    chunk_files = [os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith(pattern)]

    if not chunk_files:
        print(f"❌ Nessun file di chunk trovato per env={args.env} e size={args.size} in '{chunks_dir}'.")
        print("Esegui prima: python general_pdf_pipeline/src/chunker.py")
        sys.exit(1)

    all_docs = []
    print(f"📂 Caricamento file di chunk trovati:")
    for f_path in chunk_files:
        print(f"   - {os.path.basename(f_path)}")
        all_docs.extend(load_documents_from_json(f_path))

    print(f"✅ Trovati complessivamente {len(all_docs)} frammenti da indicizzare.")

    # Configurazione del modello di Embedding
    if args.env == "locale":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Inizializzazione modello locale: BAAI/bge-m3 su dispositivo '{device}'...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        print("☁️ Inizializzazione modello cloud: text-embedding-3-small via OpenRouter...")
        if "OPENAI_API_KEY" not in os.environ:
            print("❌ ERRORE: La variabile d'ambiente OPENAI_API_KEY non è impostata.")
            sys.exit(1)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )

    # Rimozione eventuale DB preesistente per pulizia
    if os.path.exists(persist_dir):
        print(f"⚠️ Un database preesistente in '{persist_dir}' verrà rimosso e sovrascritto.")
        shutil.rmtree(persist_dir)
        
    os.makedirs(persist_dir, exist_ok=True)

    print(f"🗄️ Generazione embeddings e scrittura DB in: {persist_dir}")
    print("⏳ Attendi completamento...")
    
    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"🎉 Database vettoriale completato e salvato in '{persist_dir}'!")
    
    # Test di verifica rapido
    print("\n🔍 Esecuzione test di ricerca rapido...")
    query_test = "troubleshooting"
    results = db.similarity_search_with_score(query_test, k=1)
    if results:
        best_doc, score = results[0]
        print(f"   - Match trovato: '{best_doc.metadata.get('title', 'Generale')}' (Pagina {best_doc.metadata.get('page')})")
        print(f"   - Score distanza: {score:.4f}")
        print(f"   - Testo estratto: {best_doc.page_content[:120].strip()}...")


if __name__ == "__main__":
    main()
