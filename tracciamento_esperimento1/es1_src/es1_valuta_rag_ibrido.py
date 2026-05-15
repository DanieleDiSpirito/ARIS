import pandas as pd
import chromadb
import argparse
import os
import re
from dotenv import load_dotenv

import json
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
# Import basato sui risultati della ricerca nel tuo ambiente
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma

# Importiamo gli stessi moduli usati per creare il vector db
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# --- CONFIGURAZIONE PERCORSI ISOLATA ESPERIMENTO 1 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Cartella es1_src
PROJECT_ROOT = os.path.dirname(BASE_DIR)              # Cartella tracciamento_esperimento1
# Il CSV va messo nella root della cartella tracciamento_esperimento1
TEST_FILE = os.path.join(PROJECT_ROOT, "test_questions_en.csv") 

class LangchainEmbeddingAdapter:
    """Adattatore per usare gli embedding di Langchain con il client nativo di ChromaDB"""
    def __init__(self, lc_embedder):
        self.lc_embedder = lc_embedder

    def __call__(self, input):
        return self.lc_embedder.embed_documents(input)

    def embed_query(self, text: str):
        return self.lc_embedder.embed_query(text)

    def embed_documents(self, texts: list):
        return self.lc_embedder.embed_documents(texts)

def carica_documenti_per_bm25(json_path):
    """Funzione di supporto: carica il JSON specifico per il BM25."""
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    documents = []
    for chunk in chunks:
        testo = chunk.get("text", "")
        metadati = {k: v for k, v in chunk.items() if k != "text"}
        documents.append(Document(page_content=testo, metadata=metadati))
    return documents

def controlla_corrispondenza(pagina_trovata, pagina_attesa):
    """Applica la logica Regex con tolleranza +-1 per verificare la pagina."""
    if pagina_trovata == pagina_attesa:
        return True
    try:
        f_match = re.search(r'\d+', pagina_trovata)
        e_match = re.search(r'\d+', pagina_attesa)
        if f_match and e_match:
            f_prefix = pagina_trovata[:f_match.start()]
            e_prefix = pagina_attesa[:e_match.start()]
            if f_prefix == e_prefix:
                f_num = int(f_match.group())
                e_num = int(e_match.group())
                if abs(f_num - e_num) <= 1:
                    return True
    except Exception:
        pass
    return False

def calcola_hit_rate(db_path, env):
    # 1. SETUP CHROMA E EMBEDDER
    client = chromadb.PersistentClient(path=db_path)
    
    if env == "locale":
        lc_embedder = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif env == "cloud":
        lc_embedder = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )

    # Identificazione collezione
    collections = [c.name for c in client.list_collections()]
    # Priorità alla collezione specifica dell'esperimento
    collection_name = "manuali_fanuc_es1" if "manuali_fanuc_es1" in collections else (
        collections[0] if collections else "langchain"
    )

    # 2. SETUP RETRIEVER VETTORIALE
    lc_chroma = Chroma(
        client=client, 
        collection_name=collection_name, 
        embedding_function=lc_embedder
    )
    retriever_vettoriale = lc_chroma.as_retriever(search_kwargs={"k": 5})

    # 3. SETUP RETRIEVER TESTUALE BM25 (Puntando alla cartella es1_chunks)
    db_basename = os.path.basename(db_path)
    parts = db_basename.split('_')
    json_filename = f"dataset_chunks_{parts[1]}_{parts[2]}.json" if len(parts) >= 3 else "dataset_chunks_locale_700.json"
    
    # MODIFICA PERCORSO: Cerca in ../es1_chunks
    json_path = os.path.join(PROJECT_ROOT, "es1_chunks", json_filename)
    
    if not os.path.exists(json_path):
        print(f"⚠️ JSON BM25 non trovato in: {json_path}. Uso solo Chroma.")
        retriever_ibrido = retriever_vettoriale
    else:
        docs_bm25 = carica_documenti_per_bm25(json_path)
        retriever_bm25 = BM25Retriever.from_documents(docs_bm25)
        retriever_bm25.k = 5
        retriever_ibrido = EnsembleRetriever(
            retrievers=[retriever_bm25, retriever_vettoriale], 
            weights=[0.4, 0.6]
        )

    # 4. VALUTAZIONE
    if not os.path.exists(TEST_FILE):
        print(f"❌ File di test non trovato in: {TEST_FILE}")
        return 0.0

    df = pd.read_csv(TEST_FILE)
    hits = 0

    for _, row in df.iterrows():
        results = retriever_ibrido.invoke(row['question'])[:3]

        trovato = False
        if results:
            for doc in results:
                meta = doc.metadata
                if meta and meta.get('file_name') == row['expected_file']:
                    found_p = str(meta.get('page')).strip()
                    # CONTROLLO TRACCIAMENTO: Verifichiamo anche la fonte originale arricchita
                    source_p = str(meta.get('original_source_page', found_p)).strip()
                    exp_p = str(row['expected_page']).strip()
                    
                    if controlla_corrispondenza(found_p, exp_p) or controlla_corrispondenza(source_p, exp_p):
                        trovato = True
                        break

        if trovato:
            hits += 1
        else:
            print(f"\n❌ Errore domanda: {row['id']}")
            print(f"Atteso: {row['expected_file']} - Pag: {row['expected_page']}")
            print("Top 3 trovati:")
            for i, doc in enumerate(results):
                p_fisica = doc.metadata.get('page')
                p_orig = doc.metadata.get('original_source_page', 'N/A')
                print(f"  {i+1}) Pag: {p_fisica} (Origine: {p_orig})")
            print("-" * 30)

    return (hits / len(df)) * 100

def main():
    parser = argparse.ArgumentParser(description="Valutatore Esperimento 1")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale")
    args = parser.parse_args()

    # MODIFICA PERCORSI: Cerca i DB in ../es1_vector_db
    db_list = [f"chroma_{args.env}_300", f"chroma_{args.env}_700", f"chroma_{args.env}_1000"]
    
    db_to_eval = []
    for db in db_list:
        db_path = os.path.join(PROJECT_ROOT, "es1_vector_db", db)
        if os.path.exists(db_path):
            db_to_eval.append((db, db_path))

    if not db_to_eval:
        print(f"⚠️ Nessun database trovato in {os.path.join(PROJECT_ROOT, 'es1_vector_db')}")
        return

    print(f"🔄 Avvio valutazione Esperimento 1 (Ambiente: {args.env})...\n")

    for db_name, db_path in db_to_eval:
        print(f"📊 Analisi: {db_name}")
        try:
            score = calcola_hit_rate(db_path, args.env)
            print(f"✅ Hit Rate@3: {score:.2f}%\n")
        except Exception as e:
            print(f"❌ Errore su {db_name}: {e}")

if __name__ == "__main__":
    main()