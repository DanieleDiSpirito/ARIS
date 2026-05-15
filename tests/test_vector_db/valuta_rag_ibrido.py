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

# Determiniamo i percorsi in modo robusto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)

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

def get_embedding_function(env):
    if env == "locale":
        lc_embedder = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return LangchainEmbeddingAdapter(lc_embedder)
    elif env == "cloud":
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY non è impostata nell'ambiente.")

        lc_embedder = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )
        return LangchainEmbeddingAdapter(lc_embedder)
    else:
        raise ValueError(f"Ambiente {env} non supportato.")

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

def calcola_hit_rate(db_path, env, test_file):
    # 1. SETUP CHROMA NATIVO E LANGCHAIN EMBEDDER (Il tuo setup originale)
    client = chromadb.PersistentClient(path=db_path)
    
    if env == "locale":
        lc_embedder = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif env == "cloud":
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY non è impostata nell'ambiente.")
        lc_embedder = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )

    # Trova il nome della collection
    collections = [c.name for c in client.list_collections()]
    if not collections:
        print(f"❌ Nessuna collezione trovata in {db_path}")
        return 0.0

    collection_name = "manuali_fanuc" if "manuali_fanuc" in collections else (
        "langchain" if "langchain" in collections else collections[0]
    )

    # 2. SETUP RETRIEVER VETTORIALE (Langchain wrapper sul tuo client Chroma)
    lc_chroma = Chroma(
        client=client, 
        collection_name=collection_name, 
        embedding_function=lc_embedder
    )
    retriever_vettoriale = lc_chroma.as_retriever(search_kwargs={"k": 5})

    # 3. SETUP RETRIEVER TESTUALE BM25
    # Ricaviamo il nome del JSON dal nome della cartella del DB (es. chroma_locale_700 -> dataset_chunks_locale_700.json)
    db_basename = os.path.basename(db_path)
    parts = db_basename.split('_')
    json_filename = f"dataset_chunks_{parts[1]}_{parts[2]}.json" if len(parts) >= 3 else "dataset_chunks_locale_700.json"
    json_path = os.path.join(PROJECT_ROOT, "data/chunks", json_filename) # Aggiusta il percorso se i JSON sono in una sottocartella
    
    if not os.path.exists(json_path):
        print(f"⚠️ JSON non trovato per BM25: {json_path}. Uso solo Chroma.")
        retriever_ibrido = retriever_vettoriale
    else:
        docs_bm25 = carica_documenti_per_bm25(json_path)
        retriever_bm25 = BM25Retriever.from_documents(docs_bm25)
        retriever_bm25.k = 5
        # 4. CREAZIONE DELL'ENSEMBLE (IL MOTORE IBRIDO)
        retriever_ibrido = EnsembleRetriever(
            retrievers=[retriever_bm25, retriever_vettoriale], 
            weights=[0.4, 0.6]
        )

    # 5. VALUTAZIONE CON LA TUA LOGICA REGEX
    if not os.path.exists(test_file):
        print(f"❌ File di test non trovato: {test_file}")
        return 0.0

    df = pd.read_csv(test_file)
    if len(df) == 0: return 0.0

    hits = 0

    for _, row in df.iterrows():
        # Eseguiamo la query sul motore ibrido anziché solo su Chroma
        results = retriever_ibrido.invoke(row['question'])[:3] # Prendiamo i Top 3 finali

        trovato = False
        if results:
            for doc in results:
                meta = doc.metadata
                if meta and meta.get('file_name') == row['expected_file']:
                    found_p = str(meta.get('page')).strip()
                    exp_p = str(row['expected_page']).strip()
                    
                    # 1. Controllo esatto
                    if found_p == exp_p:
                        trovato = True
                        break
                    
                    # 2. Il tuo Controllo tolleranza +- 1 con Regex
                    try:
                        f_match = re.search(r'\d+', found_p)
                        e_match = re.search(r'\d+', exp_p)
                        
                        if f_match and e_match:
                            f_prefix = found_p[:f_match.start()]
                            e_prefix = exp_p[:e_match.start()]
                            
                            if f_prefix == e_prefix:
                                f_num = int(f_match.group())
                                e_num = int(e_match.group())
                                if abs(f_num - e_num) <= 1:
                                    trovato = True
                                    break
                    except Exception:
                        pass 

        if trovato:
            hits += 1
        else:
            print(f"\n❌ Errore sulla domanda: {row['id']}")
            print(f"Atteso: File '{row['expected_file']}' - Pagina '{row['expected_page']}'")
            print("Trovato nei top 3:")
            for i, doc in enumerate(results):
                print(f"  {i+1}) File: '{doc.metadata.get('file_name')}' - Pagina: '{doc.metadata.get('page')}'")
            print("-" * 40)

    return (hits / len(df)) * 100

def main():
    parser = argparse.ArgumentParser(description="Valuta il RAG su DB specifici")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale",
                        help="Ambiente usato per gli embedding (locale o cloud)")
    parser.add_argument("--db", type=str, nargs='*',
                        help="Nomi delle cartelle DB da valutare (es. chroma_cloud_700). Se non specificato, valuta quelli predefiniti.")
    parser.add_argument("--lang", type=str, default="it", choices=["it", "en"],
                        help="Lingua del test ('it' o 'en')")

    args = parser.parse_args()

    test_file = os.path.join(TESTS_DIR, f"test_questions_{args.lang}.csv")

    # Se l'utente non specifica i DB, cerchiamo quelli generati per l'ambiente corrente
    if args.db:
        db_list = args.db
    else:
        db_list = [f"chroma_{args.env}_300", f"chroma_{args.env}_700", f"chroma_{args.env}_1000"]
        # DB legacy
        if args.env == "locale":
            db_list.extend(["db_nitro", "db_standard", "db_exacto"])

    # Controlliamo quali DB esistono effettivamente
    db_to_eval = []
    for db in db_list:
        db_path = os.path.join(PROJECT_ROOT, "vector_db", db)
        if os.path.exists(db_path):
            db_to_eval.append((db, db_path))

    if not db_to_eval:
        print(f"⚠️ Nessun vector DB trovato per la valutazione nell'ambiente '{args.env}'.")
        print(f"Usa --db per specificare nomi custom, oppure esegui prima 'python src/embeddings/create_vector_db.py --env {args.env}'")
        return

    print(f"🔄 Avvio valutazione RAG per ambiente '{args.env}'...\n")

    for db_name, db_path in db_to_eval:
        print(f"📊 Valutazione DB: {db_name}")
        try:
            score = calcola_hit_rate(db_path, args.env, test_file)
            print(f"✅ Risultato {db_name}: Hit Rate@3 = {score:.2f}%\n")
        except Exception as e:
            print(f"❌ Errore durante la valutazione di {db_name}: {e}\n")

if __name__ == "__main__":
    main()