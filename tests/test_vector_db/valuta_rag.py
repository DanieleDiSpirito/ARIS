import pandas as pd
import chromadb
import argparse
import os
from dotenv import load_dotenv

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

def calcola_hit_rate(db_path, env, test_file):
    client = chromadb.PersistentClient(path=db_path)
    emb_fn = get_embedding_function(env)

    # Determiniamo il nome della collection. Langchain di default usa "langchain".
    collections = [c.name for c in client.list_collections()]
    if not collections:
        print(f"❌ Nessuna collezione trovata in {db_path}")
        return 0.0

    collection_name = "manuali_fanuc"
    if "manuali_fanuc" not in collections and "langchain" in collections:
        collection_name = "langchain"
    elif collection_name not in collections:
        collection_name = collections[0]

    collection = client.get_collection(name=collection_name, embedding_function=emb_fn)

    # Caricamento domande
    if not os.path.exists(test_file):
        print(f"❌ File di test non trovato: {test_file}")
        return 0.0

    df = pd.read_csv(test_file)
    if len(df) == 0:
        return 0.0

    hits = 0

    for _, row in df.iterrows():
        # Embedding manuale della query per evitare ambiguità con l'adapter
        query_embedding = emb_fn.embed_query(row['question'])

        # Query al database con embedding pre-calcolato
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        # Verifica nei metadati
        trovato = False
        if results['metadatas'] and len(results['metadatas']) > 0:
            for meta in results['metadatas'][0]:
                if meta and str(meta.get('page')) == str(row['expected_page']) and meta.get('file_name') == row['expected_file']:
                    trovato = True
                    break

        if trovato:
            hits += 1

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