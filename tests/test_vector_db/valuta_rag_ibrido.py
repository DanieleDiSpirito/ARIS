"""
valuta_rag_ibrido.py
====================
Valutazione del retriever ibrido (BM25 + Vector) con metriche complete
dalla Fase 13 (INFO.md):
  Hit Rate@k, Precision@k, Recall@k, MRR,
  breakdown per categoria e difficoltà, tempo medio retrieval.
"""
import time
import re
import json
import pandas as pd
import chromadb
import argparse
import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from retrieval_metrics import calcola_metriche_query, stampa_report

load_dotenv()

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR    = os.path.dirname(BASE_DIR)
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


def carica_documenti_per_bm25(json_path: str) -> list:
    """Carica il JSON di chunks e restituisce una lista di Document per BM25."""
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    documents = []
    for chunk in chunks:
        testo   = chunk.get("text", "")
        metadati = {k: v for k, v in chunk.items() if k != "text"}
        documents.append(Document(page_content=testo, metadata=metadati))
    return documents


def _pagine_match_tol(found_p: str, exp_p: str, tolleranza: int = 1) -> bool:
    """Confronto pagine con tolleranza ±N."""
    found_p, exp_p = str(found_p).strip(), str(exp_p).strip()
    if found_p == exp_p:
        return True
    try:
        f_m = re.search(r'\d+', found_p)
        e_m = re.search(r'\d+', exp_p)
        if f_m and e_m:
            if found_p[:f_m.start()] == exp_p[:e_m.start()]:
                if abs(int(f_m.group()) - int(e_m.group())) <= tolleranza:
                    return True
    except Exception:
        pass
    return False


def valuta_db(db_path: str, env: str, test_file: str, k: int = 3,
              tolleranza: int = 1, debug: bool = False) -> dict:
    """
    Retrieval ibrido (BM25 + Chroma) con calcolo completo delle metriche.

    Parametri
    ---------
    tolleranza : tolleranza di pagina ±N (default 1)
    debug      : se True, stampa i dettagli delle domande non trovate
    """
    client = chromadb.PersistentClient(path=db_path)

    # --- Embedding ---
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

    collections = [c.name for c in client.list_collections()]
    if not collections:
        print(f"❌ Nessuna collezione trovata in {db_path}")
        return {}

    collection_name = "manuali_fanuc" if "manuali_fanuc" in collections else (
        "langchain" if "langchain" in collections else collections[0]
    )

    # --- Retriever vettoriale ---
    lc_chroma = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=lc_embedder
    )
    retriever_vettoriale = lc_chroma.as_retriever(search_kwargs={"k": k + 2})

    # --- Retriever BM25 ---
    db_basename  = os.path.basename(db_path)
    parts        = db_basename.split('_')
    json_filename = (f"dataset_chunks_{parts[1]}_{parts[2]}.json"
                     if len(parts) >= 3 else "dataset_chunks_locale_700.json")
    json_path = os.path.join(PROJECT_ROOT, "data/chunks", json_filename)

    if not os.path.exists(json_path):
        print(f"⚠️ JSON non trovato per BM25: {json_path}. Uso solo Chroma.")
        retriever_ibrido = retriever_vettoriale
    else:
        docs_bm25 = carica_documenti_per_bm25(json_path)
        retriever_bm25 = BM25Retriever.from_documents(docs_bm25)
        retriever_bm25.k = k + 2
        retriever_ibrido = EnsembleRetriever(
            retrievers=[retriever_bm25, retriever_vettoriale],
            weights=[0.4, 0.6]
        )

    if not os.path.exists(test_file):
        print(f"❌ File di test non trovato: {test_file}")
        return {}

    df = pd.read_csv(test_file)
    if len(df) == 0:
        return {}

    all_metas = []
    tempi     = []

    for _, row in df.iterrows():
        t0      = time.perf_counter()
        results = retriever_ibrido.invoke(row['question'])[:k]
        tempi.append(time.perf_counter() - t0)

        # Converti Document → dict metadati (formato atteso da retrieval_metrics)
        metas = [doc.metadata for doc in results]
        all_metas.append(metas)

        if debug:
            trovato = any(
                meta and meta.get('file_name') == row['expected_file']
                and _pagine_match_tol(str(meta.get('page', '')), str(row['expected_page']), tolleranza)
                for meta in metas
            )
            if not trovato:
                print(f"\n❌ Errore sulla domanda: {row['id']}")
                print(f"   Atteso: File '{row['expected_file']}' — Pagina '{row['expected_page']}'")
                print(f"   Top {k} trovati (tolleranza ±{tolleranza}):")
                for i, meta in enumerate(metas):
                    print(f"     {i+1}) File: '{meta.get('file_name')}' — Pagina: '{meta.get('page')}'")
                print("-" * 50)

    cats  = df['category'].tolist()   if 'category'   in df.columns else None
    diffs = df['difficulty'].tolist() if 'difficulty' in df.columns else None

    return calcola_metriche_query(
        retrieved_metas=all_metas,
        expected_files=df['expected_file'].tolist(),
        expected_pages=df['expected_page'].astype(str).tolist(),
        categories=cats,
        difficulties=diffs,
        k=k,
        tolleranza=tolleranza,
        tempi_retrieval=tempi,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Valuta il RAG ibrido (BM25 + Vettoriale)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--env",        type=str, choices=["locale", "cloud"], default="locale")
    parser.add_argument("--db",         type=str, nargs='*')
    parser.add_argument("--lang",       type=str, default="it", choices=["it", "en"])
    parser.add_argument("--k",          type=int, default=3,
                        help="Numero di chunk recuperati per query")
    parser.add_argument("--tolleranza", type=int, default=1,
                        help="Tolleranza di pagina ±N (0 = corrispondenza esatta)")
    parser.add_argument("--debug",      action="store_true",
                        help="Stampa i dettagli delle domande non trovate")
    args = parser.parse_args()

    test_file = os.path.join(TESTS_DIR, f"test_questions_{args.lang}.csv")

    if args.db:
        db_list = args.db
    else:
        db_list = [f"chroma_{args.env}_300", f"chroma_{args.env}_700", f"chroma_{args.env}_1000"]
        if args.env == "locale":
            db_list.extend(["db_nitro", "db_standard", "db_exacto"])

    db_to_eval = [
        (db, os.path.join(PROJECT_ROOT, "vector_db", db))
        for db in db_list
        if os.path.exists(os.path.join(PROJECT_ROOT, "vector_db", db))
    ]

    if not db_to_eval:
        print(f"⚠️ Nessun vector DB trovato per l'ambiente '{args.env}'.")
        print(f"Usa --db o esegui 'python src/embeddings/create_vector_db.py --env {args.env}'")
        return

    print(
        f"🔄 Avvio valutazione RAG ibrido — env={args.env} "
        f"lang={args.lang} k={args.k} tolleranza=±{args.tolleranza}"
        f"{' [DEBUG]' if args.debug else ''}\n"
    )

    for db_name, db_path in db_to_eval:
        print(f"⏳ Valutazione DB: {db_name} ...")
        try:
            metriche = valuta_db(
                db_path, args.env, test_file,
                k=args.k, tolleranza=args.tolleranza, debug=args.debug
            )
            if metriche:
                stampa_report(db_name, metriche)
            else:
                print(f"⚠️ Nessuna metrica calcolata per {db_name}\n")
        except Exception as e:
            print(f"❌ Errore durante la valutazione di {db_name}: {e}\n")


if __name__ == "__main__":
    main()