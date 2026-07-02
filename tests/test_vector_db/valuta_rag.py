import time
import re
import pandas as pd
import chromadb
import argparse
import os
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from retrieval_metrics import calcola_metriche_query, stampa_report

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

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
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lc_embedder = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': device},
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


def _pagine_match(found_p: str, exp_p: str, tolleranza: int) -> bool:
    """Confronto pagine con tolleranza opzionale ±N."""
    found_p, exp_p = str(found_p).strip(), str(exp_p).strip()
    if found_p == exp_p:
        return True
    if tolleranza == 0:
        return False
    try:
        f_m = re.search(r'\d+', found_p)
        e_m = re.search(r'\d+', exp_p)
        if f_m and e_m and found_p[:f_m.start()] == exp_p[:e_m.start()]:
            if abs(int(f_m.group()) - int(e_m.group())) <= tolleranza:
                return True
    except Exception:
        pass
    return False


def valuta_db(db_path: str, env: str, test_file: str, k: int = 3,
              tolleranza: int = 1, debug: bool = False) -> dict:
    """
    Esegue il retrieval vettoriale per ogni domanda del test set e
    calcola l'insieme completo di metriche (Hit Rate@k, Precision@k,
    Recall@k, MRR, breakdown per categoria e difficoltà, tempo medio).

    Parametri
    ---------
    tolleranza : tolleranza di pagina ±N (default 1)
    debug      : se True, stampa i dettagli delle domande non trovate
    """
    client = chromadb.PersistentClient(path=db_path)
    emb_fn = get_embedding_function(env)

    collections = [c.name for c in client.list_collections()]
    if not collections:
        print(f"❌ Nessuna collezione trovata in {db_path}")
        return {}

    collection_name = "manuali_fanuc"
    if "manuali_fanuc" not in collections and "langchain" in collections:
        collection_name = "langchain"
    elif collection_name not in collections:
        collection_name = collections[0]

    collection = client.get_collection(name=collection_name, embedding_function=emb_fn)

    if not os.path.exists(test_file):
        print(f"❌ File di test non trovato: {test_file}")
        return {}

    df = pd.read_csv(test_file)
    if len(df) == 0:
        return {}

    all_metas = []
    tempi     = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Valutazione {os.path.basename(db_path)}"):
        t0 = time.perf_counter()
        query_embedding = emb_fn.embed_query(row['question'])
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        tempi.append(time.perf_counter() - t0)

        metas = results['metadatas'][0] if results['metadatas'] else []
        all_metas.append(metas)

        if debug:
            trovato = any(
                m and m.get('file_name') == row['expected_file']
                and _pagine_match(str(m.get('page', '')), str(row['expected_page']), tolleranza)
                for m in metas
            )
            if not trovato:
                print(f"\n❌ Errore sulla domanda: {row['id']}")
                print(f"   Atteso : File '{row['expected_file']}' — Pagina '{row['expected_page']}'")
                print(f"   Top {k} trovati (tolleranza ±{tolleranza}):")
                for i, m in enumerate(metas):
                    print(f"     {i+1}) File: '{m.get('file_name')}' — Pagina: '{m.get('page')}'")
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
        description="Valuta il RAG vettoriale su DB specifici",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--env",        type=str, choices=["locale", "cloud"], default="locale",
                        help="Ambiente usato per gli embedding")
    parser.add_argument("--db",         type=str, nargs='*',
                        help="Nomi delle cartelle DB (es. chroma_cloud_700)")
    parser.add_argument("--lang",       type=str, default="it", choices=["it", "en"],
                        help="Lingua del test")
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
        db_list = []
        vector_db_dir = os.path.join(PROJECT_ROOT, "vector_db")
        if os.path.exists(vector_db_dir):
            for item in os.listdir(vector_db_dir):
                if item.startswith("chroma_") and item.endswith(f"_{args.env}_700"):
                    db_list.append(item)
        if not db_list:
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
        f"🔄 Avvio valutazione RAG (vettoriale) — env={args.env} "
        f"lang={args.lang} k={args.k} tolleranza=±{args.tolleranza}"
        f"{' [DEBUG]' if args.debug else ''}\n"
    )

    summary_results = []

    for db_name, db_path in db_to_eval:
        print(f"⏳ Valutazione DB: {db_name} ...")
        try:
            metriche = valuta_db(
                db_path, args.env, test_file,
                k=args.k, tolleranza=args.tolleranza, debug=args.debug
            )
            if metriche:
                stampa_report(db_name, metriche)
                summary_results.append({
                    "DB Name": db_name,
                    "Hit Rate@k (%)": round(metriche.get("hit_rate_k", 0.0), 2),
                    "Precision@k (%)": round(metriche.get("precision_k", 0.0), 2),
                    "Recall@k (%)": round(metriche.get("recall_k", 0.0), 2),
                    "MRR": round(metriche.get("mrr", 0.0), 4),
                    "Tempo Medio (s)": round(metriche.get("tempo_medio_s", 0.0), 4)
                })
            else:
                print(f"⚠️ Nessuna metrica calcolata per {db_name}\n")
        except Exception as e:
            print(f"❌ Errore durante la valutazione di {db_name}: {e}\n")

    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        print("\n📊 TABELLA COMPARATIVA DI RETRIEVAL VETTORIALE:")
        print("=" * 90)
        print(df_summary.to_string(index=False))
        print("=" * 90)

        # Salva in Markdown
        metrics_dir = os.path.join(PROJECT_ROOT, "data", "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        report_path = os.path.join(metrics_dir, "benchmark_retrieval_vettoriale.md")
        
        md_table = df_summary.to_markdown(index=False)
        report_content = f"""# Benchmark Retrieval Vettoriale (ARIS)

Questo report riassume le metriche di accuratezza del **Retrieval Vettoriale** per i database estratti con i diversi metodi di parsing, valutati sul test set di domande.

## Tabella Comparativa

{md_table}

---
*Report generato automaticamente dallo script `valuta_rag.py` il 20 Giugno 2026.*
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"✅ Report riassuntivo salvato in: {report_path}\n")


if __name__ == "__main__":
    main()