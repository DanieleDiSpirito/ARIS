"""
evaluate_rag.py
===============
Script di valutazione quantitativa del retrieval.
Legge test_questions.csv ed esegue il confronto tra:
  1. Vector Search (Ricerca semantica ChromaDB)
  2. BM25 Search (Ricerca lessicale)
  3. Hybrid Search (Vector + BM25 con EnsembleRetriever)
Calcola Hit Rate@k, Precision@k, MRR e tempi di esecuzione, stampando un report comparativo.
"""

import os
import sys
import time
import argparse
import pandas as pd
import json

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Import sicuro per EnsembleRetriever per gestire diverse versioni di langchain
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_classic.retrievers.ensemble import EnsembleRetriever

from retrieval_metrics import calcola_metriche_query, stampa_report

# Configura console Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def get_embeddings(env: str):
    """Carica il modello di embedding corretto in base all'ambiente."""
    if env == "locale":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif env == "cloud":
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"Ambiente {env} non supportato.")


def carica_documenti_da_chunks(chunks_dir: str, env: str, size: int) -> List[Document]:
    """Carica tutti i chunk corrispondenti a env/size per inizializzare BM25."""
    pattern = f"_chunks_{env}_{size}.json"
    chunk_files = [os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith(pattern)]
    
    docs = []
    for file_path in chunk_files:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        for chunk in chunks:
            testo = chunk.get("text", "")
            metadati = {k: v for k, v in chunk.items() if k != "text"}
            docs.append(Document(page_content=testo, metadata=metadati))
    return docs


def valuta_strategia(
    retriever, 
    df_questions: pd.DataFrame, 
    k: int, 
    tolleranza: int
) -> Dict[str, Any]:
    """Esegue il retrieval su tutte le domande del test set e calcola le metriche."""
    retrieved_metas = []
    tempi = []

    for _, row in df_questions.iterrows():
        t0 = time.perf_counter()
        # Esegue il retrieval top-k
        results = retriever.invoke(row['question'])[:k]
        tempi.append(time.perf_counter() - t0)

        # Convertiamo la lista di Document in metadati dict per il calcolo delle metriche
        metas = [doc.metadata for doc in results]
        retrieved_metas.append(metas)

    cats = df_questions['category'].tolist() if 'category' in df_questions.columns else None
    diffs = df_questions['difficulty'].tolist() if 'difficulty' in df_questions.columns else None

    return calcola_metriche_query(
        retrieved_metas=retrieved_metas,
        expected_files=df_questions['expected_file'].tolist(),
        expected_pages=df_questions['expected_page'].astype(str).tolist(),
        categories=cats,
        difficulties=diffs,
        k=k,
        tolleranza=tolleranza,
        tempi_retrieval=tempi
    )


def main():
    parser = argparse.ArgumentParser(description="Valutatore RAG quantitativo (Vector, BM25, Hybrid)")
    parser.add_argument("--size", type=int, default=500, help="Dimensione dei chunk usata per il Vector DB")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale",
                        help="Scegli tra 'locale' o 'cloud'")
    parser.add_argument("--k", type=int, default=3, help="Numero di chunk da recuperare per query")
    parser.add_argument("--tolleranza", type=int, default=1, help="Tolleranza sul numero di pagina ±N")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(base_dir, "tests", "test_questions.csv")
    db_path = os.path.join(base_dir, "vector_db", f"chroma_general_{args.env}_{args.size}")
    chunks_dir = os.path.join(base_dir, "output_data", "chunks")

    if not os.path.exists(test_file):
        print(f"❌ File delle domande di test '{test_file}' non trovato. Crealo prima di valutare.")
        sys.exit(1)

    if not os.path.exists(db_path):
        print(f"❌ Database vettoriale '{db_path}' non trovato. Esegui prima vector_store.py.")
        sys.exit(1)

    df_questions = pd.read_csv(test_file)
    if len(df_questions) == 0:
        print("⚠️ Il file delle domande è vuoto.")
        sys.exit(0)

    print(f"🔄 Avvio valutazione RAG generico — env={args.env} size={args.size} k={args.k} tolleranza=±{args.tolleranza}")
    print(f"   Caricate {len(df_questions)} domande di test.")

    # 1. Inizializza Embeddings e Chroma
    embeddings = get_embeddings(args.env)
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    # 2. Inizializza Retriever Vettoriale
    retriever_vector = db.as_retriever(search_kwargs={"k": args.k})

    # 3. Inizializza Retriever BM25
    docs_bm25 = carica_documenti_da_chunks(chunks_dir, args.env, args.size)
    if not docs_bm25:
        print("❌ Impossibile caricare i chunk per inizializzare BM25.")
        sys.exit(1)
        
    retriever_bm25 = BM25Retriever.from_documents(docs_bm25)
    retriever_bm25.k = args.k

    # 4. Inizializza Retriever Ibrido (Ensemble 50% Vector + 50% BM25)
    retriever_hybrid = EnsembleRetriever(
        retrievers=[retriever_bm25, retriever_vector],
        weights=[0.5, 0.5]
    )

    strategie = [
        ("PURE_VECTOR", retriever_vector),
        ("PURE_BM25", retriever_bm25),
        ("HYBRID_SEARCH", retriever_hybrid)
    ]

    report_righe = []

    for nome, retriever in strategie:
        print(f"\n⏳ Valutazione strategia: {nome} ...")
        t_start = time.perf_counter()
        metriche = valuta_strategia(retriever, df_questions, args.k, args.tolleranza)
        t_dur = time.perf_counter() - t_start
        
        stampa_report(nome, metriche)
        
        report_righe.append({
            "Strategia": nome,
            "Hit Rate@k (%)": round(metriche.get("hit_rate_k", 0.0), 2),
            "Precision@k (%)": round(metriche.get("precision_k", 0.0), 2),
            "Recall@k (%)": round(metriche.get("recall_k", 0.0), 2),
            "MRR": round(metriche.get("mrr", 0.0), 4),
            "Tempo Medio (s)": round(metriche.get("tempo_medio_s", 0.0), 4),
            "Tempo Totale (s)": round(t_dur, 2)
        })

    # Stampa tabella riassuntiva in console
    df_report = pd.DataFrame(report_righe)
    print("\n📊 TABELLA COMPARATIVA DEL RETRIEVAL:")
    print("=" * 95)
    print(df_report.to_string(index=False))
    print("=" * 95)

    # Scrittura del report finale in Markdown
    output_report_path = os.path.join(base_dir, "output_data", f"benchmark_retrieval_{args.env}_{args.size}.md")
    md_table = df_report.to_markdown(index=False)
    
    report_md = f"""# Benchmark di Retrieval Generico (RAG)

Questo report riassume le metriche di accuratezza del **Retrieval RAG** (Vector, BM25, Ibrido) calcolate sul manuale di test.

## Parametri di Esecuzione
- **Ambiente**: {args.env.upper()}
- **Dimensione Chunk**: {args.size} token
- **Numero Domande**: {len(df_questions)}
- **Top-k selezionati**: {args.k}
- **Tolleranza pagina**: ±{args.tolleranza}

## Tabella Comparativa

{md_table}

---
*Report generato automaticamente il {time.strftime('%d-%m-%Y %H:%M:%S')}.*
"""

    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(f"\n✅ Report di valutazione salvato con successo in: {output_report_path}\n")


if __name__ == "__main__":
    main()
