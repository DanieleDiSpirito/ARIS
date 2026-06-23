"""
query_rag.py
============
Interfaccia interattiva da riga di comando (CLI) per testare il retrieval.
Permette di effettuare ricerche semantiche, lessicali o ibride sul database generalizzato.
"""

import os
import sys
import argparse
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Import sicuro per EnsembleRetriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_classic.retrievers.ensemble import EnsembleRetriever

from evaluate_rag import get_embeddings, carica_documenti_da_chunks

# Configura console Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Interfaccia di ricerca RAG interattiva")
    parser.add_argument("--size", type=int, default=500, help="Dimensione dei chunk del database da caricare")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale",
                        help="Scegli tra 'locale' o 'cloud'")
    parser.add_argument("--k", type=int, default=3, help="Numero di risultati da mostrare")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "vector_db", f"chroma_general_{args.env}_{args.size}")
    chunks_dir = os.path.join(base_dir, "output_data", "chunks")

    if not os.path.exists(db_path):
        print(f"❌ Database vettoriale '{db_path}' non trovato. Esegui prima vector_store.py.")
        sys.exit(1)

    print(f"🔄 Inizializzazione motori di ricerca (env={args.env}, size={args.size})...")
    
    # 1. Carica Embeddings e Chroma
    embeddings = get_embeddings(args.env)
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    retriever_vector = db.as_retriever(search_kwargs={"k": args.k})

    # 2. Carica BM25
    print("⏳ Caricamento dei chunk per BM25...")
    docs_bm25 = carica_documenti_da_chunks(chunks_dir, args.env, args.size)
    if not docs_bm25:
        print("❌ Impossibile caricare i chunk per inizializzare BM25.")
        sys.exit(1)
    retriever_bm25 = BM25Retriever.from_documents(docs_bm25)
    retriever_bm25.k = args.k

    # 3. Carica Ibrido
    retriever_hybrid = EnsembleRetriever(
        retrievers=[retriever_bm25, retriever_vector],
        weights=[0.5, 0.5]
    )

    print("\n🎉 Motori di ricerca pronti!")
    print("Comandi disponibili:")
    print("  - Digita la tua domanda per cercare con il metodo Ibrido (default)")
    print("  - Digita ':v <domanda>' per cercare SOLO con la Vector Search (semantica)")
    print("  - Digita ':b <domanda>' per cercare SOLO con la BM25 Search (parole chiave)")
    print("  - Digita 'exit' o 'quit' per uscire\n")

    while True:
        try:
            query_raw = input("🗣️ Inserisci domanda > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nUscita.")
            break

        if not query_raw:
            continue

        if query_raw.lower() in ["exit", "quit"]:
            print("Uscita. Arrivederci!")
            break

        # Riconoscimento del comando di ricerca specifico
        search_mode = "IBRIDO (Vector + BM25)"
        retriever_attivo = retriever_hybrid
        query = query_raw

        if query_raw.startswith(":v "):
            search_mode = "VETTORIALE (Semantica)"
            retriever_attivo = retriever_vector
            query = query_raw[3:].strip()
        elif query_raw.startswith(":b "):
            search_mode = "BM25 (Parole chiave)"
            retriever_attivo = retriever_bm25
            query = query_raw[3:].strip()

        if not query:
            print("⚠️ Inserisci una domanda valida.")
            continue

        print(f"\n🔍 Esecuzione ricerca con metodo {search_mode} per: '{query}'...")
        t0 = time.perf_counter()
        results = retriever_attivo.invoke(query)[:args.k]
        durata = time.perf_counter() - t0

        print(f"⏱️ Tempo di ricerca: {durata:.4f} secondi. Trovati {len(results)} risultati:")
        print("=" * 80)

        for idx, doc in enumerate(results, start=1):
            meta = doc.metadata
            file_name = meta.get("file_name", "N/A")
            page = meta.get("page", "N/A")
            section = meta.get("section", "N/A")
            title = meta.get("title", "N/A")
            doc_id = meta.get("document_id", "N/A")

            print(f"👉 RISULTATO {idx} [Doc ID: {doc_id} | File: {file_name} | Pagina: {page}]")
            print(f"   Sezione: {section} - Titolo: {title}")
            print("-" * 80)
            print(doc.page_content.strip())
            print("=" * 80)
        print()


if __name__ == "__main__":
    main()
