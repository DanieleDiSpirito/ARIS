import os
import argparse
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

COLLECTION_NAME = "langchain"

def get_db_path(env: str, chunk_size: int, metodo: str = "pdf4llm") -> str:
    return os.path.join("vector_db", f"chroma_{metodo}_{env}_{chunk_size}")

def get_embeddings(env: str):
    if env == "locale":
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif env == "cloud":
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"env non valido: '{env}'.")

def main():
    parser = argparse.ArgumentParser(description="Ispeziona la Knowledge Base (Vector DB)")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale")
    parser.add_argument("--chunk_size", type=int, default=700)
    parser.add_argument("--metodo", type=str, default="pdf4llm")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Elenca i documenti e i relativi conteggi di chunk")
    group.add_argument("--search", type=str, help="Cerca chunk che contengono una parola chiave (case-insensitive)")
    group.add_argument("--show-file", type=str, help="Mostra tutti i chunk estratti da un determinato file")
    group.add_argument("--sample", type=int, help="Mostra un campione di N chunk casuali")
    
    parser.add_argument("--page", type=int, help="Filtra per pagina specifica (da usare insieme a --show-file)", default=None)
    args = parser.parse_args()

    # Imposta la root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    os.chdir(project_root)

    db_path = get_db_path(args.env, args.chunk_size, args.metodo)
    if not os.path.exists(db_path):
        print(f"❌ Database non trovato: {db_path}")
        return

    embeddings = get_embeddings(args.env)
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    all_data = db.get()
    documents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])
    ids = all_data.get("ids", [])

    if args.list:
        counts = {}
        for meta in metadatas:
            f = meta.get("file_name", "Sconosciuto")
            counts[f] = counts.get(f, 0) + 1
        print(f"\n📂 Knowledge Base in '{db_path}':")
        print(f"Totale chunk: {len(documents)}")
        for doc, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {doc}: {count} chunk")

    elif args.search:
        keyword = args.search.lower()
        results = []
        for i, text in enumerate(documents):
            if keyword in text.lower():
                results.append((ids[i], metadatas[i], text))
        
        print(f"\n🔎 Trovati {len(results)} chunk contenenti '{args.search}':")
        for idx, (cid, meta, text) in enumerate(results[:10]):
            f = meta.get("file_name", "Sconosciuto")
            p = meta.get("page", "N/A")
            print(f"\n[{idx+1}] ID: {cid} | File: {f} | Pagina: {p}")
            print("-" * 50)
            print(text[:300] + ("..." if len(text) > 300 else ""))
        if len(results) > 10:
            print(f"\n... e altri {len(results) - 10} chunk.")

    elif args.show_file:
        target_file = args.show_file.lower()
        results = []
        for i, meta in enumerate(metadatas):
            f = meta.get("file_name", "").lower()
            if target_file in f:
                if args.page is None or int(meta.get("page", -1)) == args.page:
                    results.append((ids[i], meta, documents[i]))
        
        print(f"\n📄 Trovati {len(results)} chunk per il file '{args.show_file}'" + (f" (Pagina {args.page})" if args.page else "") + ":")
        for idx, (cid, meta, text) in enumerate(results[:10]):
            print(f"\n[{idx+1}] ID: {cid} | Pagina: {meta.get('page', 'N/A')}")
            print("-" * 50)
            print(text[:400] + ("..." if len(text) > 400 else ""))
        if len(results) > 10:
            print(f"\n... e altri {len(results) - 10} chunk.")

    elif args.sample:
        import random
        n = min(args.sample, len(documents))
        indices = random.sample(range(len(documents)), n)
        print(f"\n🎲 Campione di {n} chunk casuali:")
        for idx, i in enumerate(indices):
            meta = metadatas[i]
            print(f"\n[{idx+1}] ID: {ids[i]} | File: {meta.get('file_name')} | Pagina: {meta.get('page')}")
            print("-" * 50)
            print(documents[i][:400] + ("..." if len(documents[i]) > 400 else ""))

if __name__ == "__main__":
    main()
