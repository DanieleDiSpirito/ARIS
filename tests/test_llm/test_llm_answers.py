import os
import sys
import time
import argparse
import pandas as pd
import re
from dotenv import load_dotenv

def _pagine_match_tol(found_p: str, exp_p: str, tolleranza: int) -> bool:
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

def _estrai_fonti_llm(risposta: str) -> list:
    """Estrae tutte le coppie (file, pagina) dalla risposta dell'LLM."""
    pattern = r'([a-zA-Z0-9_\-\.]+\.pdf)(?:[^0-9]{0,30}?(?:pagina|page|pag|p)\.?\s*[:]?\s*([a-zA-Z0-9\-]*\d+))?'
    matches = re.findall(pattern, risposta, re.IGNORECASE)
    
    fonti = []
    for f_name, p_num in matches:
        fonti.append((f_name.strip(), p_num.strip() if p_num else ""))
    return fonti

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src", "rag_pipeline"))

import rag_pipeline_hybrid
import rag_pipeline_rerank
import rag_pipeline_graph
from rag_pipeline_hybrid import get_db_path, get_embeddings, COLLECTION_NAME
from langchain_chroma import Chroma

load_dotenv()
# Disabilita il tracing di LangSmith per evitare errori di limite di quota nei benchmark
os.environ["LANGCHAIN_TRACING_V2"] = "false"

def _is_model_not_loaded_error(error: Exception) -> bool:
    """Verifica se l'errore è dovuto al modello non caricato in LM Studio o errori di rete (es. 400)."""
    err_msg = str(error).lower()
    keywords = ["not loaded", "no model", "400", "bad request", "ejected", "connection refused", "failed to connect", "connection error"]
    return any(kw in err_msg for kw in keywords)

def valuta_llm(env: str, lang: str, chunk_size: int = 700, max_questions: int = None, tolleranza: int = 1, model: str = None, rag_type: str = "ibrido", metodo: str = "pdf4llm"):
    test_file = os.path.join(TESTS_DIR, f"test_questions_{lang}.csv")
    if not os.path.exists(test_file):
        print(f"❌ File di test non trovato: {test_file}")
        return

    df = pd.read_csv(test_file)
    if max_questions:
        # Campiona casualmente in modo riproducibile e ordina per indice originale
        df = df.sample(n=min(max_questions, len(df)), random_state=42).sort_index()

    db_path = os.path.join(PROJECT_ROOT, get_db_path(env, chunk_size, metodo))
    if not os.path.exists(db_path):
        print(f"❌ Database non trovato in: {db_path}")
        return

    # Determina il vero nome del modello
    actual_model = model
    if not actual_model:
        if env == "locale":
            actual_model = os.getenv("LOCAL_LLM_MODEL", "local_model")
        elif env == "cloud":
            actual_model = "openai/gpt-4o-mini"

    print(f"🗄️ Caricamento DB e Retriever ({env}, chunk: {chunk_size}, lingua: {lang}, RAG: {rag_type}, Modello: {actual_model})...")
    embedder = get_embeddings(env)
    lc_chroma = Chroma(
        persist_directory=db_path,
        collection_name=COLLECTION_NAME,
        embedding_function=embedder
    )
    if rag_type == "puro":
        retriever = lc_chroma.as_retriever(search_kwargs={"k": 5})
        rag_chain = rag_pipeline_hybrid.setup_rag_chain(retriever, env=env, model_name=model)
    elif rag_type == "ibrido":
        retriever = rag_pipeline_hybrid.build_hybrid_retriever(lc_chroma, k=5)
        rag_chain = rag_pipeline_hybrid.setup_rag_chain(retriever, env=env, model_name=model)
    elif rag_type == "rerank":
        ensemble = rag_pipeline_rerank.build_hybrid_retriever(lc_chroma, k=10)
        retriever = rag_pipeline_rerank.HybridRerankRetriever(ensemble, top_n=5)
        rag_chain = rag_pipeline_rerank.setup_rag_chain(retriever, env=env, model_name=model)
    elif rag_type == "graph":
        retriever = rag_pipeline_graph.build_graph_retriever(lc_chroma, k_base=15, top_n=5)
        rag_chain = rag_pipeline_graph.setup_rag_chain(retriever, env=env, model_name=model)
    else:
        raise ValueError(f"rag_type non valido: '{rag_type}'")

    risultati = []
    hit_count = 0
    no_page_count = 0
    miss_count = 0
    total_valid = 0
    
    # Per breakdown
    cat_stats = {}
    diff_stats = {}
    
    print(f"\n🚀 Inizio test su {len(df)} domande (Tolleranza pagina: ±{tolleranza})...\n")
    for index, row in df.iterrows():
        domanda = row['question']
        print(f"[{index+1}/{len(df)}] Q: {domanda}")
        
        t0 = time.time()
        while True:
            try:
                risposta = rag_chain.invoke({
                    "question": domanda,
                    "history": ""
                })
                stato = "OK"
                break
            except Exception as e:
                if env == "locale" and _is_model_not_loaded_error(e):
                    print(f"\n⚠️ [ERRORE LM STUDIO] Il modello locale sembra non essere caricato o è andato in crash.")
                    print(f"Dettaglio errore: {e}")
                    print("Assicurati che LM Studio sia attivo e che il modello sia caricato correttamente.")
                    input("Premi [INVIO] dopo aver caricato/riavviato il modello per riprovare...")
                    print("Ripristino esecuzione della domanda corrente...\n")
                    continue
                else:
                    risposta = f"ERRORE: {str(e)}"
                    stato = "ERRORE"
                    print(risposta)
                    break
            
        t1 = time.time()
        tempo = round(t1 - t0, 2)
        
        expected_file = str(row.get('expected_file', '')).strip()
        expected_page = str(row.get('expected_page', '')).strip()
        
        fonti_llm = _estrai_fonti_llm(risposta)
        
        # Controlla se almeno una delle fonti fa match con quella attesa
        is_hit = False
        match_found = False
        for file_llm, pag_llm in fonti_llm:
            match_file = (file_llm.lower() == expected_file.lower()) if expected_file and file_llm else False
            match_page = _pagine_match_tol(pag_llm, expected_page, tolleranza) if expected_page and pag_llm else False
            if match_file and match_page:
                match_found = True
                break
        
        match_status = "NO"
        
        if expected_file != 'nan' and expected_file != '':
            total_valid += 1
            if not fonti_llm:
                match_status = "NO PAGE"
                no_page_count += 1
            elif match_found:
                is_hit = True
                match_status = "SI"
                hit_count += 1
            else:
                match_status = "MISS"
                miss_count += 1
                
            # Calcola breakdown per cat e diff
            cat = row.get('category', 'Sconosciuta')
            diff = row.get('difficulty', 'Sconosciuta')
            if cat not in cat_stats:
                cat_stats[cat] = {"hits": 0, "total": 0}
            if diff not in diff_stats:
                diff_stats[diff] = {"hits": 0, "total": 0}
                
            cat_stats[cat]["total"] += 1
            diff_stats[diff]["total"] += 1
            if is_hit:
                cat_stats[cat]["hits"] += 1
                diff_stats[diff]["hits"] += 1
        
        files_str = ", ".join([f for f, p in fonti_llm])
        pags_str = ", ".join([p for f, p in fonti_llm])
        
        risultati.append({
            "id": row.get('id', index),
            "domanda": domanda,
            "risposta_llm": risposta,
            "llm_file_trovato": files_str,
            "llm_pag_trovata": pags_str,
            "expected_file": expected_file,
            "expected_page": expected_page,
            "match": match_status,
            "tempo_sec": tempo,
            "stato": stato
        })
        
        icon = "✅" if match_status == "SI" else ("⚠️" if match_status == "NO PAGE" else "❌")
        print(f"   ⏱️ Tempo: {tempo}s | Stato: {stato} | Match: {icon} {match_status} (Estratto: {files_str} p.{pags_str})\n")

    model_suffix = actual_model.replace('/', '_').replace(':', '_')
    out_dir = os.path.join(TESTS_DIR, "results_llm", "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"llm_results_{env}_{lang}_{chunk_size}_{rag_type}_{metodo}_{model_suffix}.csv")
    
    df_out = pd.DataFrame(risultati)
    df_out.to_csv(out_file, index=False, encoding='utf-8')
    print(f"✅ Test completato. Risultati salvati in:\n{out_file}")
    
    if total_valid > 0:
        hit_rate = (hit_count / total_valid) * 100
        no_page_rate = (no_page_count / total_valid) * 100
        miss_rate = (miss_count / total_valid) * 100
        avg_time = df_out["tempo_sec"].mean()
        
        # Genera tabelle breakdown in MD
        cat_rows = []
        for cat, s in sorted(cat_stats.items()):
            hr = s["hits"] / s["total"] * 100 if s["total"] else 0
            cat_rows.append(f"| {cat} | {s['hits']}/{s['total']} ({hr:.1f}%) |")
        cat_table = "\n".join(cat_rows)

        diff_rows = []
        ordine = ["bassa", "media", "alta", "low", "medium", "high"]
        chiavi = sorted(diff_stats.keys(), key=lambda x: ordine.index(x) if x in ordine else 99)
        for diff in chiavi:
            s = diff_stats[diff]
            hr = s["hits"] / s["total"] * 100 if s["total"] else 0
            diff_rows.append(f"| {diff} | {s['hits']}/{s['total']} ({hr:.1f}%) |")
        diff_table = "\n".join(diff_rows)
        
        # Salva il report in quick_eval
        quick_eval_dir = os.path.join(TESTS_DIR, "results_llm", "quick_eval")
        os.makedirs(quick_eval_dir, exist_ok=True)
        quick_eval_file = os.path.join(quick_eval_dir, f"llm_results_summary_{env}_{lang}_{chunk_size}_{rag_type}_{metodo}_{model_suffix}.md")
        
        summary_md = f"""# Report Risultati LLM Summary ({env}, {lang}, chunk: {chunk_size}, metodo: {metodo})

| Metric | Value |
| :--- | :--- |
| **RAG Algorithm** | {rag_type.capitalize()} |
| **LLM Model** | {actual_model} |
| **Domande valutate** | {total_valid} |
| **Risposte corrette (SI)** | {hit_count} ({hit_rate:.1f}%) |
| **Pagine mancanti (NO PAGE)** | {no_page_count} ({no_page_rate:.1f}%) |
| **Pagine errate (MISS)** | {miss_count} ({miss_rate:.1f}%) |
| **Hit Rate (LLM Accuracy)** | {hit_rate:.1f}% |
| **Tempo medio risposta** | {avg_time:.2f} s |

## Breakdown per Categoria
| Categoria | Accuratezza (SI/Totale) |
| :--- | :--- |
{cat_table}

## Breakdown per Difficoltà
| Difficoltà | Accuratezza (SI/Totale) |
| :--- | :--- |
{diff_table}
"""
        with open(quick_eval_file, "w", encoding="utf-8") as f:
            f.write(summary_md)

        print(f"⚡ Report di sintesi salvato in:\n{quick_eval_file}")
        
        print(f"\n📊 --- REPORT RISULTATI ---")
        print(f"   RAG Algorithm: {rag_type.capitalize()}")
        print(f"   LLM Model: {actual_model}")
        print(f"   Domande valutate: {total_valid}")
        print(f"   Risposte corrette (SI): {hit_count} ({hit_rate:.1f}%)")
        print(f"   Pagine mancanti (NO PAGE): {no_page_count} ({no_page_rate:.1f}%)")
        print(f"   Pagine errate (MISS): {miss_count} ({miss_rate:.1f}%)")
        print(f"   Hit Rate (LLM Accuracy): {hit_rate:.1f}%")
        print(f"   Tempo medio risposta: {avg_time:.2f} s/query")
        print(f"---------------------------")
        
        print("\nBreakdown per CATEGORIA:")
        for cat, s in sorted(cat_stats.items()):
            hr = s["hits"] / s["total"] * 100 if s["total"] else 0
            print(f"  {cat:<30}  {s['hits']:>3}/{s['total']:<3}  ({hr:.1f}%)")
            
        print("\nBreakdown per DIFFICOLTÀ:")
        for diff in chiavi:
            s = diff_stats[diff]
            hr = s["hits"] / s["total"] * 100 if s["total"] else 0
            print(f"  {diff:<10}  {s['hits']:>3}/{s['total']:<3}  ({hr:.1f}%)")
        print(f"{'═' * 55}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa la generazione dell'LLM (locale o cloud) sulle domande di test")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale", help="Ambiente LLM")
    parser.add_argument("--lang", type=str, choices=["it", "en"], default="it", help="Lingua del dataset di test")
    parser.add_argument("--chunk_size", type=int, default=700, help="Dimensione dei chunk")
    parser.add_argument("--limit", type=int, default=None, help="Limita il numero di domande da testare (es. 5 per test veloci)")
    parser.add_argument("--tolleranza", type=int, default=1, help="Tolleranza di pagina ±N")
    parser.add_argument("--model", type=str, default=None,
                        help="Modello LLM da usare (es. openai/gpt-4o-mini, google/gemini-2.5-flash, google/gemini-3.5-flash, meta-llama/llama-3.3-70b-instruct, qwen/qwen-2.5-72b-instruct, deepseek/deepseek-chat)")
    parser.add_argument("--rag_type", type=str, choices=["puro", "ibrido", "rerank", "graph"], default="ibrido",
                        help="Algoritmo RAG da usare: 'puro' (solo Vector Search), 'ibrido' (BM25 + Vector Search), 'rerank' (BM25 + Vector + Re-ranking) o 'graph' (GraphRAG leggero)")
    parser.add_argument(
        "--metodo", "-m", 
        type=str, 
        choices=["euristico", "pdf4llm", "docling", "llamaparse", "qwen"],
        default="pdf4llm",
        help="Metodo di estrazione dei PDF da testare."
    )
    args = parser.parse_args()

    valuta_llm(args.env, args.lang, args.chunk_size, args.limit, args.tolleranza, args.model, args.rag_type, args.metodo)
