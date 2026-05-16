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

def _estrai_fonte_llm(risposta: str):
    # Cerca un file .pdf e (opzionalmente) un indicatore di pagina entro i successivi 30 caratteri
    pattern = r'([a-zA-Z0-9_\-\.]+\.pdf)(?:[^0-9]{0,30}?(?:pagina|page|pag|p)\.?\s*[:]?\s*(\d+))?'
    matches = re.findall(pattern, risposta, re.IGNORECASE)
    
    if matches:
        ultimo_match = matches[-1]
        file_trovato = ultimo_match[0]
        pag_trovata = ultimo_match[1] if ultimo_match[1] else ""
        return file_trovato, pag_trovata
        
    return "", ""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src", "rag_pipeline"))

from rag_pipeline_hybrid import setup_rag_chain, get_db_path, get_embeddings, build_hybrid_retriever, COLLECTION_NAME
from langchain_chroma import Chroma

load_dotenv()

def valuta_llm(env: str, lang: str, chunk_size: int = 700, max_questions: int = None, tolleranza: int = 1):
    test_file = os.path.join(TESTS_DIR, f"test_questions_{lang}.csv")
    if not os.path.exists(test_file):
        print(f"❌ File di test non trovato: {test_file}")
        return

    df = pd.read_csv(test_file)
    if max_questions:
        df = df.head(max_questions)

    db_path = os.path.join(PROJECT_ROOT, get_db_path(env, chunk_size))
    if not os.path.exists(db_path):
        print(f"❌ Database non trovato in: {db_path}")
        return

    print(f"🗄️ Caricamento DB e Retriever ({env}, chunk: {chunk_size}, lingua: {lang})...")
    embedder = get_embeddings(env)
    lc_chroma = Chroma(
        persist_directory=db_path,
        collection_name=COLLECTION_NAME,
        embedding_function=embedder
    )
    retriever = build_hybrid_retriever(lc_chroma, k=3)
    rag_chain = setup_rag_chain(retriever, env=env)

    risultati = []
    hit_count = 0
    total_valid = 0
    
    print(f"\n🚀 Inizio test su {len(df)} domande (Tolleranza pagina: ±{tolleranza})...\n")
    for index, row in df.iterrows():
        domanda = row['question']
        print(f"[{index+1}/{len(df)}] Q: {domanda}")
        
        t0 = time.time()
        try:
            risposta = rag_chain.invoke({
                "question": domanda,
                "history": ""
            })
            stato = "OK"
        except Exception as e:
            risposta = f"ERRORE: {str(e)}"
            stato = "ERRORE"
            print(risposta)
            
        t1 = time.time()
        tempo = round(t1 - t0, 2)
        
        expected_file = str(row.get('expected_file', '')).strip()
        expected_page = str(row.get('expected_page', '')).strip()
        
        file_llm, pag_llm = _estrai_fonte_llm(risposta)
        match_file = (file_llm.lower() == expected_file.lower()) if expected_file and file_llm else False
        match_page = _pagine_match_tol(pag_llm, expected_page, tolleranza) if expected_page and pag_llm else False
        
        is_hit = False
        match_status = "NO"
        
        if expected_file != 'nan' and expected_file != '':
            total_valid += 1
            if not file_llm or not pag_llm:
                match_status = "NO PAGE"
            elif match_file and match_page:
                is_hit = True
                match_status = "SI"
                hit_count += 1
        
        risultati.append({
            "id": row.get('id', index),
            "domanda": domanda,
            "risposta_llm": risposta,
            "llm_file_trovato": file_llm,
            "llm_pag_trovata": pag_llm,
            "expected_file": expected_file,
            "expected_page": expected_page,
            "match": match_status,
            "tempo_sec": tempo,
            "stato": stato
        })
        
        icon = "✅" if match_status == "SI" else ("⚠️" if match_status == "NO PAGE" else "❌")
        print(f"   ⏱️ Tempo: {tempo}s | Stato: {stato} | Match: {icon} {match_status} (Estratto: {file_llm} p.{pag_llm})\n")

    out_dir = os.path.join(TESTS_DIR, "risultati_llm")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"llm_results_{env}_{lang}_{chunk_size}.csv")
    
    df_out = pd.DataFrame(risultati)
    df_out.to_csv(out_file, index=False, encoding='utf-8')
    print(f"✅ Test completato. Risultati salvati in:\n{out_file}")
    
    if total_valid > 0:
        hit_rate = (hit_count / total_valid) * 100
        print(f"\n📊 --- REPORT RISULTATI ---")
        print(f"   Domande valutate: {total_valid}")
        print(f"   Risposte corrette (File + Pagina ±{tolleranza}): {hit_count}")
        print(f"   Hit Rate (LLM Accuracy): {hit_rate:.1f}%")
        print(f"---------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa la generazione dell'LLM (locale o cloud) sulle domande di test")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale", help="Ambiente LLM")
    parser.add_argument("--lang", type=str, choices=["it", "en"], default="it", help="Lingua del dataset di test")
    parser.add_argument("--chunk_size", type=int, default=700, help="Dimensione dei chunk")
    parser.add_argument("--limit", type=int, default=None, help="Limita il numero di domande da testare (es. 5 per test veloci)")
    parser.add_argument("--tolleranza", type=int, default=1, help="Tolleranza di pagina ±N")
    args = parser.parse_args()

    valuta_llm(args.env, args.lang, args.chunk_size, args.limit, args.tolleranza)
