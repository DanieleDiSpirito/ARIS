import os
import re
import json
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))


def estrai_json(testo: str) -> dict:
    """Estrae l'oggetto JSON dal testo della risposta del modello."""
    match = re.search(r'\{.*\}', testo, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def genera_domande(env: str, lang: str, metodo: str, target_count: int = 100, model_name: str = None):
    # 1. Carica tutti i file JSON del metodo selezionato
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed", metodo)
    if not os.path.exists(processed_dir):
        print(f"❌ Cartella processed non trovata in: {processed_dir}")
        return

    json_files = [f for f in os.listdir(processed_dir) if f.endswith(".json")]
    if not json_files:
        print(f"❌ Nessun file JSON trovato in {processed_dir}")
        return

    print(f"📂 Trovati {len(json_files)} file JSON di testo pre-elaborato.")

    # Raccogliamo tutti i chunk disponibili
    tutti_chunk = []
    for file_name in json_files:
        path = os.path.join(processed_dir, file_name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                # Filtra i chunk che non hanno testo utile o sono troppo corti
                for chunk in chunks:
                    if chunk.get("text") and len(str(chunk["text"]).strip()) > 100:
                        tutti_chunk.append(chunk)
        except Exception as e:
            print(f"   ⚠️ Errore durante il caricamento di {file_name}: {e}")

    total_chunks = len(tutti_chunk)
    print(f"📚 Totale chunk utili caricati: {total_chunks}")
    if total_chunks == 0:
        print("❌ Nessun chunk disponibile per la generazione.")
        return

    # Seleziona i chunk in modo uniforme per coprire l'intera documentazione
    passo = max(1, total_chunks // target_count)
    selezionati = []
    for i in range(target_count):
        idx = (i * passo) % total_chunks
        selezionati.append(tutti_chunk[idx])

    print(f"🎯 Selezionati {len(selezionati)} chunk per generare {target_count} domande.")

    # 2. Configura l'LLM
    if env == "locale":
        print("🤖 Generatore LLM: Locale (localhost:1234)")
        local_model = os.getenv("LOCAL_LLM_MODEL", None)
        base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
        llm = ChatOpenAI(
            model=local_model,
            base_url=base_url,
            api_key="lm-studio",
            temperature=0.3
        )
    elif env == "cloud":
        print("☁️ Generatore LLM: Cloud (OpenRouter)")
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("❌ ERRORE: Variabile OPENAI_API_KEY non trovata nel file .env")
        selected_model = model_name if model_name else "openai/gpt-4o-mini"
        print(f"🔎 Modello: {selected_model}")
        llm = ChatOpenAI(
            model=selected_model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3,
            max_tokens=1000
        )
    else:
        raise ValueError("Il parametro env deve essere 'locale' o 'cloud'")

    nuove_domande = []
    generati_con_successo = 0

    print(f"\n🚀 Avvio generazione delle domande...\n")

    for idx, chunk in enumerate(selezionati):
        file_basename = os.path.basename(chunk.get("file_name", "manuale.pdf"))
        page = chunk.get("page", "N/A")
        section = chunk.get("section", "N/A")
        title = chunk.get("title", "N/A")
        text = chunk.get("text", "")

        print(f"[{idx+1}/{len(selezionati)}] Generazione da: {file_basename} (Pag: {page}, Sez: {section})...")

        # Prompt per la generazione delle domande
        prompt = f"""Dato il seguente testo estratto da un manuale tecnico del controller Fanuc R-30iB (File: {file_basename}, Pagina: {page}, Sezione: {section}, Titolo: {title}):
---
{text}
---

Genera una domanda di test realistica e la relativa risposta attesa dettagliata per valutare un sistema RAG di assistenza tecnica.
La domanda deve basarsi ESCLUSIVAMENTE sulle informazioni contenute nel testo fornito. Non fare riferimento a informazioni esterne o non descritte nel testo.
La risposta deve essere corretta, esatta e interamente deducibile dal testo fornito.

Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. Non aggiungere spiegazioni prima o dopo il JSON. Il JSON deve avere questa struttura esatta:
{{
  "question": "Una domanda tecnica chiara e specifica in italiano (es. 'Qual è la corrente di uscita massima per ogni punto DO?')",
  "expected_answer": "La risposta esatta e completa basata solo sul testo (es. 'La corrente di uscita massima è 0.2A.')",
  "category": "Una tra: 'Consultazione tecnica', 'Codici errore', 'Procedure', 'Troubleshooting'",
  "difficulty": "Una tra: 'low', 'medium', 'hard'"
}}
"""

        success = False
        retry_count = 3
        while not success and retry_count > 0:
            try:
                t0 = time.time()
                response = llm.invoke(prompt)
                raw_content = response.content.strip()
                judgement = estrai_json(raw_content)

                if judgement and all(k in judgement for k in ["question", "expected_answer", "category", "difficulty"]):
                    qid = f"Q{generati_con_successo+1:03d}"
                    
                    diff_val = str(judgement["difficulty"]).strip().lower()
                    diff_map = {
                        "bassa": "low",
                        "low": "low",
                        "media": "medium",
                        "medium": "medium",
                        "alta": "hard",
                        "high": "hard",
                        "hard": "hard"
                    }
                    difficulty = diff_map.get(diff_val, "medium")

                    nuove_domande.append({
                        "id": qid,
                        "question": judgement["question"],
                        "category": judgement["category"],
                        "expected_answer": judgement["expected_answer"],
                        "expected_file": file_basename,
                        "expected_page": page,
                        "difficulty": difficulty
                    })
                    generati_con_successo += 1
                    success = True
                    tempo = round(time.time() - t0, 2)
                    print(f"   ✅ Generata: {judgement['question'][:60]}... ({tempo}s)")
                else:
                    print(f"   ⚠️ Risposta JSON non conforme. Tentativi rimasti: {retry_count-1}")
                    retry_count -= 1
            except Exception as e:
                print(f"   ⚠️ Errore durante la chiamata LLM: {e}. Tentativi rimasti: {retry_count-1}")
                retry_count -= 1
                time.sleep(1)

    # Scrive il nuovo file CSV
    out_file = os.path.join(os.path.dirname(BASE_DIR), f"test_questions_{lang}.csv")
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        f.write("id,question,category,expected_answer,expected_file,expected_page,difficulty\n")
        for q in nuove_domande:
            qid = str(q["id"])
            
            def clean_and_quote(val):
                val_str = str(val) if val is not None else ""
                val_escaped = val_str.replace('"', '""')
                return f'"{val_escaped}"'
            
            question = clean_and_quote(q["question"])
            category = clean_and_quote(q["category"])
            expected_answer = clean_and_quote(q["expected_answer"])
            expected_file = clean_and_quote(q["expected_file"])
            expected_page = clean_and_quote(q["expected_page"])
            difficulty = clean_and_quote(q["difficulty"])
            
            f.write(f"{qid},{question},{category},{expected_answer},{expected_file},{expected_page},{difficulty}\n")
            
    print(f"\n📊 Generazione completata con successo!")
    print(f"💾 Salvate {generati_con_successo} domande in:\n{out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera domande di test a partire dai testi dei manuali tecnici")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="cloud", help="Ambiente LLM")
    parser.add_argument("--lang", type=str, choices=["it"], default="it", help="Lingua del test set (solo 'it' supportato per la generazione diretta)")
    parser.add_argument("--count", type=int, default=100, help="Numero totale di domande da generare")
    parser.add_argument("--model", type=str, default=None, help="Modello OpenRouter da usare (es. openai/gpt-4o-mini)")
    parser.add_argument(
        "--metodo", "-m", 
        type=str, 
        choices=["euristico", "docling", "llamaparse", "qwen"],
        default="docling",
        help="Metodo di estrazione da cui prelevare i testi (euristico, docling, llamaparse, qwen)."
    )
    args = parser.parse_args()

    genera_domande(args.env, args.lang, args.metodo, args.count, args.model)
