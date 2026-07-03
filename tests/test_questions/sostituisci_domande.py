import os
import re
import json
import time
import argparse
import random
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from traduci_domande import traduci_testo

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)

def carica_testo_pagina(expected_file: str, expected_page: str, metodo: str) -> str:
    """Carica il testo della pagina specificata dal file JSON preelaborato."""
    expected_file = str(expected_file).strip()
    expected_page = str(expected_page).strip()

    json_filename = expected_file.replace(".pdf", ".json")
    json_path = os.path.join(PROJECT_ROOT, "data", "processed", metodo, json_filename)

    if not os.path.exists(json_path):
        return f"[ERRORE: File JSON non trovato per {expected_file} in {metodo}]"

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = []
        for item in data:
            if str(item.get("page", "")).strip() == expected_page:
                texts.append(item.get("text", ""))
        
        if texts:
            return "\n\n".join(texts)
    except Exception as e:
        return f"[ERRORE durante la lettura del file: {str(e)}]"

    return f"[ERRORE: Pagina {expected_page} non trovata nel file {expected_file}]"

def estrai_json_judge(testo: str) -> dict:
    """Estrae l'oggetto JSON del Judge usando espressioni regolari."""
    match = re.search(r'\{.*\}', testo, re.DOTALL)
    if not match:
        return None
        
    json_str = match.group().strip()
    
    try:
        # Prova standard
        return json.loads(json_str)
    except Exception:
        pass
        
    try:
        # Estrai "valido"
        valido_match = re.search(r'[\'"]valido[\'"]\s*:\s*(true|false|[\'"][^\'"]*[\'"])', json_str, re.IGNORECASE)
        valido = False
        if valido_match:
            val_val = valido_match.group(1).lower()
            valido = "true" in val_val or "1" in val_val
            
        # Estrai "categoria_errore"
        cat_match = re.search(r'[\'"]categoria_errore[\'"]\s*:\s*[\'"]([^\'"]*)[\'"]', json_str)
        categoria_errore = cat_match.group(1) if cat_match else "Nessuna"
        
        # Estrai "motivo_del_giudizio" catturando il testo tra virgolette strutturali (singole o doppie)
        motivo = "Nessun motivo fornito."
        motivo_match = re.search(r'[\'"]motivo_del_giudizio[\'"]\s*:\s*[\'"](.*)[\'"]\s*,\s*[\'"]categoria_errore[\'"]', json_str, re.DOTALL)
        if not motivo_match:
            motivo_match = re.search(r'[\'"]motivo_del_giudizio[\'"]\s*:\s*[\'"](.*)[\'"]\s*\}', json_str, re.DOTALL)
            
        if motivo_match:
            motivo_raw = motivo_match.group(1).strip()
            if motivo_raw.endswith('",') or motivo_raw.endswith("',"):
                motivo_raw = motivo_raw[:-2]
            elif motivo_raw.endswith('"') or motivo_raw.endswith("'"):
                motivo_raw = motivo_raw[:-1]
            motivo = motivo_raw
            
        return {
            "valido": valido,
            "motivo_del_giudizio": motivo,
            "categoria_errore": categoria_errore
        }
    except Exception:
        pass
        
    return None


def estrai_json_generatore(testo: str) -> dict:
    """Estrae l'oggetto JSON del generatore usando espressioni regolari."""
    match = re.search(r'\{.*\}', testo, re.DOTALL)
    if not match:
        return None
        
    json_str = match.group().strip()
    
    try:
        # Prova standard
        return json.loads(json_str)
    except Exception:
        pass
        
    try:
        # Categoria
        cat_match = re.search(r'[\'"]category[\'"]\s*:\s*[\'"]([^\'"]*)[\'"]', json_str)
        category = cat_match.group(1) if cat_match else "Consultazione tecnica"
        
        # Difficoltà
        diff_match = re.search(r'[\'"]difficulty[\'"]\s*:\s*[\'"]([^\'"]*)[\'"]', json_str)
        difficulty = diff_match.group(1) if diff_match else "medium"
        
        # question
        question = ""
        q_match = re.search(r'[\'"]question[\'"]\s*:\s*[\'"](.*)[\'"]\s*,\s*[\'"]expected_answer[\'"]', json_str, re.DOTALL)
        if not q_match:
            q_match = re.search(r'[\'"]question[\'"]\s*:\s*[\'"](.*)[\'"]\s*(?:,|\})', json_str, re.DOTALL)
        if q_match:
            q_raw = q_match.group(1).strip()
            if q_raw.endswith('",') or q_raw.endswith("',"):
                q_raw = q_raw[:-2]
            elif q_raw.endswith('"') or q_raw.endswith("'"):
                q_raw = q_raw[:-1]
            question = q_raw
            
        # expected_answer
        expected_answer = ""
        a_match = re.search(r'[\'"]expected_answer[\'"]\s*:\s*[\'"](.*)[\'"]\s*,\s*[\'"](?:category|difficulty)[\'"]', json_str, re.DOTALL)
        if not a_match:
            a_match = re.search(r'[\'"]expected_answer[\'"]\s*:\s*[\'"](.*)[\'"]\s*\}', json_str, re.DOTALL)
        if a_match:
            a_raw = a_match.group(1).strip()
            if a_raw.endswith('",') or a_raw.endswith("',"):
                a_raw = a_raw[:-2]
            elif a_raw.endswith('"') or a_raw.endswith("'"):
                a_raw = a_raw[:-1]
            expected_answer = a_raw
            
        if question and expected_answer:
            return {
                "question": question,
                "expected_answer": expected_answer,
                "category": category,
                "difficulty": difficulty
            }
    except Exception:
        pass
        
    return None

def salva_csv_personalizzato(df, filepath):
    """Salva il dataframe nel formato CSV specificato (id senza virgolette, altre colonne con virgolette)."""
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        f.write("id,question,category,expected_answer,expected_file,expected_page,difficulty\n")
        for _, row in df.iterrows():
            qid = str(row["id"])
            
            def clean_and_quote(val):
                val_str = str(val) if pd.notna(val) else ""
                val_escaped = val_str.replace('"', '""')
                return f'"{val_escaped}"'
            
            question = clean_and_quote(row["question"])
            category = clean_and_quote(row["category"])
            expected_answer = clean_and_quote(row["expected_answer"])
            expected_file = clean_and_quote(row["expected_file"])
            expected_page = clean_and_quote(row["expected_page"])
            difficulty = clean_and_quote(row["difficulty"])
            
            f.write(f"{qid},{question},{category},{expected_answer},{expected_file},{expected_page},{difficulty}\n")

def main():
    parser = argparse.ArgumentParser(description="Valida e sostituisce le domande non valide in-place su DataFrame")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="cloud", help="Ambiente LLM")
    parser.add_argument("--model-gen", type=str, default="openai/gpt-4o-mini", help="Modello di generazione")
    parser.add_argument("--model-judge", type=str, default="meta-llama/llama-3.3-70b-instruct", help="Modello di validazione")
    parser.add_argument(
        "--metodo", "-m", 
        type=str, 
        choices=["euristico", "pdf4llm", "docling", "llamaparse", "qwen"],
        default="pdf4llm",
        help="Metodo di estrazione per il caricamento dei testi."
    )
    args = parser.parse_args()

    it_file = os.path.join(TESTS_DIR, "test_questions_it.csv")
    en_file = os.path.join(TESTS_DIR, "test_questions_en.csv")

    if not os.path.exists(it_file):
        print(f"❌ File delle domande in italiano non trovato: {it_file}")
        return

    if not os.path.exists(en_file):
        print(f"❌ File delle domande in inglese non trovato: {en_file}")
        return

    print("📋 Caricamento dei file originali...")
    df_questions = pd.read_csv(it_file)
    df_en = pd.read_csv(en_file)

    if len(df_questions) == 0:
        print("❌ Il file italiano è vuoto.")
        return

    # Carica tutti i chunk disponibili per la generazione
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed", args.metodo)
    if not os.path.exists(processed_dir):
        print(f"❌ Cartella processed non trovata in: {processed_dir}")
        return

    json_files = [f for f in os.listdir(processed_dir) if f.endswith(".json")]
    tutti_chunk = []
    for file_name in json_files:
        path = os.path.join(processed_dir, file_name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                for chunk in chunks:
                    if chunk.get("text") and len(str(chunk["text"]).strip()) > 100:
                        tutti_chunk.append(chunk)
        except Exception as e:
            print(f"   ⚠️ Errore caricamento {file_name}: {e}")

    # Configurazione LLM
    if args.env == "locale":
        print("🤖 LLM: Locale (localhost:1234)")
        local_model = os.getenv("LOCAL_LLM_MODEL", None)
        base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
        llm_gen = ChatOpenAI(model=local_model, base_url=base_url, api_key="lm-studio", temperature=0.3, max_tokens=500)
        llm_judge = ChatOpenAI(model=local_model, base_url=base_url, api_key="lm-studio", temperature=0.0, max_tokens=400)
        llm_trans = llm_gen
    elif args.env == "cloud":
        print(f"☁️ LLM: Cloud (OpenRouter) -> Gen: {args.model_gen} | Judge: {args.model_judge}")
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("❌ ERRORE: Variabile OPENAI_API_KEY non trovata nel file .env")
        llm_gen = ChatOpenAI(
            model=args.model_gen,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3,
            max_tokens=500,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        llm_judge = ChatOpenAI(
            model=args.model_judge,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
            max_tokens=400,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        llm_trans = ChatOpenAI(
            model="openai/gpt-4o-mini",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
            max_tokens=300
        )

    modificati = 0
    validi_count = 0

    print(f"\n🚀 Avvio verifica e sostituzione in-place su {len(df_questions)} domande...\n")
    
    for idx, row in df_questions.iterrows():
        qid = row["id"]
        question = row["question"]
        expected_answer = row["expected_answer"]
        expected_file = row["expected_file"]
        expected_page = str(row["expected_page"])
        target_category = row["category"]
        target_difficulty = row["difficulty"]
        
        print(f"[{idx+1}/{len(df_questions)}] Verifica {qid}...")
        
        # 1. Carica il testo della pagina originaria
        page_text = carica_testo_pagina(expected_file, expected_page, args.metodo)
        
        valido = False
        motivo = "Impossibile caricare il testo della pagina"
        
        if "[ERRORE:" not in page_text:
            # Esegui la validazione della domanda corrente
            prompt_judge = f"""Sei un ispettore di qualità per dataset di Machine Learning/QA.
Ti viene fornito un testo estratto da un manuale tecnico di un robot industriale Fanuc, una domanda formulata a partire da quel testo e la risposta attesa.
Devi verificare se la domanda e la risposta attesa sono corrette, coerenti e completamente deducibili dal testo fornito.

Testo del manuale:
---
{page_text}
---

Domanda: {question}
Risposta Attesa: {expected_answer}

Verifiche da fare:
1. La domanda è sensata, chiara e formulata correttamente in italiano?
2. La risposta alla domanda è effettivamente e interamente supportata dal testo fornito? Non ci sono dettagli inventati (hallucination) o informazioni esterne non verificabili da questo testo?
3. Il riferimento alla pagina del documento è appropriato (il testo contiene effettivamente l'argomento della domanda)?

Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. Non aggiungere spiegazioni prima o dopo il JSON.
IMPORTANTE: Non inserire MAI virgolette doppie (") all'interno dei valori stringa (es. dentro 'motivo_del_giudizio'). Se devi fare citazioni o racchiudere parole, usa ESCLUSIVAMENTE virgolette singole (').

Il JSON deve avere questa struttura esatta:
{{
  "valido": true o false (scrivi in minuscolo true o false, senza virgolette),
  "motivo_del_giudizio": "Spiegazione sintetica del perché è valido o meno.",
  "categoria_errore": "Nessuna" o "Allucinazione nella risposta" o "Domanda non supportata dal testo" o "Errore grammaticale o di formulazione" o "Testo non pertinente"
}}
"""
            try:
                judge_response = llm_judge.invoke(prompt_judge)
                raw_content = judge_response.content.strip()
                judge_decision = estrai_json_judge(raw_content)
                if judge_decision:
                    valido = bool(judge_decision.get("valido", False))
                    motivo = judge_decision.get("motivo_del_giudizio", "Nessun motivo fornito.")
                else:
                    valido = False
                    content_snippet = re.sub(r'\\+', r'\\', raw_content).strip()
                    content_snippet = content_snippet[:150] + ("..." if len(content_snippet) > 150 else "")
                    motivo = f"Risposta del Judge non formattabile come JSON: {content_snippet}"
            except Exception as e:
                print(f"   ⚠️ Errore chiamata Judge per verifica: {e}")
                motivo = f"Errore chiamata Judge: {e}"
        else:
            motivo = page_text
            
        if valido:
            print(f"   ⚖️ Giudizio: ✅ VALIDO")
            validi_count += 1
            continue
            
        # Domanda non valida -> avvia la sostituzione
        print(f"   ⚖️ Giudizio: ❌ NON VALIDO. Motivo: {motivo}")
        print(f"   🔄 Avvio rigenerazione per {qid} per lo stesso profilo e argomento...")
        
        # Trova il chunk corrispondente allo stesso argomento
        # Raccogliamo i chunk candidati della pagina specifica e dell'intero file
        chunk_pagina = [
            c for c in tutti_chunk 
            if os.path.basename(c.get("file_name", "")) == expected_file 
            and str(c.get("page", "")) == expected_page
        ]
        chunk_file_raw = [
            c for c in tutti_chunk 
            if os.path.basename(c.get("file_name", "")) == expected_file
        ]
        
        # Filtro intelligente basato su parole chiave della categoria per evitare di selezionare chunk non pertinenti
        keywords_categoria = {
            "Codici errore": ["alarm", "allarme", "codice", "srvo-", "code", "errore"],
            "Troubleshooting": ["risoluzione", "problema", "troubleshooting", "ripristino", "allarme", "errore"],
            "Procedure": ["procedura", "passo", "step", "procedere", "collegamento", "sostituzione", "cavo", "connessione"],
            "Consultazione tecnica": []
        }
        target_kws = keywords_categoria.get(target_category, [])
        
        chunk_file = []
        if target_kws:
            chunk_file = [
                c for c in chunk_file_raw 
                if any(kw in str(c.get("text", "")).lower() for kw in target_kws)
            ]
        
        # Fallback se non ci sono chunk filtrati o se la categoria è generica
        if not chunk_file:
            chunk_file = chunk_file_raw
            
        success_gen = False
        attempts = 0
        max_attempts = 10
        
        while not success_gen and attempts < max_attempts:
            attempts += 1
            
            # Nei primi 3 tentativi proviamo sulla pagina esatta, poi allarghiamo a tutto il file per evitare stalli
            if attempts <= 3 and chunk_pagina:
                chunk = random.choice(chunk_pagina)
            elif chunk_file:
                if attempts == 4:
                    print(f"      ℹ️ Allargo la ricerca dei chunk a tutto il file {expected_file}...")
                chunk = random.choice(chunk_file)
            else:
                if attempts == 4:
                    print(f"      ℹ️ Allargo la ricerca a tutti i chunk disponibili...")
                chunk = random.choice(tutti_chunk)
                
            file_basename = os.path.basename(chunk.get("file_name", "manuale.pdf"))
            page = str(chunk.get("page", "N/A"))
            section = chunk.get("section", "N/A")
            title = chunk.get("title", "N/A")
            text = chunk.get("text", "")
            
            print(f"      [Tentativo {attempts}] Generazione da: {file_basename} (Pag: {page}, Sez: {section})...")
            
            prompt_gen = f"""Dato il seguente testo estratto da un manuale tecnico del controller Fanuc R-30iB (File: {file_basename}, Pagina: {page}, Sezione: {section}, Titolo: {title}):
---
{text}
---

Genera una domanda di test tecnica realistica e la relativa risposta attesa dettagliata per valutare un sistema RAG di assistenza tecnica.
La domanda deve basarsi ESCLUSIVAMENTE sulle informazioni contenute nel testo fornito. Non fare riferimento a informazioni esterne o non descritte nel testo.
La risposta deve essere corretta, esatta e interamente deducibile dal testo fornito.

ATTENZIONE CRITICA: Usa ESCLUSIVAMENTE i termini, numeri di pin e codici di errore (ad es. codici allarme come SRVO-074) presenti letteralmente nel testo. Non inventare, modificare o associare codici non citati (ad es. non generare domande su SRVO-096 se nel testo c'è scritto SRVO-074).
Se il testo non contiene sufficienti informazioni o dettagli complessi per formulare una domanda '{target_difficulty}' o inerente a '{target_category}', semplifica la domanda rimanendo strettamente aderente e fedele al testo, piuttosto che inventare particolari inesistenti.

La domanda DEVE avere le seguenti caratteristiche prefissate:
- Categoria: '{target_category}'
- Difficoltà: '{target_difficulty}' (dove 'low' indica una domanda diretta su specifiche o dati chiaramente visibili nel testo, 'medium' richiede il confronto o l'estrazione di informazioni da più frasi, e 'hard' richiede la comprensione di una procedura complessa o la risoluzione di un problema con più passi descritti).

Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. Non aggiungere spiegazioni prima o dopo il JSON. Il JSON deve avere questa struttura esatta:
{{
  "question": "Una domanda tecnica chiara e specifica in italiano",
  "expected_answer": "La risposta esatta e completa basata solo sul testo",
  "category": "{target_category}",
  "difficulty": "{target_difficulty}"
}}
"""
            try:
                response = llm_gen.invoke(prompt_gen)
                judgement = estrai_json_generatore(response.content.strip())
                
                if not judgement or not all(k in judgement for k in ["question", "expected_answer"]):
                    continue
                    
                new_q = judgement["question"]
                new_a = judgement["expected_answer"]
                
                # Validazione immediata del nuovo chunk
                prompt_judge_new = f"""Sei un ispettore di qualità per dataset di Machine Learning/QA.
Ti viene fornito un testo estratto da un manuale tecnico di un robot industriale Fanuc, una domanda formulata a partire da quel testo e la risposta attesa.
Devi verificare se la domanda e la risposta attesa sono corrette, coerenti e completamente deducibili dal testo fornito.

Testo del manuale:
---
{text}
---

Domanda: {new_q}
Risposta Attesa: {new_a}

Verifiche da fare:
1. La domanda è sensata, chiara e formulata correttamente in italiano?
2. La risposta alla domanda è effettivamente e interamente supportata dal testo fornito? Non ci sono dettagli inventati (hallucination) o informazioni esterne non verificabili da questo testo?
3. Il riferimento alla pagina del documento è appropriato (il testo contiene effettivamente l'argomento della domanda)?

Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. Non aggiungere spiegazioni prima o dopo il JSON.
IMPORTANTE: Non inserire MAI virgolette doppie (") all'interno dei valori stringa (es. dentro 'motivo_del_giudizio'). Se devi fare citazioni o racchiudere parole, usa ESCLUSIVAMENTE virgolette singole (').

Il JSON deve avere questa struttura esatta:
{{
  "valido": true o false (scrivi in minuscolo true o false, senza virgolette),
  "motivo_del_giudizio": "Spiegazione sintetica del perché è valido o meno.",
  "categoria_errore": "Nessuna" o "Allucinazione nella risposta" o "Domanda non supportata dal testo" o "Errore grammaticale o di formulazione" o "Testo non pertinente"
}}
"""
                judge_response_new = llm_judge.invoke(prompt_judge_new)
                judge_decision_new = estrai_json_judge(judge_response_new.content.strip())
                
                if judge_decision_new and judge_decision_new.get("valido") == True:
                    print(f"      ✅ Nuova domanda approvata dal Judge! Motivo: {judge_decision_new.get('motivo_del_giudizio')}")
                    
                    # Aggiorna df_questions in-place
                    df_questions.loc[idx, "question"] = new_q
                    df_questions.loc[idx, "expected_answer"] = new_a
                    df_questions.loc[idx, "expected_file"] = file_basename
                    df_questions.loc[idx, "expected_page"] = page
                    
                    # Traduzione e aggiornamento df_en in-place
                    print(f"      🌍 Traduzione in inglese per {qid}...")
                    q_en = traduci_testo(llm_trans, new_q, "domanda")
                    a_en = traduci_testo(llm_trans, new_a, "risposta attesa")
                    
                    en_indices = df_en[df_en["id"] == qid].index
                    if len(en_indices) > 0:
                        en_idx = en_indices[0]
                        df_en.loc[en_idx, "question"] = q_en
                        df_en.loc[en_idx, "expected_answer"] = a_en
                        df_en.loc[en_idx, "expected_file"] = file_basename
                        df_en.loc[en_idx, "expected_page"] = page
                        df_en.loc[en_idx, "category"] = target_category
                        df_en.loc[en_idx, "difficulty"] = target_difficulty
                        
                    success_gen = True
                    modificati += 1
                else:
                    if judge_decision_new:
                        motivo_new = judge_decision_new.get("motivo_del_giudizio", "Non specificato")
                    else:
                        raw_content_new = judge_response_new.content.strip()
                        content_snippet_new = re.sub(r'\\+', r'\\', raw_content_new).strip()
                        content_snippet_new = content_snippet_new[:150] + ("..." if len(content_snippet_new) > 150 else "")
                        motivo_new = f"JSON non valido: {content_snippet_new}"
                    print(f"      ❌ Nuova domanda rifiutata dal Judge. Motivo: {motivo_new}")
            except Exception as e:
                print(f"      ⚠️ Errore durante generazione/verifica: {e}")
                time.sleep(1)

    print("\n--- REPORT DI FINE PROCESSO ---")
    print(f"  Domande analizzate : {len(df_questions)}")
    print(f"  Domande valide mantenute : {validi_count}")
    print(f"  Domande rigenerate/tradotte: {modificati}")
    print("--------------------------------\n")

    if modificati > 0:
        # Salva entrambi i file CSV aggiornati nel formato corretto
        salva_csv_personalizzato(df_questions, it_file)
        salva_csv_personalizzato(df_en, en_file)
        print(f"💾 Aggiornato con successo {it_file}!")
        print(f"💾 Aggiornato con successo {en_file} (tradotte solo le {modificati} domande modificate)!")
    else:
        print("ℹ️ Nessuna domanda ha subito modifiche.")

if __name__ == "__main__":
    main()
