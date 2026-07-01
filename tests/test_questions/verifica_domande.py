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
TESTS_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)


def carica_testo_pagina(expected_file: str, expected_page: str, metodo: str = "pdf4llm") -> str:
    """Carica il testo della pagina specificata dal file JSON preelaborato."""
    expected_file = str(expected_file).strip()
    expected_page = str(expected_page).strip()

    # Mappa .pdf a .json
    json_filename = expected_file.replace(".pdf", ".json")
    json_path = os.path.join(PROJECT_ROOT, "data", "processed", metodo, json_filename)

    if not os.path.exists(json_path):
        return f"[ERRORE: File JSON non trovato per {expected_file}]"

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


def estrai_json(testo: str) -> dict:
    """Estrae l'oggetto JSON dal testo della risposta del modello usando espressioni regolari."""
    match = re.search(r'\{.*\}', testo, re.DOTALL)
    if not match:
        return None
        
    json_str = match.group().strip()
    
    # Estraiamo le chiavi e i valori con regex specifiche (estremamente robusto contro virgolette singole o doppie non scappate)
    # in caso di virgolette doppie o singole errate o non scappate
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
        # Cerca fino a ', "categoria_errore"' o similar
        motivo_match = re.search(r'[\'"]motivo_del_giudizio[\'"]\s*:\s*[\'"](.*)[\'"]\s*,\s*[\'"]categoria_errore[\'"]', json_str, re.DOTALL)
        if not motivo_match:
            # Magari categoria_errore è prima o motivo_del_giudizio è all'ultimo posto
            motivo_match = re.search(r'[\'"]motivo_del_giudizio[\'"]\s*:\s*[\'"](.*)[\'"]\s*\}', json_str, re.DOTALL)
            
        if motivo_match:
            motivo_raw = motivo_match.group(1).strip()
            # Pulizia di eventuali virgolette di chiusura residue se il regex ha preso troppo
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


def valida_domande(env: str, lang: str, limit: int = None, model_name: str = None, metodo: str = "pdf4llm"):
    test_file = os.path.join(TESTS_DIR, f"test_questions_{lang}.csv")
    if not os.path.exists(test_file):
        print(f"❌ File di test non trovato: {test_file}")
        return

    df = pd.read_csv(test_file)
    if limit:
        df = df.sample(n=min(limit, len(df))).reset_index(drop=True)

    print(f"📋 Caricamento di {len(df)} domande per la validazione...")

    # Configurazione LLM Judge
    if env == "locale":
        print("🤖 LLM Judge: Locale (localhost:1234)")
        local_model = os.getenv("LOCAL_LLM_MODEL", None)
        base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
        llm = ChatOpenAI(
            model=local_model,
            base_url=base_url,
            api_key="lm-studio",
            temperature=0.0
        )
    elif env == "cloud":
        print("☁️ LLM Judge: Cloud (OpenRouter)")
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("❌ ERRORE: Variabile OPENAI_API_KEY non trovata nel file .env")
        # Di default per la validazione usiamo un modello potente ma economico, es. meta-llama/llama-3.3-70b-instruct
        selected_model = model_name if model_name else "meta-llama/llama-3.3-70b-instruct"
        print(f"🔎 Modello: {selected_model}")
        llm = ChatOpenAI(
            model=selected_model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
            max_tokens=1000,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    else:
        raise ValueError("Il parametro env deve essere 'locale' o 'cloud'")

    risultati_validazione = []
    validi = 0
    non_validi = 0

    print(f"\n🚀 Inizio validazione tramite LLM Judge...\n")
    for index, row in df.iterrows():
        qid = row.get('id', f"Q{index+1}")
        question = row['question']
        expected_answer = row['expected_answer']
        expected_file = row['expected_file']
        expected_page = row['expected_page']

        print(f"[{index+1}/{len(df)}] Verifica {qid}...")

        # Carica il contesto originale della pagina
        page_text = carica_testo_pagina(expected_file, expected_page, metodo)

        if "[ERRORE:" in page_text:
            print(f"   ⚠️ {page_text}")
            risultati_validazione.append({
                "id": qid,
                "question": question,
                "expected_file": expected_file,
                "expected_page": expected_page,
                "valido": False,
                "motivo": page_text,
                "categoria_errore": "Pagina/file errato"
            })
            non_validi += 1
            continue

        # Costruisci il prompt per il Judge
        prompt = f"""Sei un ispettore di qualità per dataset di Machine Learning/QA.
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

        t0 = time.time()
        try:
            response = llm.invoke(prompt)
            raw_content = response.content.strip()
            judgement = estrai_json(raw_content)
            
            if judgement is None:
                # Fallback se non restituisce JSON
                valido = False
                # Pulisce sequenze ripetute di backslash e tronca la stringa per evitare righe giganti nell'MD/CSV
                content_snippet = re.sub(r'\\+', r'\\', raw_content).strip()
                content_snippet = content_snippet[:150] + ("..." if len(content_snippet) > 150 else "")
                motivo = f"Errore: Risposta del Judge non formattabile come JSON: {content_snippet}"
                cat_errore = "Errore di parsing del Judge"
            else:
                valido = bool(judgement.get("valido", False))
                motivo = judgement.get("motivo_del_giudizio", "Nessun motivo fornito.")
                cat_errore = judgement.get("categoria_errore", "Nessuna")
                
        except Exception as e:
            valido = False
            motivo = f"Errore durante la chiamata LLM: {str(e)}"
            cat_errore = "Errore del Judge"

        tempo = round(time.time() - t0, 2)
        
        status_icon = "✅ VALIDO" if valido else "❌ NON VALIDO"
        print(f"   ⚖️ Giudizio: {status_icon} | Tempo: {tempo}s")
        print(f"   📝 Motivo: {motivo}\n")

        if valido:
            validi += 1
        else:
            non_validi += 1

        risultati_validazione.append({
            "id": qid,
            "question": question,
            "expected_file": expected_file,
            "expected_page": expected_page,
            "valido": valido,
            "motivo": motivo,
            "categoria_errore": cat_errore
        })
    
    df_out = pd.DataFrame(risultati_validazione)
    out_file = os.path.join(PROJECT_ROOT, "data", "metrics", f"questions_quality_{lang}.csv")
    df_out.to_csv(out_file, index=False, encoding='utf-8')
    print(f"📊 Validazione completata. Risultati salvati in:\n{out_file}")
    
    # Calcolo percentuali
    totale = len(df)
    pct_valide = (validi / totale) * 100 if totale > 0 else 0
    pct_non_valide = (non_validi / totale) * 100 if totale > 0 else 0

    # Creazione della tabella di sintesi
    summary_data = [
        {"Stato": "Risposte Corrette (Valide)", "Numero": validi, "Percentuale": f"{pct_valide:.1f}%"},
        {"Stato": "Risposte Sbagliate (Non Valide)", "Numero": non_validi, "Percentuale": f"{pct_non_valide:.1f}%"},
        {"Stato": "Totale", "Numero": totale, "Percentuale": "100.0%"}
    ]
    df_summary = pd.DataFrame(summary_data)

    print("\n📊 TABELLA DI SINTESI QUALITÀ:")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print("=" * 60)

    # Filtraggio delle domande non valide per la stampa e salvataggio
    df_invalid = df_out[df_out["valido"] == False]

    print("\n❌ DETTAGLIO RISPOSTE SBAGLIATE (CON CATEGORIA ERRORE):")
    if not df_invalid.empty:
        print("=" * 100)
        for _, row in df_invalid.iterrows():
            print(f"ID: {row['id']} | Categoria Errore: {row['categoria_errore']}")
            print(f"Domanda: {row['question']}")
            print(f"Motivo: {row['motivo']}")
            print("-" * 100)
    else:
        print("Nessuna risposta sbagliata riscontrata!")
        print("=" * 60)

    # Scrittura del file Markdown
    out_file_md = os.path.join(PROJECT_ROOT, "data", "metrics", f"questions_quality_{lang}.md")
    
    md_summary_table = df_summary.to_markdown(index=False)
    
    if not df_invalid.empty:
        md_details_table = df_invalid[["id", "question", "categoria_errore", "motivo"]].to_markdown(index=False)
    else:
        md_details_table = "_Nessuna risposta sbagliata riscontrata._"

    md_content = f"""# Report Qualità Domande (Lingua: {lang.upper()})

## Sintesi della Validazione

{md_summary_table}

## Dettaglio Errori delle Risposte Sbagliate

{md_details_table}
"""

    with open(out_file_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"📝 Report di qualità salvato in formato Markdown in:\n{out_file_md}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica la qualità e correttezza delle domande di test generate")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="cloud", help="Ambiente LLM Judge")
    parser.add_argument("--lang", type=str, choices=["it", "en"], default="it", help="Lingua del test")
    parser.add_argument("--limit", type=int, default=None, help="Limita il numero di domande da testare")
    parser.add_argument("--model", type=str, default=None, help="Modello OpenRouter da usare (es. openai/gpt-4o-mini)")
    parser.add_argument(
        "--metodo", "-m", 
        type=str, 
        choices=["euristico", "pdf4llm", "docling", "llamaparse", "qwen"],
        default="pdf4llm",
        help="Metodo di estrazione per caricare i testi delle pagine."
    )
    args = parser.parse_args()

    valida_domande(args.env, args.lang, args.limit, args.model, args.metodo)
