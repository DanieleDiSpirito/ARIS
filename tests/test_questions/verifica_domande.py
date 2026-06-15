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


def carica_testo_pagina(expected_file: str, expected_page: str) -> str:
    """Carica il testo della pagina specificata dal file JSON preelaborato."""
    expected_file = str(expected_file).strip()
    expected_page = str(expected_page).strip()

    # Mappa .pdf a .json
    json_filename = expected_file.replace(".pdf", ".json")
    json_path = os.path.join(PROJECT_ROOT, "data", "processed", "euristico", json_filename)

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
    """Estrae l'oggetto JSON dal testo della risposta del modello."""
    # Cerca il blocco di codice markdown ```json o semplicemente le parentesi graffe
    match = re.search(r'\{.*\}', testo, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def valida_domande(env: str, lang: str, limit: int = None, model_name: str = None):
    test_file = os.path.join(TESTS_DIR, f"test_questions_{lang}.csv")
    if not os.path.exists(test_file):
        print(f"❌ File di test non trovato: {test_file}")
        return

    df = pd.read_csv(test_file)
    if limit:
        df = df.head(limit)

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
            max_tokens=1000
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
        page_text = carica_testo_pagina(expected_file, expected_page)

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

Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. Non aggiungere spiegazioni prima o dopo il JSON. Il JSON deve avere questa struttura esatta:
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
                motivo = f"Errore: Risposta del Judge non formattabile come JSON: {raw_content}"
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
    df_out.to_csv(out_file, index=False, encoding='utf-8')
    print(f"📊 Validazione completata. Risultati salvati in:\n{out_file}")
    
    print("\n--- REPORT DI VALIDAZIONE ---")
    print(f"  Domande analizzate: {len(df)}")
    print(f"  Domande VALIDE    : {validi} ({validi/len(df)*100:.1f}%)")
    print(f"  Domande NON VALIDE: {non_validi} ({non_validi/len(df)*100:.1f}%)")
    print("-----------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica la qualità e correttezza delle domande di test generate")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="cloud", help="Ambiente LLM Judge")
    parser.add_argument("--lang", type=str, choices=["it", "en"], default="it", help="Lingua del test")
    parser.add_argument("--limit", type=int, default=None, help="Limita il numero di domande da testare")
    parser.add_argument("--model", type=str, default=None, help="Modello OpenRouter da usare (es. openai/gpt-4o-mini)")
    args = parser.parse_args()

    valida_domande(args.env, args.lang, args.limit, args.model)
