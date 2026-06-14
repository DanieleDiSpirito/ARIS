import os
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(BASE_DIR)

def traduci_testo(llm, testo: str, tipo: str) -> str:
    """Traduce un testo dall'italiano all'inglese mantenendo intatti i termini tecnici Fanuc."""
    prompt = f"""Sei un traduttore tecnico professionista specializzato in automazione industriale e robotica Fanuc.
Traduci il seguente testo (che è una {tipo} per un test di QA su un sistema RAG) dall'italiano all'inglese.

Regole di traduzione:
1. Mantieni i termini tecnici consolidati in lingua originale se sono comunemente usati in inglese o nel settore (es. "teach pendant", "deadman switch", "backup", "jog", "servo amplificatore" -> "servo amplifier", "scheda principale" -> "main board", "arresto d'emergenza" -> "emergency stop").
2. Mantieni inalterati i codici degli allarmi (es. SRVO-005, PRIO-095).
3. Non aggiungere commenti, introduzioni o spiegazioni. Rispondi solo con la traduzione esatta.

Testo in italiano:
{testo}

Traduzione in inglese:"""
    
    success = False
    retry_count = 3
    while not success and retry_count > 0:
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"   ⚠️ Errore di traduzione: {e}. Tentativi rimasti: {retry_count-1}")
            retry_count -= 1
            time.sleep(1)
    return testo  # Fallback

def main():
    parser = argparse.ArgumentParser(description="Traduce il test set da italiano a inglese")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="cloud", help="Ambiente LLM")
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini", help="Modello OpenRouter da usare (se env=cloud)")
    args = parser.parse_args()

    input_file = os.path.join(TESTS_DIR, "test_questions_it.csv")
    output_file = os.path.join(TESTS_DIR, "test_questions_en.csv")

    if not os.path.exists(input_file):
        print(f"❌ File di input non trovato: {input_file}")
        return

    print(f"📋 Caricamento del file in italiano: {input_file}")
    df = pd.read_csv(input_file)

    if len(df) == 0:
        print("❌ Il file è vuoto.")
        return

    if args.env == "locale":
        print("🤖 Traduttore LLM: Locale (localhost:1234)")
        local_model = os.getenv("LOCAL_LLM_MODEL", None)
        base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
        llm = ChatOpenAI(
            model=local_model,
            base_url=base_url,
            api_key="lm-studio",
            temperature=0.0
        )
    elif args.env == "cloud":
        print("☁️ Traduttore LLM: Cloud (OpenRouter)...")
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("❌ ERRORE: Variabile OPENAI_API_KEY non trovata nel file .env")

        llm = ChatOpenAI(
            model=args.model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
            max_tokens=1000
        )
    else:
        raise ValueError("Il parametro env deve essere 'locale' o 'cloud'")

    tradotti = []
    print(f"\n🚀 Avvio traduzione di {len(df)} domande...\n")

    for idx, row in df.iterrows():
        qid = row["id"]
        print(f"[{idx+1}/{len(df)}] Traduzione {qid}...")
        
        q_en = traduci_testo(llm, row["question"], "domanda")
        a_en = traduci_testo(llm, row["expected_answer"], "risposta attesa")

        tradotti.append({
            "id": qid,
            "question": q_en,
            "category": row["category"],
            "expected_answer": a_en,
            "expected_file": row["expected_file"],
            "expected_page": row["expected_page"],
            "difficulty": row["difficulty"]
        })

    # Scrive il nuovo file CSV con il formato richiesto
    df_out = pd.DataFrame(tradotti)
    
    # Salvataggio nel formato personalizzato
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        f.write("id,question,category,expected_answer,expected_file,expected_page,difficulty\n")
        for _, row_out in df_out.iterrows():
            qid = str(row_out["id"])
            
            def clean_and_quote(val):
                val_str = str(val) if pd.notna(val) else ""
                val_escaped = val_str.replace('"', '""')
                return f'"{val_escaped}"'
            
            question = clean_and_quote(row_out["question"])
            category = clean_and_quote(row_out["category"])
            expected_answer = clean_and_quote(row_out["expected_answer"])
            expected_file = clean_and_quote(row_out["expected_file"])
            expected_page = clean_and_quote(row_out["expected_page"])
            difficulty = clean_and_quote(row_out["difficulty"])
            
            f.write(f"{qid},{question},{category},{expected_answer},{expected_file},{expected_page},{difficulty}\n")

    print(f"\n📊 Traduzione completata con successo!")
    print(f"💾 Salvate {len(df_out)} domande tradotte in:\n{output_file}\n")

if __name__ == "__main__":
    main()
