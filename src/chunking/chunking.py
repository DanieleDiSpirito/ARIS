import json
import os
import re
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# Configura console Windows per supportare emoji ed encoding UTF-8 senza crash
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Determina la root del repository ARIS in base alla posizione dello script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

def clean_text_for_chunking(text):
    """
    Rimuove rumore come testo specchiato o blocchi composti solo da didascalie.
    """
    if not text:
        return None
        
    # Rimuove testo specchiato (es. terminal converter board)
    if re.search(r'[A-Z]{3,}OD|1MOCIDS|DLOHX|TESER|TRATS', text):
        return None
    
    # Rimuove blocchi dove le didascalie sono preponderanti rispetto al testo utile
    testo_senza_didascalie = text.replace('[Caption]', '').replace(':', '').strip()
    if text.count("[Caption]") > 3 and len(testo_senza_didascalie) < 100:
        return None
        
    return text.strip()

def extract_table_header(text):
    """
    Estrae l'header di una tabella Markdown se presente nel testo.
    Cerca due righe consecutive: una con i nomi delle colonne, l'altra con i separatori.
    """
    lines = text.split('\n')
    for i in range(len(lines) - 1):
        line1 = lines[i].strip()
        line2 = lines[i+1].strip()
        if line1.startswith('|') and line2.startswith('|') and '-' in line2:
            if re.match(r'^[\s\|:\-]+$', line2):
                return f"{line1}\n{line2}"
    return None

def genera_dataset_chunks(input_file, chunk_size, chunk_overlap, tipo_modello, nome_modello, metodo_estrazione):
    """
    Legge il JSON pulito, divide il testo usando il tokenizer corretto (Cloud o Locale)
    e salva un nuovo file JSON nella cartella specifica per il metodo.
    """
    output_dir = os.path.join(REPO_ROOT, "data", "chunks", metodo_estrazione)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"dataset_chunks_{tipo_modello}_{chunk_size}.json")
    
    print(f"🔪 Avvio chunking per metodo '{metodo_estrazione}' - modello '{tipo_modello}' ({chunk_size} token, Overlap: {chunk_overlap})...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        dati_puliti = json.load(f)

    if tipo_modello == "cloud":
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=nome_modello,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    elif tipo_modello == "locale":
        tokenizer = AutoTokenizer.from_pretrained(nome_modello)
        
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    else:
        raise ValueError("tipo_modello deve essere 'cloud' o 'locale'")

    chunks_finali = []

    for item in dati_puliti:
        testo_pulito = clean_text_for_chunking(item["text"])
        if not testo_pulito:
            continue
            
        testo_da_dividere = f"[{item['title']}]\n{testo_pulito}"
        
        frammenti = text_splitter.split_text(testo_da_dividere)
        
        current_table_header = None
        
        for i, frammento in enumerate(frammenti):
            # --- Context-Aware Chunking per le Tabelle ---
            header_in_frammento = extract_table_header(frammento)
            if header_in_frammento:
                current_table_header = header_in_frammento
            # Se è una continuazione di tabella (inizia con '|' ma senza separatori nelle prime righe), iniettiamo l'header
            elif current_table_header and frammento.strip().startswith('|') and not extract_table_header(frammento[:200]):
                frammento = f"{current_table_header}\n{frammento.strip()}"

            # Scarta i chunk troppo corti che non hanno valore semantico
            if len(frammento) < 50:
                continue
                
            # Risolve duplicati includendo la section nel chunk_id
            section_safe = str(item.get('section', '0')).replace('.', '_')
            
            # Determina has_table PRIMA di rimuovere i trattini
            has_table_flag = "|" in frammento and "---" in frammento
            
            # Rimuove le righe divisorie di Markdown (es. |---|---|) dal testo finale
            testo_finale = re.sub(r'^[\s\|:\-]+$\n?', '', frammento, flags=re.MULTILINE).strip()

            chunk_record = {
                "chunk_id": f"{item['document_id']}_{section_safe}_{item['page']}_{i}_{tipo_modello}_{metodo_estrazione}",
                "document_id": item["document_id"],
                "file_name": item["file_name"],
                "page": item["page"],
                "section": item["section"],
                "title": item["title"],
                "text": testo_finale,
                "token_size_target": chunk_size,
                "modello_target": tipo_modello,
                "char_count": len(testo_finale),
                "has_table": has_table_flag,
                "has_alarm_code": bool(re.search(r'SRVO-\d{3}|SYST-\d{3}', frammento))
            }
            chunks_finali.append(chunk_record)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks_finali, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Fatto! Creato {output_file} con {len(chunks_finali)} chunks.\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunking universale per la pipeline ARIS.")
    parser.add_argument(
        "--metodo", "-m", 
        type=str, 
        choices=["euristico", "docling", "llamaparse", "qwen", "pdf4llm", "all"],
        default="docling",
        help="Metodo di estrazione da elaborare (euristico, docling, llamaparse, qwen, pdf4llm, all)."
    )
    args = parser.parse_args()
    
    metodi_disponibili = ["euristico", "docling", "llamaparse", "qwen", "pdf4llm"]
    
    if args.metodo == "all":
        metodi_da_elaborare = metodi_disponibili
    else:
        metodi_da_elaborare = [args.metodo]
        
    for metodo in metodi_da_elaborare:
        input_file = os.path.join(REPO_ROOT, "data", "processed", "knowledge", f"kb_{metodo}.json")
        
        if not os.path.exists(input_file):
            print(f"⚠️  File di input non trovato per '{metodo}': {input_file}. Salto.")
            continue
            
        print(f"\n📂 =========================================")
        print(f"📂 ELABORAZIONE CHUNKING METODO: {metodo.upper()}")
        print(f"📂 =========================================")
        
        config_esperimenti_cloud = [
            {"size": 300, "overlap": 40},
            {"size": 700, "overlap": 100},
            {"size": 1000, "overlap": 150}
        ]
        
        print("\n--- ☁️ PREPARAZIONE DATI PER MODELLO CLOUD ---")
        for config in config_esperimenti_cloud:
            genera_dataset_chunks(
                input_file=input_file, 
                chunk_size=config["size"], 
                chunk_overlap=config["overlap"],
                tipo_modello="cloud",
                nome_modello="text-embedding-3-small",
                metodo_estrazione=metodo
            )
            
        config_esperimenti_locale = [
            {"size": 300, "overlap": 40},
            {"size": 700, "overlap": 100},
            {"size": 1000, "overlap": 150}
        ]
        
        print("\n--- 💻 PREPARAZIONE DATI PER MODELLO LOCALE ---")
        for config in config_esperimenti_locale:
            genera_dataset_chunks(
                input_file=input_file, 
                chunk_size=config["size"], 
                chunk_overlap=config["overlap"],
                tipo_modello="locale",
                nome_modello="BAAI/bge-m3",
                metodo_estrazione=metodo
            )
            
    print("🎉 Elaborazione di chunking completata con successo per tutti i metodi richiesti!")