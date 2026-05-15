import json
import os
import re
# Richiede: pip install langchain-text-splitters tiktoken transformers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

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
            # Verifica che line2 contenga solo pipe, trattini, spazi o due punti
            if re.match(r'^[\s\|:\-]+$', line2):
                return f"{line1}\n{line2}"
    return None

def genera_dataset_chunks(input_file, chunk_size, chunk_overlap, tipo_modello, nome_modello):
    """
    Legge il JSON pulito, divide il testo usando il tokenizer corretto (Cloud o Locale)
    e salva un nuovo file JSON.
    """
    # MODIFICA PERCORSI: Salviamo nella cartella locale dell'esperimento
    output_dir = os.path.join("..", "es1_chunks")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"dataset_chunks_{tipo_modello}_{chunk_size}.json")
    
    print(f"🔪 Avvio chunking per {tipo_modello} ({chunk_size} token, Overlap: {chunk_overlap})...")
    
    # 1. Carica il JSON pulito
    with open(input_file, "r", encoding="utf-8") as f:
        dati_puliti = json.load(f)

    # 2. Configura lo Splitter in base al tipo di modello
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

    # 3. Processa ogni record
    for item in dati_puliti:
        testo_pulito = clean_text_for_chunking(item["text"])
        if not testo_pulito:
            continue
            
        testo_da_dividere = f"[{item['title']}]\n{testo_pulito}"
        frammenti = text_splitter.split_text(testo_da_dividere)
        
        current_table_header = None
        
        for i, frammento in enumerate(frammenti):
            header_in_frammento = extract_table_header(frammento)
            if header_in_frammento:
                current_table_header = header_in_frammento
            elif current_table_header and frammento.strip().startswith('|') and not extract_table_header(frammento[:200]):
                frammento = f"{current_table_header}\n{frammento.strip()}"

            if len(frammento) < 50:
                continue
                
            section_safe = str(item.get('section', '0')).replace('.', '_')
            
            # TRACCIAMENTO FONTE: Il cuore dell'esperimento
            chunk_record = {
                "chunk_id": f"{item['document_id']}_{section_safe}_{item['page']}_{i}_{tipo_modello}",
                "document_id": item["document_id"],
                "file_name": item["file_name"],
                "page": item["page"],
                "original_source_page": item.get('original_page', item['page']),
                "section": item["section"],
                "title": item["title"],
                "text": frammento,
                "token_size_target": chunk_size,
                "modello_target": tipo_modello,
                "char_count": len(frammento),
                "has_table": "|" in frammento and "---" in frammento,
                "has_alarm_code": bool(re.search(r'SRVO-\d{3}|SYST-\d{3}', frammento))
            }
            
            chunks_finali.append(chunk_record)

    # 4. Salva il file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks_finali, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Fatto! Creato {output_file} con {len(chunks_finali)} chunks.\n")

if __name__ == "__main__":
    # MODIFICA PERCORSI: Punta al file generato dal cleaner nella root dell'esperimento
    file_input = os.path.join("..", "es1_knowledge_base.json")
    
    if not os.path.exists(file_input):
        print(f"❌ Errore: Il file {file_input} non esiste. Esegui prima es1_data_cleaner.py")
    else:
        # 1. GENERIAMO I DATASET PER L'ESPERIMENTO LOCALE (BAAI/bge-m3)
        # Per l'esperimento 1 ci concentriamo sui test locali
        config_esperimenti_locale = [
            {"size": 300, "overlap": 40},
            {"size": 700, "overlap": 100},
            {"size": 1000, "overlap": 150}
        ]
        
        print("\n--- 💻 PREPARAZIONE DATI PER MODELLO LOCALE (ESPERIMENTO 1) ---")
        for config in config_esperimenti_locale:
            genera_dataset_chunks(
                input_file=file_input, 
                chunk_size=config["size"], 
                chunk_overlap=config["overlap"],
                tipo_modello="locale",
                nome_modello="BAAI/bge-m3"
            )
            
        print("🎉 Dataset isolati per l'esperimento pronti!")