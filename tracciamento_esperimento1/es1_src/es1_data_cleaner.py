import json
import os
import re
import glob

def is_garbage_line(line):
    """Rileva se una riga è rumore da diagramma."""
    test_line = line.replace("[Caption]:", "").strip()
    if len(test_line) < 3: return True
    letters = len(re.findall(r'[a-zA-Z]', test_line))
    if letters == 0 and len(test_line) > 5: return True 
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', test_line))
    if special_chars > letters and len(test_line) > 10: return True
    return False

def clean_text_content(text):
    """Pulizia semantica del testo."""
    japanese_pattern = re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\uff00-\uffef]')
    text = japanese_pattern.sub(' ', text)
    text = re.sub(r'(\(\d+\))', r'\n\1', text)
    text = re.sub(r'(\([a-z]\)\s)', r'\n\1', text)
    text = re.sub(r'(\s-\s)', r'\n- ', text)
    text = re.sub(r'\b([a-zA-Z])\s(?=[a-zA-Z]\b)', r'\1', text)
    righe_pulite = []
    for riga in text.split('\n'):
        if '|' in riga:
            if len(re.sub(r'[|\-\s]', '', riga)) == 0: continue
        righe_pulite.append(riga)
    text = '\n'.join(righe_pulite)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def fix_hierarchy(record):
    """Ripara le sezioni usando i titoli."""
    match = re.match(r'^([A-Z]\.\d+\.?\d*)\s', record['title'])
    if match: record['section'] = match.group(1)
    return record

def preprocess_knowledge_base(input_folder, output_file):
    all_records = []
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    print(f"🧹 Inizio Preprocessing di {len(json_files)} file...")

    alarm_map = {}

    # 1. Prima passata: Pulizia e mappatura allarmi
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for rec in data:
                if rec['title'].upper() in ["INDEX", "REVISION RECORD"]: continue
                if "[Caption]:" in rec['text'] and len(rec['text']) < 40: continue
                
                rec['text'] = clean_text_content(rec['text'])
                righe_pulite = [riga for riga in rec['text'].split('\n') if not is_garbage_line(riga)]
                rec['text'] = '\n'.join(righe_pulite).strip()
                
                if not rec['text']: continue

                title_clean = rec['title'].strip().upper()
                if rec['text'].upper().startswith(title_clean):
                    rec['text'] = rec['text'][len(title_clean):].strip()
                
                rec = fix_hierarchy(rec)

                # Mappatura per l'arricchimento
                if "SRVO-" in rec['title']:
                    code = rec['title'].split(' ')[0]
                    if code in alarm_map:
                        alarm_map[code]['text'] += "\n" + rec['text']
                    else:
                        alarm_map[code] = {'text': rec['text'], 'page': rec.get('page', 'N/A')}
                
                all_records.append(rec)

    # 2. Seconda passata: Iniezione Cross-References (VERSIONE UNIVERSALE)
    print(f"⚓ Avvio iniezione cross-references su {len(all_records)} record...")
    
    for rec in all_records:
        # Troviamo tutti i codici SRVO-XXX nel testo
        codici_trovati = re.findall(r'SRVO-(\d+)', rec['text'])
        
        for num in codici_trovati:
            code_ref = f"SRVO-{num}"
            
            # Arricchiamo solo se il codice esiste in mappa e NON è l'allarme corrente
            if code_ref in alarm_map and code_ref not in rec['title']:
                source_data = alarm_map[code_ref]
                
                # Evitiamo duplicati nello stesso record
                if f"[ENRICHMENT FROM {code_ref}]" not in rec['text']:
                    rec['text'] += f"\n\n[ENRICHMENT FROM {code_ref} - ORIGINAL PAGE: {source_data['page']}]:\n" + source_data['text']
                    # Metadato per lo script di test
                    rec['original_page'] = source_data['page']
                    print(f"    -> Collegato {rec['title']} all'allarme {code_ref} (Pag. {source_data['page']})")

    # Salvataggio
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Preprocessing completato! File salvato: {output_file}")

if __name__ == "__main__":
    preprocess_knowledge_base("../es1_processed", "../es1_knowledge_base.json")