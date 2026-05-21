import json
import os
import re
import glob

# 🆕 MODIFICA: Importiamo il modulo condiviso
from domain_enrichment import applica_domain_enrichment

def clean_text_for_chunking(text):
    if not text: return ""
    if re.search(r'[A-Z]{3,}OD|1MOCIDS|DLOHX|TESER|TRATS', text): return ""
    testo_senza_didascalie = text.replace('[Caption]', '').replace(':', '').strip()
    if text.count("[Caption]") > 3 and len(testo_senza_didascalie) < 100: return ""
    return text.strip()

def is_garbage_line(line):
    test_line = line.replace("[Caption]:", "").strip()
    if len(test_line) < 3: return True
    letters = len(re.findall(r'[a-zA-Z]', test_line))
    if letters == 0 and len(test_line) > 5: return True 
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', test_line))
    if special_chars > letters and len(test_line) > 10: return True
    return False

def clean_text_content(text):
    japanese_pattern = re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\uff00-\uffef]')
    text = japanese_pattern.sub(' ', text)
    text = re.sub(r'(\(\d+\))', r'\n\1', text)
    text = re.sub(r'(\([a-z]\)\s)', r'\n\1', text)
    text = re.sub(r'(\s-\s)', r'\n- ', text)
    text = re.sub(r'\b([a-zA-Z])\s(?=[a-zA-Z]\b)', r'\1', text)
    
    righe_pulite = []
    for riga in text.split('\n'):
        if '|' in riga and len(re.sub(r'[|\s]', '', riga)) == 0:
            continue
        righe_pulite.append(riga)
    text = '\n'.join(righe_pulite)
    
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def fix_hierarchy(record):
    match = re.match(r'^([A-Z]\.\d+\.?\d*)\s', record['title'])
    if match:
        record['section'] = match.group(1)
    return record

def preprocess_knowledge_base(input_folder, output_file):
    all_records = []
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    print(f"🧹 Inizio Preprocessing Euristico di {len(json_files)} file da {input_folder}...")

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for rec in data:
                if rec['title'].upper() in ["INDEX", "REVISION RECORD"]: continue
                if "[Caption]:" in rec['text'] and len(rec['text']) < 40: continue
                
                rec['text'] = clean_text_content(rec['text'])
                
                righe_pulite = [riga for riga in rec['text'].split('\n') if not is_garbage_line(riga)]
                rec['text'] = '\n'.join(righe_pulite).strip()
                rec['text'] = clean_text_for_chunking(rec['text'])
                
                if not rec['text']: continue

                title_clean = rec['title'].strip().upper()
                if rec['text'].upper().startswith(title_clean):
                    rec['text'] = rec['text'][len(title_clean):].strip()
                
                rec = fix_hierarchy(rec)
                all_records.append(rec)

    # 🆕 MODIFICA: Chiamata pulita al modulo condiviso (Agnostico)
    all_records = applica_domain_enrichment(all_records)

    cartella_output = os.path.dirname(output_file)
    os.makedirs(cartella_output, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=4, ensure_ascii=False)
    
    print(f"📂 Knowledge Base salvata in: {output_file}")

if __name__ == "__main__":
    percorso_input = "../../data/processed/euristico"
    percorso_output = "../../data/processed/knowledge/kb_euristico.json"
    preprocess_knowledge_base(percorso_input, percorso_output)