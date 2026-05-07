import json
import os
import re
import glob

def is_garbage_line(line):
    """
    Rileva se una riga è rumore da diagramma, ignorando la parola 'Caption'.
    """
    # Rimuove il tag per non falsare il conteggio delle lettere
    test_line = line.replace("[Caption]:", "").strip()
    
    if len(test_line) < 3: 
        return True
        
    letters = len(re.findall(r'[a-zA-Z]', test_line))
    
    # Se la riga è lunga ma non ha lettere (es. " ) / / / "), è rumore
    if letters == 0 and len(test_line) > 5: 
        return True 
        
    # Se ci sono più caratteri speciali che lettere
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', test_line))
    if special_chars > letters and len(test_line) > 10:
        return True
        
    return False

def clean_text_content(text):
    """
    Applica la pulizia semantica al testo mantenendo i simboli tecnici (±, Ω, °, µ).
    """
    # 1. Rimuove SPECIFICAMENTE i caratteri giapponesi (Hiragana, Katakana, Kanji, Full-width)
    japanese_pattern = re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\uff00-\uffef]')
    text = japanese_pattern.sub(' ', text)
    
    # 2. Ricostruzione Elenchi: va a capo prima di (1), (a), o trattini
    text = re.sub(r'(\(\d+\))', r'\n\1', text)
    text = re.sub(r'(\([a-z]\)\s)', r'\n\1', text)
    text = re.sub(r'(\s-\s)', r'\n- ', text)
    
    # 3. Ricompatta le lettere singole per correggere frammentazioni OCR
    text = re.sub(r'\b([a-zA-Z])\s(?=[a-zA-Z]\b)', r'\1', text)
    
    # 4. MICRO-ASPIRAPOLVERE: Rimuove righe di tabelle vuote (es. | | | o |---|---| )
    righe_pulite = []
    for riga in text.split('\n'):
        if '|' in riga:
            # Se la riga contiene solo pipe, trattini e spazi, saltala
            if len(re.sub(r'[|\-\s]', '', riga)) == 0:
                continue
        righe_pulite.append(riga)
    text = '\n'.join(righe_pulite)
    
    # 5. Pulizia spazi multipli e ritorni a capo
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()

def fix_hierarchy(record):
    """
    Ripara le sezioni 'incastrate' (es. 5.6.4) usando i titoli.
    """
    match = re.match(r'^([A-Z]\.\d+\.?\d*)\s', record['title'])
    if match:
        record['section'] = match.group(1)
    return record

def preprocess_knowledge_base(input_folder, output_file):
    all_records = []
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    print(f"🧹 Inizio Preprocessing di {len(json_files)} file...")

    alarm_map = {}

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for rec in data:
                # A. Filtro Indice e Revisioni
                if rec['title'].upper() in ["INDEX", "REVISION RECORD"]:
                    continue
                
                # B. Filtro Didascalie brevi
                if "[Caption]:" in rec['text'] and len(rec['text']) < 40:
                    continue
                
                # C. Normalizzazione testo e rimozione righe tabella vuote
                rec['text'] = clean_text_content(rec['text'])
                
                # D. Filtro Garbage Line
                righe_pulite = [riga for riga in rec['text'].split('\n') if not is_garbage_line(riga)]
                rec['text'] = '\n'.join(righe_pulite).strip()
                
                if not rec['text']:
                    continue

                # E. Deduplicazione Titolo-Testo
                title_clean = rec['title'].strip().upper()
                if rec['text'].upper().startswith(title_clean):
                    rec['text'] = rec['text'][len(title_clean):].strip()
                
                # F. Fix Gerarchia Appendici
                rec = fix_hierarchy(rec)

                # G. Popola mappa allarmi (Unendo il testo se l'allarme occupa più pagine!)
                if "SRVO-" in rec['title']:
                    code = rec['title'].split(' ')[0]
                    if code in alarm_map:
                        alarm_map[code] += "\n" + rec['text']
                    else:
                        alarm_map[code] = rec['text']
                
                all_records.append(rec)

    # 2. Iniezione Cross-References
    for rec in all_records:
        xref_match = re.search(r'same actions as (SRVO-\d+)', rec['text'])
        if xref_match:
            code_ref = xref_match.group(1)
            if code_ref in alarm_map:
                rec['text'] += f"\n\n[ENRICHMENT FROM {code_ref}]:\n" + alarm_map[code_ref]
                print(f"    ⚓ Arricchito {rec['title']} con dati da {code_ref}")

    with open("../../data/processed/" + output_file, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Preprocessing completato! Creati {len(all_records)} record puliti e arricchiti.")
    print(f"📂 File salvato: {output_file}")

if __name__ == "__main__":
    preprocess_knowledge_base("../../data/processed", "knowledge_base.json")
    