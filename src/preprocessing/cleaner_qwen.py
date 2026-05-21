import json
import os
import re
import glob
import sys

# Riconfigura la codifica standard in UTF-8 se eseguito su Windows per evitare UnicodeEncodeError in console
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Aggiunge il path per importare moduli dal preprocessore e dalla radice del progetto
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(CURRENT_DIR)

import collections

# Importiamo il modulo condiviso di Domain Enrichment
from domain_ernichment import applica_domain_enrichment

def deduplicate_table_line(line):
    if '|' not in line:
        return line
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 6:
        return line
    
    non_empty = [p for p in parts if p]
    if not non_empty:
        return line
        
    counts = collections.Counter(non_empty)
    most_common = counts.most_common(1)[0]
    
    # Compress a sequence when it repeats a short token (>5 times) or longer token (>3 times)
    if (len(most_common[0]) >= 4 and most_common[1] > 3) or (len(most_common[0]) >= 2 and most_common[1] > 5):
        seen = set()
        new_parts = []
        for p in parts:
            if not p:
                new_parts.append("")
            elif p in seen:
                continue
            else:
                seen.add(p)
                new_parts.append(p)
        return " | ".join(new_parts)
    return line

def remove_internal_repetitions(text):
    if not text:
        return ""
        
    lines = text.split('\n')
    cleaned_lines = []
    seen_lines = set()
    
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            cleaned_lines.append("")
            continue
            
        # Keep table layout markdown dividers
        if '|' in line_strip and len(re.sub(r'[|\s\-\:]', '', line_strip)) == 0:
            cleaned_lines.append(line)
            continue
            
        line = deduplicate_table_line(line)
        line_strip = line.strip()
        
        # Line-level compression for repeating sentences/blocks
        if len(line_strip) >= 6:
            norm_line = re.sub(r'[^a-z0-9]', '', line_strip.lower())
            if norm_line in seen_lines:
                continue
                
            # Prefix check to prevent truncated repeating loops
            is_prefix = False
            for seen in seen_lines:
                if len(norm_line) >= 10 and seen.startswith(norm_line):
                    is_prefix = True
                    break
            if is_prefix:
                continue
                
            seen_lines.add(norm_line)
            
        cleaned_lines.append(line)
        
    text_clean = '\n'.join(cleaned_lines)
    text_clean = re.sub(r'\n{3,}', '\n\n', text_clean)
    return text_clean.strip()

def clean_text_for_chunking(text):
    """
    Rimuove chunk spazzatura o frammenti privi di valore informativo.
    """
    if not text:
        return ""
    # Rileva stringhe casuali o token binari residui
    if re.search(r'[A-Z]{3,}OD|1MOCIDS|DLOHX|TESER|TRATS', text):
        return ""
    
    return text.strip()

def is_garbage_line(line):
    """
    Identifica righe orfane, rumore, didascalie o formattazione errata.
    """
    test_line = line.strip()
    if len(test_line) < 3:
        return True
    
    # 1. Rileva didascalie di figure e tabelle (es: Fig.4.1 Main board, Figure 3, Table 5.2, etc.)
    if re.match(r'^(Fig\.|Figure|Table|Tab\.)\s*\d+', test_line, re.IGNORECASE):
        return True
        
    # 2. Rileva numeri di pagina orfani (es. "- 93 -" o "- 100 -")
    if re.match(r'^\s*-\s*\d+\s*-\s*$', test_line):
        return True

    # 3. Rileva intestazioni/codici del manuale Fanuc (es. B-83525EN/07 o B-83525EN/07 MAINTENANCE)
    if re.search(r'B-\d{5}EN/\d+', test_line):
        return True

    # 4. Rileva descrizioni visive orfane residue
    if re.match(r'^\((Connector converter board|Terminal converter board|R-30\s*i\s*B\s*Mate|Connector converter|Terminal converter)\)$', test_line, re.IGNORECASE):
        return True

    # Controlli di densità caratteri per escludere frammenti di layout orfani
    letters = len(re.findall(r'[a-zA-Z]', test_line))
    if letters == 0 and len(test_line) > 5:
        # Se non contiene lettere ed è lunga, probabilmente è rumore grafico (es. linee tratteggiate)
        return True 
    
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', test_line))
    if special_chars > letters and len(test_line) > 10:
        return True
        
    return False

def clean_text_content(text):
    """
    Pulisce il testo estratto rimuovendo caratteri non ASCII e
    ottimizzando la formattazione dei paragrafi e delle tabelle markdown.
    """
    if not text:
        return ""
    
    # Rimuove ideogrammi giapponesi residui
    japanese_pattern = re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\uff00-\uffef]')
    text = japanese_pattern.sub(' ', text)
    
    # Normalizza ritorni a capo per elenchi e parentesi numeriche
    text = re.sub(r'(\(\d+\))', r'\n\1', text)
    text = re.sub(r'(\([a-z]\)\s)', r'\n\1', text)
    text = re.sub(r'(\s-\s)', r'\n- ', text)
    
    # Riunisce caratteri spaccati da errori di codifica
    text = re.sub(r'\b([a-zA-Z])\s(?=[a-zA-Z]\b)', r'\1', text)
    
    # Filtra righe di tabelle vuote o formattazione markdown orfana
    righe_pulite = []
    for riga in text.split('\n'):
        if '|' in riga and len(re.sub(r'[|\s\-\:]', '', riga)) == 0:
            continue
        righe_pulite.append(riga)
    text = '\n'.join(righe_pulite)
    
    # Normalizza spazi bianchi consecutivi
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()

def strip_redundant_header(text, title, section=None):
    """
    Rimuove l'intestazione iniziale se corrisponde al titolo della sezione,
    anche in presenza di marcatori markdown (es. ##, ###) e numeri di sezione.
    """
    if not text:
        return ""
    
    linee = text.split('\n')
    if not linee:
        return text
    
    prima_riga = linee[0].strip()
    riga_senza_md = re.sub(r'^#+\s*', '', prima_riga).strip()
    
    title_clean = title.strip().upper()
    riga_upper = riga_senza_md.upper()
    
    # 1. Corrispondenza esatta del titolo
    if riga_upper == title_clean:
        return '\n'.join(linee[1:]).strip()
        
    # 2. Corrispondenza con sezione + titolo (es. "2.3 CHECKS AND MAINTENANCE" o "4.4 PROCESS I/O...")
    riga_senza_sezione = re.sub(r'^([A-Z0-9\.\-]+\s+)', '', riga_upper).strip()
    if riga_senza_sezione == title_clean:
        return '\n'.join(linee[1:]).strip()
        
    # 3. Se il titolo è contenuto all'inizio ed è seguito da punteggiatura/numeri trascurabili
    if title_clean in riga_upper and len(riga_upper) < len(title_clean) + 15:
        residuo = riga_upper.replace(title_clean, "").strip()
        if not residuo or re.match(r'^[\d\.\-\s\:\(\)]*$', residuo):
            return '\n'.join(linee[1:]).strip()
            
    # 4. Rimuove prefissi o titoli di sola sezione (es. "## 4.1" o "## 4.")
    if section:
        section_clean = section.strip()
        if riga_senza_md == section_clean or riga_senza_md.replace('.', '') == section_clean.replace('.', ''):
            return '\n'.join(linee[1:]).strip()

    return text

def fix_hierarchy(record):
    """
    Regola la gerarchia della sezione in base al titolo estratto (es. A.1 Titolo -> Sezione A.1).
    """
    match = re.match(r'^([A-Z]\.\d+\.?\d*)\s', record['title'])
    if match:
        record['section'] = match.group(1)
    return record

def preprocess_knowledge_base(input_folder, output_file):
    """
    Carica i file estratti da Qwen VLM, applica filtri di pulizia avanzati,
    risolve le cross-references tramite domain enrichment e salva il file finale JSON.
    """
    all_records = []
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    print(f"🧹 [QWEN CLEANER] Inizio Preprocessing di {len(json_files)} file da {input_folder}...")

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for rec in data:
                # Salta pagine di indice e documentazione amministrativa o codici manuale
                title_upper = rec['title'].upper()
                if title_upper in ["INDEX", "REVISION RECORD"]:
                    continue
                
                # Esegue la pulizia del testo principale
                rec['text'] = clean_text_content(rec['text'])
                
                # Rimuove loop interni, righe ripetitive e tabelle corrotte
                rec['text'] = remove_internal_repetitions(rec['text'])
                
                # Rimuove righe non informative, didascalie o piene di rumore grafico
                righe_pulite = [riga for riga in rec['text'].split('\n') if not is_garbage_line(riga)]
                rec['text'] = '\n'.join(righe_pulite).strip()
                
                # Applica i filtri per il chunking finale
                rec['text'] = clean_text_for_chunking(rec['text'])
                
                # Evita ridondanza del titolo all'interno del testo
                rec['text'] = strip_redundant_header(rec['text'], rec['title'], rec.get('section'))
                
                # Se dopo la pulizia il testo è vuoto o sotto la soglia di 15 caratteri informativi, lo scartiamo
                if not rec['text'] or len(rec['text'].strip()) < 15:
                    continue
                
                # Ripristina e corregge la gerarchia strutturale
                rec = fix_hierarchy(rec)
                all_records.append(rec)

    # Applica il Domain Enrichment per le cross-references dei manuali Fanuc
    all_records = applica_domain_enrichment(all_records)

    # Rimuove record duplicati a livello globale (stesso file, pagina, titolo, testo)
    all_records_clean = []
    seen_records = set()
    for rec in all_records:
        rec_key = (rec["file_name"], rec["page"], rec["title"].lower().strip(), rec["text"].lower().strip())
        if rec_key in seen_records:
            continue
        seen_records.add(rec_key)
        all_records_clean.append(rec)
    all_records = all_records_clean

    # Crea la directory di output se non esiste
    cartella_output = os.path.dirname(output_file)
    os.makedirs(cartella_output, exist_ok=True)

    # Salva il dataset finale per il RAG
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=4, ensure_ascii=False)
    
    print(f"📂 [QWEN CLEANER] Knowledge Base finale salvata con successo in: {output_file} ({len(all_records)} record)\n")

if __name__ == "__main__":
    percorso_input = os.path.join(PROJECT_ROOT, "data", "processed", "qwen")
    percorso_output = os.path.join(PROJECT_ROOT, "data", "processed", "knowledge", "kb_qwen.json")
    preprocess_knowledge_base(percorso_input, percorso_output)