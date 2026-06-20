import os
import sys
import csv
import glob
import json
import re
from sys import argv

# Riconfigura la codifica standard in UTF-8 se eseguito su Windows per evitare UnicodeEncodeError in console
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Configurazione dei path per consentire l'importazione della telemetria
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

from src.utils.telemetry import misura_performance

# Import nativi di pymupdf4llm
try:
    import pymupdf4llm
except ImportError:
    print("❌ Errore: Libreria 'pymupdf4llm' non installata. Esegui: pip install pymupdf4llm")
    sys.exit(1)


def leggi_metadati_csv(percorso_csv, nome_file_pdf):
    """
    Legge l'indice documentale per estrarre l'ID e la pagina iniziale del manuale.
    Mantiene la compatibilità assoluta con gli altri loader del progetto.
    """
    document_id = "00"
    pagina_iniziale_int = 0
    sezione_speciale = False

    try:
        with open(percorso_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['nome_file'] == nome_file_pdf:
                    document_id = row['id_documento']
                    pag_str = row['pagina_manuale']
                    
                    if pag_str.startswith('s'):
                        sezione_speciale = True
                        pagina_iniziale_int = 1
                    else:
                        pagina_iniziale_int = int(pag_str)
                    break
    except FileNotFoundError:
        print(f"⚠️ ATTENZIONE: File CSV {percorso_csv} non trovato.")
        
    return document_id, pagina_iniziale_int, sezione_speciale


def calcola_nome_pagina(indice_pagina_pdf, pagina_iniziale_int, sezione_speciale):
    """Calcola la stringa della pagina corretta (es. '45' o 's-3')"""
    if sezione_speciale:
        return f"s-{pagina_iniziale_int + indice_pagina_pdf}"
    else:
        return str(pagina_iniziale_int + indice_pagina_pdf)


def iper_pulizia_campo(testo):
    """Rimuove eventuali tag markdown residui dai titoli delle sezioni"""
    if not testo:
        return "Generale"
    cleaned = re.sub(r'^[#\s\-\*\_\`\:]+', '', testo)
    cleaned = re.sub(r'[#\s\-\*\_\`\:]+$', '', cleaned)
    return cleaned.strip()


def is_callout_or_header_to_ignore(text):
    """
    Ritorna True se l'intestazione è un richiamo (callout) o rumore ripetitivo di intestazione pagina.
    """
    # Pulisce marcatori di stile markdown ed eventuali caratteri speciali sia all'inizio che alla fine
    cleaned = re.sub(r'^[#\s\-\*\_\`\:]+', '', text)
    cleaned = re.sub(r'[#\s\-\*\_\`\:]+$', '', cleaned)
    cleaned = cleaned.strip().upper()
    
    if not cleaned:
        return True
    
    # 1. Parole chiave dei richiami (Callouts) e intestazioni orfane
    if cleaned in ["NOTE", "WARNING", "CAUTION", "IMPORTANT", "NOTICE", "DANGER", 
                   "PRECAUTION", "PRECAUTIONS", "ATTENZIONE", "AVVERTENZA", "AVVISO", 
                   "NOTE IN CASE OF CE CONTROLLER"]:
        return True
        
    # 2. Intestazioni di pagina ripetitive / capitoli / titoli di manuali (Running Headers)
    if cleaned in [
        "MAINTENANCE", "SAFETY PRECAUTIONS", "TROUBLESHOOTING", "OVERVIEW", 
        "OVERVIEW AND CONFIGURATION", "CONFIGURATION", "CHECKS AND MAINTENANCE", 
        "DIAGNOSTICS", "VISUAL DIAGNOSTICS", "PRINTED CIRCUIT BOARDS", "AMPLIFIERS", 
        "REPLACING UNITS", "CONNECTIONS", "CABLE CONNECTION", "CONNECTION DIAGRAM"
    ]:
        return True
    
    # 3. Codici Fanuc di documentazione (es: B-83525EN/07)
    if re.search(r'B-\d{5}EN/\d+', cleaned):
        return True
        
    # 4. Didascalie di figure o tabelle
    if re.match(r'^(FIG\.|FIGURE|TABLE|TAB\.)\s*\d+', cleaned):
        return True
        
    # 5. Numeri di pagina o indicatori orfani
    if re.match(r'^-\s*\d+\s*-$', cleaned) or re.match(r'^\d+$', cleaned):
        return True
        
    return False



def is_parent_section(new_sec, current_sec):
    """
    Ritorna True se la nuova sezione candidata è un genitore o meno specifica 
    rispetto alla sezione corrente attiva (evitando downgrade gerarchici dovuti a page headers).
    """
    if current_sec == "Generale":
        return False
        
    ns = re.sub(r'^APPENDIX\s+', '', new_sec, flags=re.IGNORECASE).strip().upper()
    cs = re.sub(r'^APPENDIX\s+', '', current_sec, flags=re.IGNORECASE).strip().upper()
    
    if not ns or ns in ["GENERALE", "APPENDIX"]:
        return True
        
    if cs == ns:
        return False
        
    if cs.startswith(ns + ".") or cs.startswith(ns + " "):
        return True
        
    return False


def dividi_pagina_in_blocchi_strutturati(markdown_text, document_id, nome_file, page_label, stato_sezione):
    """
    Divide il markdown della pagina in blocchi separati per ciascuna sezione/titolo trovati.
    Riconosce sia intestazioni markdown standard (#) sia intestazioni implicite non taggate 
    (es. "B. TOTAL CONNECTION DIAGRAM" o "4.1 Main Board") per garantire un parsing perfetto.
    Ritorna una lista di dizionari conformi allo schema a 6 campi e lo stato di sezione aggiornato.
    """
    blocchi = []
    linee_accumulate = []
    
    sezione_corrente = stato_sezione["sezione"]
    titolo_corrente = stato_sezione["titolo"]
    
    for line in markdown_text.split("\n"):
        line_strip = line.strip()
        if not line_strip:
            continue
            
        # 1. Rileva intestazioni standard (# a ######)
        match_heading = re.match(r"^(#{1,6})\s+(.*)", line_strip)
        
        # 2. Rileva intestazioni implicite non formattate (es. "B. TOTAL CONNECTION DIAGRAM")
        is_implicit_heading = False
        implicit_text = ""
        
        if not match_heading and len(line_strip) < 80 and not line_strip.startswith(("-", "*", "|", "(", "[", "{", "•")):
            # Un'intestazione implicita reale NON deve terminare con un punto o altri segni di punteggiatura da frase
            if not line_strip.endswith((".", "?", "!", ";", ",")):
                match_appendix = re.match(r"^(APPENDIX\s+[A-Z0-9])", line_strip, re.IGNORECASE)
                match_multilevel = re.match(r"^([A-Z0-9]\.[\d\.]+)\s+", line_strip, re.IGNORECASE)
                match_single = re.match(r"^([A-Z0-9]\.?)\s+(.*)", line_strip, re.IGNORECASE)
                
                is_valid = False
                if match_appendix:
                    is_valid = True
                elif match_multilevel:
                    # Numero multi-livello (es. 4.1 o A.1) indica chiaramente una sezione reale
                    is_valid = True
                elif match_single:
                    num_or_letter = match_single.group(1).rstrip('.')
                    rest_of_title = match_single.group(2).strip()
                    
                    # Se è un singolo numero (es. "1") o una singola lettera (es. "A"),
                    # deve essere in ALL CAPS per essere considerato un capitolo principale (es. "1 OVERVIEW")
                    # ed evitare elenchi numerati del tipo "1. The peripheral..." o "1 Pull up..."
                    if num_or_letter.isalnum() and len(num_or_letter) == 1:
                        if rest_of_title.isupper():
                            is_valid = True
                    else:
                        is_valid = True
                        
                if is_valid:
                    is_implicit_heading = True
                    implicit_text = line_strip
                
        testo_heading = ""
        if match_heading:
            testo_heading = match_heading.group(2).strip()
        elif is_implicit_heading:
            testo_heading = implicit_text
            
        # Controlla se l'intestazione è solo un callout o rumore di layout
        if testo_heading and is_callout_or_header_to_ignore(testo_heading):
            testo_heading = ""
            
        if testo_heading:
            # Pulisce i marcatori di stile
            testo_heading_clean = re.sub(r'[*_`]', '', testo_heading).strip()
            
            # --- PARSING ESTRATTIVO AVANZATO DELLA GERARCHIA ---
            # 1. Caso Appendice: "APPENDIX B. TOTAL CONNECTION DIAGRAM" o "APPENDIX B"
            match_appendix = re.match(r"^(APPENDIX\s+[A-Z0-9](?:\.[\d\.]*)?)(?:\s+(.*))?$", testo_heading_clean, re.IGNORECASE)
            
            # 2. Caso Sezione Standard con titolo: "4.1.2 Main Board" o "A.1 Test"
            match_sezione = re.match(r"^([A-Z0-9]\.[\d\.]*)\s+(.*)", testo_heading_clean, re.IGNORECASE)
            
            # 3. Caso Sezione Cifra con titolo: "4 Main Board"
            match_cifra = re.match(r"^(\d+)\s+(.*)", testo_heading_clean)
            
            # 4. Caso Solo Numero Sezione: "4" o "4.1" o "B"
            match_solo_sezione = re.match(r"^([A-Z0-9](?:\.[\d\.]*)+)$", testo_heading_clean, re.IGNORECASE)
            
            candidate_sezione = sezione_corrente
            candidate_titolo = titolo_corrente
            
            if match_appendix:
                candidate_sezione = match_appendix.group(1).strip().rstrip('.')
                candidate_titolo = match_appendix.group(2).strip() if match_appendix.group(2) else "Dettagli"
            elif testo_heading_clean.upper() == "APPENDIX":
                candidate_sezione = "APPENDIX"
                candidate_titolo = "Introduzione"
            elif match_sezione:
                candidate_sezione = match_sezione.group(1).strip().rstrip('.')
                candidate_titolo = match_sezione.group(2).strip()
            elif match_cifra:
                candidate_sezione = match_cifra.group(1).strip()
                candidate_titolo = match_cifra.group(2).strip()
            elif match_solo_sezione:
                candidate_sezione = match_solo_sezione.group(1).strip().rstrip('.')
                candidate_titolo = "Introduzione"
            else:
                candidate_titolo = testo_heading_clean
                
            # Applica la protezione gerarchica per evitare downgrade da page headers
            if is_parent_section(candidate_sezione, sezione_corrente):
                # Se la sezione è un genitore o meno specifica, la ignoriamo del tutto come intestazione
                linee_accumulate.append(line)
            else:
                # Salviamo il blocco accumulato finora (se contiene testo significativo)
                testo_accumulato = "\n".join(linee_accumulate).strip()
                if testo_accumulato:
                    blocchi.append({
                        "document_id": document_id,
                        "file_name": nome_file,
                        "page": page_label,
                        "section": iper_pulizia_campo(sezione_corrente),
                        "title": iper_pulizia_campo(titolo_corrente),
                        "text": testo_accumulato
                    })
                    linee_accumulate = []
                
                # Applica il nuovo stato
                sezione_corrente = candidate_sezione
                titolo_corrente = candidate_titolo
        else:
            linee_accumulate.append(line)
            
    # Salva l'ultimo blocco residuo al termine della pagina
    testo_accumulato = "\n".join(linee_accumulate).strip()
    if testo_accumulato:
        blocchi.append({
            "document_id": document_id,
            "file_name": nome_file,
            "page": page_label,
            "section": iper_pulizia_campo(sezione_corrente),
            "title": iper_pulizia_campo(titolo_corrente),
            "text": testo_accumulato
        })
        
    # Aggiorna lo stato esterno per la pagina successiva
    stato_sezione["sezione"] = sezione_corrente
    stato_sezione["titolo"] = titolo_corrente
    
    return blocchi


@misura_performance(metodo="pdf4llm")
def esegui_estrazione_pdf4llm(percorso_pdf, percorso_csv):
    """
    Esegue l'estrazione del PDF tramite pymupdf4llm.
    Ottiene la struttura in Markdown pagina per pagina, escludendo immagini sul disco per efficienza.
    Mappa i blocchi nel formato JSON standard ad alta fedeltà.
    """
    nome_file = os.path.basename(percorso_pdf)
    document_id, pagina_iniziale_int, sezione_speciale = leggi_metadati_csv(percorso_csv, nome_file)
    
    print(f"📄 pymupdf4llm sta analizzando il layout di: {nome_file}...")
    
    # Esegue l'estrazione pagina per pagina (write_images=False per massimizzare le prestazioni)
    md_chunks = pymupdf4llm.to_markdown(percorso_pdf, page_chunks=True, write_images=False)
    
    risultato_json = []
    
    # Stato gerarchico persistente tra le pagine
    stato_sezione = {
        "sezione": "Generale",
        "titolo": "Introduzione"
    }
    
    for chunk in md_chunks:
        metadata = chunk.get("metadata", {})
        idx_pagina = 0
        if "page_number" in metadata:
            idx_pagina = metadata["page_number"] - 1
        elif "page" in chunk:
            idx_pagina = chunk["page"]
            
        markdown_text = chunk.get("text", "")
        
        page_label = calcola_nome_pagina(idx_pagina, pagina_iniziale_int, sezione_speciale)
        
        # Estrae i blocchi strutturati dividendo la pagina per ciascuna intestazione trovata
        blocchi_pagina = dividi_pagina_in_blocchi_strutturati(
            markdown_text=markdown_text,
            document_id=document_id,
            nome_file=nome_file,
            page_label=page_label,
            stato_sezione=stato_sezione
        )
        
        risultato_json.extend(blocchi_pagina)
        
    return risultato_json


# ==========================================
# EXECUTOR MASSIVO BATCH
# ==========================================
if __name__ == "__main__":
    if len(argv) < 2:
        print("❌ Uso: python loader_pdf4llm.py <nome_cartella_manuali | all>")
        sys.exit(1)
        
    cartella_target = argv[1]
    csv_metadati = os.path.join(PROJECT_ROOT, "data", "raw", "metadata", "document_index.csv")
    cartella_output = os.path.join(PROJECT_ROOT, "data", "processed", "pdf4llm")
    os.makedirs(cartella_output, exist_ok=True)
    
    lista_pdf = []
    
    if cartella_target.lower() == "all":
        print(f"📂 [PDF4LLM BATCH] Lettura di tutti i PDF definiti nell'indice documentale...")
        try:
            with open(csv_metadati, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    nome_file = row['nome_file']
                    tipo_doc = row['tipo_documento']
                    pdf_path = os.path.join(PROJECT_ROOT, "data", "raw", tipo_doc, nome_file)
                    if os.path.exists(pdf_path):
                        lista_pdf.append(pdf_path)
                    else:
                        print(f"⚠️ File PDF non trovato all'indirizzo previsto: {pdf_path}")
        except Exception as e:
            print(f"❌ Errore durante la lettura di document_index.csv: {e}")
            sys.exit(1)
    else:
        percorso_raw = os.path.join(PROJECT_ROOT, "data", "raw", cartella_target)
        lista_pdf = glob.glob(os.path.join(percorso_raw, "*.pdf"))
    
    print(f"🚀 [PDF4LLM] Trovati {len(lista_pdf)} PDF da analizzare.\n")
    
    for pdf_path in lista_pdf:
        nome_pdf = os.path.basename(pdf_path)
        print(f"--- 📄 Inizio Parsing Documentale: {nome_pdf} ---")
        
        records_estratti = esegui_estrazione_pdf4llm(pdf_path, csv_metadati)
        
        if records_estratti:
            nome_json = nome_pdf.replace(".pdf", ".json")
            percorso_salvataggio = os.path.join(cartella_output, nome_json)
            
            with open(percorso_salvataggio, "w", encoding="utf-8") as f:
                json.dump(records_estratti, f, indent=4, ensure_ascii=False)
                
            print(f"    ✓ Completato. Salvato in data/processed/pdf4llm/{nome_json} ({len(records_estratti)} blocchi).\n")
        else:
            print(f"    ⚠️ Nessun blocco estratto da {nome_pdf}\n")
            
    print("===================================================")
    print("🎉 BATCH PROCESSING CON PYMUPDF4LLM COMPLETATO! 🎉")
    print("===================================================")
