import os
import sys
import csv
import glob
import json
import re
import shutil
import subprocess
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
    cleaned = re.sub(r'^[#\s\-\*\_\`\:]+', '', text)
    cleaned = re.sub(r'[#\s\-\*\_\`\:]+$', '', cleaned)
    cleaned = cleaned.strip().upper()
    
    if not cleaned:
        return True
    
    if cleaned in ["NOTE", "WARNING", "CAUTION", "IMPORTANT", "NOTICE", "DANGER", 
                   "PRECAUTION", "PRECAUTIONS", "ATTENZIONE", "AVVERTENZA", "AVVISO", 
                   "NOTE IN CASE OF CE CONTROLLER"]:
        return True
        
    if cleaned in [
        "MAINTENANCE", "SAFETY PRECAUTIONS", "TROUBLESHOOTING", "OVERVIEW", 
        "OVERVIEW AND CONFIGURATION", "CONFIGURATION", "CHECKS AND MAINTENANCE", 
        "DIAGNOSTICS", "VISUAL DIAGNOSTICS", "PRINTED CIRCUIT BOARDS", "AMPLIFIERS", 
        "REPLACING UNITS", "CONNECTIONS", "CABLE CONNECTION", "CONNECTION DIAGRAM"
    ]:
        return True
    
    if re.search(r'B-\d{5}EN/\d+', cleaned):
        return True
        
    if re.match(r'^(FIG\.|FIGURE|TABLE|TAB\.)\s*\d+', cleaned):
        return True
        
    if re.match(r'^-\s*\d+\s*-$', cleaned) or re.match(r'^\d+$', cleaned):
        return True
        
    return False


def is_parent_section(new_sec, current_sec):
    """
    Ritorna True se la nuova sezione candidata è un genitore o meno specifica 
    rispetto alla sezione corrente attiva.
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
    Identico alla logica degli altri loader per coerenza scientifica.
    """
    blocchi = []
    linee_accumulate = []
    
    sezione_corrente = stato_sezione["sezione"]
    titolo_corrente = stato_sezione["titolo"]
    
    for line in markdown_text.split("\n"):
        line_strip = line.strip()
        if not line_strip:
            continue
            
        match_heading = re.match(r"^(#{1,6})\s+(.*)", line_strip)
        is_implicit_heading = False
        implicit_text = ""
        
        if not match_heading and len(line_strip) < 80 and not line_strip.startswith(("-", "*", "|", "(", "[", "{", "•")):
            if not line_strip.endswith((".", "?", "!", ";", ",")):
                match_appendix = re.match(r"^(APPENDIX\s+[A-Z0-9])", line_strip, re.IGNORECASE)
                match_multilevel = re.match(r"^([A-Z0-9]\.[\d\.]+)\s+", line_strip, re.IGNORECASE)
                match_single = re.match(r"^([A-Z0-9]\.?)\s+(.*)", line_strip, re.IGNORECASE)
                
                is_valid = False
                if match_appendix:
                    is_valid = True
                elif match_multilevel:
                    is_valid = True
                elif match_single:
                    num_or_letter = match_single.group(1).rstrip('.')
                    rest_of_title = match_single.group(2).strip()
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
            
        if testo_heading and is_callout_or_header_to_ignore(testo_heading):
            testo_heading = ""
            
        if testo_heading:
            testo_heading_clean = re.sub(r'[*_`]', '', testo_heading).strip()
            
            match_appendix = re.match(r"^(APPENDIX\s+[A-Z0-9](?:\.[\d\.]*)?)(?:\s+(.*))?$", testo_heading_clean, re.IGNORECASE)
            match_sezione = re.match(r"^([A-Z0-9]\.[\d\.]*)\s+(.*)", testo_heading_clean, re.IGNORECASE)
            match_cifra = re.match(r"^(\d+)\s+(.*)", testo_heading_clean)
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
                
            if is_parent_section(candidate_sezione, sezione_corrente):
                linee_accumulate.append(line)
            else:
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
                
                sezione_corrente = candidate_sezione
                titolo_corrente = candidate_titolo
        else:
            linee_accumulate.append(line)
            
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
        
    stato_sezione["sezione"] = sezione_corrente
    stato_sezione["titolo"] = titolo_corrente
    
    return blocchi


from html.parser import HTMLParser

class HTMLTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.current_row = []
        self.current_cell = ""
        self.in_cell = False
        self.cell_attrs = {}

    def handle_starttag(self, tag, attrs):
        if tag in ["td", "th"]:
            self.in_cell = True
            self.current_cell = ""
            self.cell_attrs = dict(attrs)
        elif tag == "tr":
            self.current_row = []

    def handle_endtag(self, tag):
        if tag in ["td", "th"]:
            self.in_cell = False
            colspan = int(self.cell_attrs.get("colspan", 1))
            rowspan = int(self.cell_attrs.get("rowspan", 1))
            self.current_row.append({
                "text": self.current_cell.strip().replace("\n", " "),
                "colspan": colspan,
                "rowspan": rowspan
            })
        elif tag == "tr":
            self.rows.append(self.current_row)

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data

def converti_html_tabella_a_markdown(html_content):
    parser = HTMLTableParser()
    try:
        parser.feed(html_content)
    except Exception as e:
        print(f"⚠️ Errore di parsing HTML della tabella: {e}")
        return ""
    
    if not parser.rows:
        return ""
    
    # Costruisci una griglia bidimensionale per gestire colspan e rowspan
    max_cols = 0
    for r in parser.rows:
        cols_in_row = sum(cell["colspan"] for cell in r)
        if cols_in_row > max_cols:
            max_cols = cols_in_row
            
    if max_cols == 0:
        return ""
        
    grid = []
    for r_idx in range(len(parser.rows)):
        grid.append([""] * max_cols)
        
    for r_idx, row in enumerate(parser.rows):
        c_idx = 0
        for cell in row:
            while c_idx < max_cols and grid[r_idx][c_idx] != "":
                c_idx += 1
                
            if c_idx >= max_cols:
                break
                
            text = cell["text"]
            colspan = cell["colspan"]
            rowspan = cell["rowspan"]
            
            for dr in range(rowspan):
                for dc in range(colspan):
                    target_r = r_idx + dr
                    target_c = c_idx + dc
                    if target_r < len(grid) and target_c < max_cols:
                        grid[target_r][target_c] = text
            c_idx += colspan

    md_lines = []
    if len(grid) > 0:
        header_row = grid[0]
        md_lines.append("| " + " | ".join(header_row) + " |")
        md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        for row in grid[1:]:
            md_lines.append("| " + " | ".join(row) + " |")
        
    return "\n".join(md_lines) + "\n"


@misura_performance(metodo="mineru")
def esegui_estrazione_mineru(percorso_pdf, percorso_csv):
    """
    Esegue l'estrazione del PDF usando il parser SOTA MinerU.
    Invocato tramite CLI magic-pdf per garantire isolamento e stabilità su Windows.
    Reinizializza e mappa i blocchi nel formato JSON standard.
    """
    nome_file = os.path.basename(percorso_pdf)
    nome_pdf_senza_est = os.path.splitext(nome_file)[0]
    document_id, pagina_iniziale_int, sezione_speciale = leggi_metadati_csv(percorso_csv, nome_file)
    
    # Cartella temporanea di output per questo specifico file
    cartella_temp = os.path.join(PROJECT_ROOT, "data", "processed", "mineru", "temp_extract")
    os.makedirs(cartella_temp, exist_ok=True)
    
    print(f"🤖 MinerU sta analizzando il layout di: {nome_file}...")
    
    # Comando CLI per eseguire magic-pdf via Python
    comando = [
        sys.executable,
        "-m",
        "magic_pdf.tools.cli",
        "-p", percorso_pdf,
        "-o", cartella_temp,
        "-m", "auto"
    ]
    
    try:
        # Esegui magic-pdf CLI
        risultato_cli = subprocess.run(
            comando,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            check=True
        )
        print(f"   ✓ CLI MinerU completata con successo.")
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione di MinerU CLI: {e}")
        # Pulizia cartella temporanea ed eccezione
        if os.path.exists(cartella_temp):
            shutil.rmtree(cartella_temp)
        return []

    # Individua il file di output json generato (content_list)
    # MinerU crea una cartella col nome del pdf all'interno di cartella_temp, 
    # e sotto di essa una sottocartella col nome del metodo (es: auto, txt, ocr).
    # Cerchiamo quindi ricorsivamente.
    cartella_pdf_estratto = os.path.join(cartella_temp, nome_pdf_senza_est)
    
    pattern_ricerca = os.path.join(cartella_pdf_estratto, "**", f"{nome_pdf_senza_est}_content_list.json")
    candidati = glob.glob(pattern_ricerca, recursive=True)
    
    file_content_list = None
    if candidati:
        file_content_list = candidati[0]
    else:
        # Fallback alternativo: cerca qualsiasi JSON nella cartella estratta
        pattern_generico = os.path.join(cartella_pdf_estratto, "**", "*.json")
        candidati_json = glob.glob(pattern_generico, recursive=True)
        if candidati_json:
            file_content_list = candidati_json[0]
            print(f"   ℹ️ Trovato JSON alternativo: {file_content_list}")
            
    if not file_content_list or not os.path.exists(file_content_list):
        print(f"❌ Errore: Impossibile trovare il file dei contenuti JSON ricorsivamente sotto {cartella_pdf_estratto}")
        if os.path.exists(cartella_temp):
            shutil.rmtree(cartella_temp)
        return []

    # Carica la lista di contenuti estratti
    try:
        with open(file_content_list, "r", encoding="utf-8") as f:
            content_data = json.load(f)
    except Exception as e:
        print(f"❌ Errore di lettura del JSON generato da MinerU: {e}")
        if os.path.exists(cartella_temp):
            shutil.rmtree(cartella_temp)
        return []

    # Raggruppa i blocchi di testo per pagina (page_idx)
    pagine_markdown = {}
    for block in content_data:
        block_type = block.get("type", "text")
        page_idx = block.get("page_idx")
        # In rari casi MinerU potrebbe usare 'page'
        if page_idx is None:
            page_idx = block.get("page", 0)
            
        if page_idx not in pagine_markdown:
            pagine_markdown[page_idx] = []
            
        if block_type == "table":
            caption_list = block.get("table_caption", [])
            caption = " ".join(caption_list).strip() if isinstance(caption_list, list) else str(caption_list).strip()
            
            table_body = block.get("table_body", "")
            markdown_table = ""
            if table_body:
                markdown_table = converti_html_tabella_a_markdown(table_body)
            
            footnote_list = block.get("table_footnote", [])
            footnote = " ".join(footnote_list).strip() if isinstance(footnote_list, list) else str(footnote_list).strip()
            
            table_text = ""
            if caption:
                table_text += f"**{caption}**\n\n"
            if markdown_table:
                table_text += markdown_table
            else:
                fallback_md = block.get("markdown", "")
                if fallback_md:
                    table_text += fallback_md
            if footnote:
                table_text += f"\n\n*Footnote: {footnote}*"
                    
            if table_text.strip():
                pagine_markdown[page_idx].append(table_text.strip())
        else:
            text_block = block.get("text", "").strip()
            if text_block:
                pagine_markdown[page_idx].append(text_block)

    risultato_json = []
    stato_sezione = {
        "sezione": "Generale",
        "titolo": "Introduzione"
    }

    # Ordina e processa le pagine per mantenere l'ordine corretto del PDF
    for idx_pagina in sorted(pagine_markdown.keys()):
        markdown_text = "\n\n".join(pagine_markdown[idx_pagina])
        page_label = calcola_nome_pagina(idx_pagina, pagina_iniziale_int, sezione_speciale)
        
        # Suddividi in blocchi strutturati
        blocchi_pagina = dividi_pagina_in_blocchi_strutturati(
            markdown_text=markdown_text,
            document_id=document_id,
            nome_file=nome_file,
            page_label=page_label,
            stato_sezione=stato_sezione
        )
        risultato_json.extend(blocchi_pagina)

    # Pulizia finale della directory temporanea
    try:
        shutil.rmtree(cartella_temp)
    except Exception as e:
        print(f"⚠️ Non è stato possibile rimuovere la cartella temporanea: {e}")
        
    return risultato_json


# ==========================================
# EXECUTOR MASSIVO BATCH
# ==========================================
if __name__ == "__main__":
    if len(argv) < 2:
        print("❌ Uso: python loader_mineru.py <nome_cartella_manuali | all>")
        sys.exit(1)
        
    cartella_target = argv[1]
    csv_metadati = os.path.join(PROJECT_ROOT, "data", "raw", "metadata", "document_index.csv")
    cartella_output = os.path.join(PROJECT_ROOT, "data", "processed", "mineru")
    os.makedirs(cartella_output, exist_ok=True)
    
    lista_pdf = []
    
    if cartella_target.lower() == "all":
        print(f"📂 [MINERU BATCH] Lettura di tutti i PDF definiti nell'indice documentale...")
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
    
    print(f"🚀 [MINERU] Trovati {len(lista_pdf)} PDF da analizzare.\n")
    
    for pdf_path in lista_pdf:
        nome_pdf = os.path.basename(pdf_path)
        print(f"--- 📄 Inizio Parsing Documentale con MinerU: {nome_pdf} ---")
        
        records_estratti = esegui_estrazione_mineru(pdf_path, csv_metadati)
        
        if records_estratti:
            nome_json = nome_pdf.replace(".pdf", ".json")
            percorso_salvataggio = os.path.join(cartella_output, nome_json)
            
            with open(percorso_salvataggio, "w", encoding="utf-8") as f:
                json.dump(records_estratti, f, indent=4, ensure_ascii=False)
                
            print(f"    ✓ Completato. Salvato in data/processed/mineru/{nome_json} ({len(records_estratti)} blocchi).\n")
        else:
            print(f"    ⚠️ Nessun blocco estratto da {nome_pdf}\n")
            
    print("===================================================")
    print("🎉 BATCH PROCESSING CON MINERU COMPLETATO! 🎉")
    print("===================================================")
