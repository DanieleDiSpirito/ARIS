import csv
import glob
import json
import os
import re
import statistics
import pdfplumber
from sys import argv

def estrai_elementi_pagina(pagina_pdf, y_min=75, y_max=780):
    """
    MODULO 1 POTENZIATO: Estrattore Unificato (Testo + Tabelle)
    Estrae le tabelle in formato Markdown, usa uno "scudo spaziale" per non 
    leggere le parole dentro le tabelle due volte, e allinea testo e griglie 
    dall'alto verso il basso.
    """
    elementi_pagina = []
    
    # ==========================================
    # 1. IL RADAR TABELLE
    # ==========================================
    tabelle_trovate = pagina_pdf.find_tables()
    aree_tabelle = []
    
    for tabella in tabelle_trovate:
        bbox = tabella.bbox # Coordinate: (x0, top, x1, bottom)
        aree_tabelle.append(bbox)
        
        dati_tabella = tabella.extract()
        if not dati_tabella:
            continue
            
        # Costruiamo la tabella in formato Markdown per l'LLM
        md_lines = []
        for i, riga in enumerate(dati_tabella):
            # Pulizia celle: rimpiazza gli a capo interni con spazi
            riga_pulita = [str(cella).replace('\n', ' ').strip() if cella else "" for cella in riga]
            md_lines.append("| " + " | ".join(riga_pulita) + " |")
            
            # Subito dopo l'intestazione, mettiamo la riga separatrice del Markdown
            if i == 0: 
                md_lines.append("|" + "|".join(["---"] * len(riga_pulita)) + "|")
                
        testo_markdown = "\n\n" + "\n".join(md_lines) + "\n\n"
        
        # Salviamo la tabella come "finto" elemento di testo per ingannare il Classificatore
        elementi_pagina.append({
            "testo": testo_markdown,
            "y_top": round(bbox[1]),
            "size_max": 11.0,
            "size_moda": 11.0,
            "font_moda": "TabellaMarkdown" # Nome font fittizio come "Trigger"
        })

    # ==========================================
    # 2. ESTRAZIONE PAROLE (CON SCUDO)
    # ==========================================
    parole_grezze = pagina_pdf.extract_words(extra_attrs=["fontname", "size"])
    righe_grezze = {}
    
    for parola in parole_grezze:
        x0, top, x1, bottom = parola['x0'], parola['top'], parola['x1'], parola['bottom']
        
        # La Ghigliottina Spaziale (Salta header e footer)
        if top < y_min or top > y_max:
            continue
            
        # Lo Scudo Anti-Doppioni (Salta le parole che si trovano dentro una tabella)
        dentro_tabella = False
        for bx0, btop, bx1, bbottom in aree_tabelle:
            # Controllo geometrico di intersezione
            if x1 > bx0 and x0 < bx1 and bottom > btop and top < bbottom:
                dentro_tabella = True
                break
                
        if dentro_tabella:
            continue
            
        # Raggruppamento delle parole rimanenti per riga (altezza)
        y_top = round(top)
        riga_trovata = False
        for y in righe_grezze.keys():
            if abs(y - y_top) <= 3:
                righe_grezze[y].append(parola)
                riga_trovata = True
                break

        if not riga_trovata:
            righe_grezze[y_top] = [parola]

    # ==========================================
    # 3. ORDINAMENTO E CALCOLO RIGHE DI TESTO
    # ==========================================
    for y_top in sorted(righe_grezze.keys()):
        parole_ordinate = sorted(righe_grezze[y_top], key=lambda p: p['x0'])
        testo_riga = " ".join([p['text'] for p in parole_ordinate]).strip()
        
        if not testo_riga:
            continue

        dimensioni = [p['size'] for p in parole_ordinate]
        font_names = [p['fontname'] for p in parole_ordinate]

        elementi_pagina.append({
            "testo": testo_riga,
            "y_top": y_top,
            "size_max": round(max(dimensioni), 1),
            "size_moda": round(statistics.mode(dimensioni), 1),
            "font_moda": statistics.mode(font_names)
        })

    # ==========================================
    # 4. ALLINEAMENTO FINALE Y-TOP
    # ==========================================
    # Mescoliamo tabelle e righe testuali nell'ordine esatto in cui appaiono dall'alto al basso
    elementi_pagina.sort(key=lambda e: e['y_top'])
    
    return elementi_pagina


def classifica_riga(testo_riga, size_max, size_moda, font_moda):
    """
    MODULO 2: Il Classificatore (Cervello)
    Analizza i metadati di una riga e restituisce un'etichetta (Tag)
    basata sulla tabella logica (Guida font_3.xlsx).
    """
    font_lower = font_moda.lower()
    is_bold = "bold" in font_lower
    
    # Intercettiamo immediatamente le nostre Tabelle Markdown
    if font_lower == "tabellamarkdown":
        return "TESTO_TABELLA"
    
    # 1. TRIGGER GERARCHICI
    if is_bold and size_max >= 19.5:
        return "TRIGGER_CAPITOLO"
    elif is_bold and 14.5 <= size_max <= 22.5:
        return "TRIGGER_PARAGRAFO"
        
    # 2. TRIGGER SPECIALI (Codici Errore)
    elif is_bold and 11.5 <= size_max <= 12.5 and re.match(r"^(SRVO|SYST|PRIO|MACR)-\d+", testo_riga):
        return "TRIGGER_ERRORE"
        
    # 3. TRATTAMENTO ECCEZIONI E TESTO NORMALE
    elif "mincho" in font_lower:
        return "TESTO_PUNTATO"
    elif is_bold and size_max <= 9.5:
        return "TESTO_Caption"
    elif is_bold and testo_riga.isupper() and testo_riga.startswith(("WARNING", "CAUTION", "NOTE")):
        return "TESTO_WARNING"
    else:
        return "TESTO_NORMALE"


def crea_json_definitivo(percorso_pdf, percorso_csv):
    """
    MODULO 3: La Macchina a Stati e Paginatore (Orchestratore)
    Legge il CSV per i metadati, usa il parser basato sui font, e salva 
    un record ogni volta che si cambia pagina (o capitolo).
    """
    nome_file = os.path.basename(percorso_pdf)
    
    # --- 1. LETTURA METADATI DAL CSV ---
    document_id = "00"
    pagina_iniziale_int = 0
    sezione_speciale = False

    try:
        with open(percorso_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['nome_file'] == nome_file:
                    document_id = row['id_documento']
                    pag_str = row['pagina_manuale']
                    
                    if pag_str.startswith('s'):
                        sezione_speciale = True
                        pagina_iniziale_int = 1
                    else:
                        pagina_iniziale_int = int(pag_str)
                    break
    except FileNotFoundError:
        print(f"ATTENZIONE: File CSV {percorso_csv} non trovato.")
        return []

    # --- 2. VARIABILI DI STATO ---
    risultato_json = []
    
    stato_corrente = {
        "section": "Generale",
        "title": "Introduzione",
        "text_buffer": []
    }
    
    def calcola_nome_pagina(indice_pagina_pdf):
        """Calcola la stringa della pagina corretta ('45' o 's-3')"""
        if sezione_speciale:
            return f"s-{pagina_iniziale_int + indice_pagina_pdf}"
        else:
            return str(pagina_iniziale_int + indice_pagina_pdf)

    def salva_blocco_corrente(indice_pagina_pdf):
        """Salva il buffer nel JSON e lo svuota."""
        testo_unito = " ".join(stato_corrente["text_buffer"]).strip()
        if testo_unito:
            risultato_json.append({
                "document_id": document_id,
                "file_name": nome_file,
                "page": calcola_nome_pagina(indice_pagina_pdf),
                "section": stato_corrente["section"],
                "title": stato_corrente["title"],
                "text": testo_unito
            })
        stato_corrente["text_buffer"] = []

    # --- 3. SCANSIONE DEL PDF ---
    with pdfplumber.open(percorso_pdf) as pdf:
        for indice_pagina, pagina in enumerate(pdf.pages):
            
            if stato_corrente["text_buffer"] and indice_pagina > 0:
                salva_blocco_corrente(indice_pagina - 1)

            # Estrazione logica dal Modulo 1 (Ora si chiama estrai_elementi_pagina)
            elementi = estrai_elementi_pagina(pagina)
            
            for elem in elementi:
                testo = elem["testo"]
                
                # Classificazione dal Modulo 2
                tag = classifica_riga(testo, elem["size_max"], elem["size_moda"], elem["font_moda"])
                
                # --- Smistamento Logico ---
                if tag in ["TRIGGER_CAPITOLO", "TRIGGER_PARAGRAFO"]:
                    salva_blocco_corrente(indice_pagina)
                    
                    if re.match(r"^[\d\.]+$", testo):
                        stato_corrente["section"] = testo
                    elif re.match(r"^([\d\.]+)\s+(.*)", testo):
                        match = re.match(r"^([\d\.]+)\s+(.*)", testo)
                        stato_corrente["section"] = match.group(1).strip()
                        stato_corrente["title"] = match.group(2).strip()
                    else:
                        stato_corrente["title"] = testo
                        
                elif tag == "TRIGGER_ERRORE":
                    salva_blocco_corrente(indice_pagina)
                    stato_corrente["title"] = testo
                    
                elif tag == "TESTO_TABELLA":
                    # Mettiamo le tabelle esattamente dove cadono
                    stato_corrente["text_buffer"].append(testo)
                    
                elif tag == "TESTO_PUNTATO":
                    testo_pulito = re.sub(r"^[^\w\s]\s*", "- ", testo)
                    stato_corrente["text_buffer"].append(f"\n{testo_pulito}")
                    
                elif tag == "TESTO_WARNING":
                    stato_corrente["text_buffer"].append(f"\n**{testo}**")
                    
                elif tag == "TESTO_Caption":
                    stato_corrente["text_buffer"].append(f"\n[Caption]: {testo}")
                    
                else: 
                    stato_corrente["text_buffer"].append(testo)
                    
    # --- 4. FINE DOCUMENTO ---
    salva_blocco_corrente(len(pdf.pages) - 1)
    
    return risultato_json


# ==========================================
# ESECUZIONE MASSIVA (UN JSON PER OGNI PDF)
# ==========================================
if __name__ == "__main__":
    
    cartella_dati = "../../data/raw/" + argv[1]
    file_csv = "../../data/raw/metadata/document_index.csv"
    
    pattern_ricerca = os.path.join(cartella_dati, "*.pdf")
    lista_pdf = glob.glob(pattern_ricerca)
    
    print(f"🚀 Trovati {len(lista_pdf)} file PDF. Inizio estrazione massiva con Analisi Tabelle...\n")
    
    for percorso_pdf in lista_pdf:
        nome_file = os.path.basename(percorso_pdf)
        print(f"--- 📄 Elaborazione: {nome_file} ---")
        
        dati_pdf = crea_json_definitivo(percorso_pdf, file_csv)
        
        if dati_pdf:
            nome_file_output = nome_file.replace(".pdf", ".json")
            
            with open("../../data/processed/" + nome_file_output, "w", encoding="utf-8") as f:
                json.dump(dati_pdf, f, indent=4, ensure_ascii=False)
                
            print(f"    ✓ Finito. Creato '{nome_file_output}' con {len(dati_pdf)} record.\n")
        else:
            print(f"    ⚠️ Nessun dato estratto da {nome_file} o file saltato.\n")
            
    print("===================================================")
    print("🎉 PROCESSO BATCH COMPLETATO CON SUCCESSO! 🎉")
    print("===================================================")