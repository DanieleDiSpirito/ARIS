import fitz  # PyMuPDF
import pdfplumber
import json
import re
import csv
import os

def estrai_testo_manuale():
    percorso_pdf = "../../data/raw/manuali_manutenzione/safety_precautions.pdf"
    percorso_output = "../../data/processed/safety_precautions.json"
    
    nome_file = percorso_pdf.split("/")[-1]
    document_id = "00"
    pagina_manuale = ""
    sezione_pagina = None
    pagina_manuale_int = 0

    # --- LETTURA METADATI DAL CSV ---
    try:
        with open('../../data/raw/metadata/document_index.csv', 'r', encoding='utf-8') as csvfile:
            reader = list(csv.DictReader(csvfile))
            for row in reader:
                if row['nome_file'] == nome_file:
                    document_id = row['id_documento']
                    pagina_manuale = row['pagina_manuale']
                    if pagina_manuale.startswith('s'):
                        pagina_manuale_int = 1
                        sezione_pagina = 's'
                    else:
                        pagina_manuale_int = int(pagina_manuale)
                    break
    except FileNotFoundError:
        print("ATTENZIONE: File CSV non trovato. Assicurati che il percorso sia corretto.")
        return

    assert document_id != "00", f"Document ID per {nome_file} non trovato nel CSV"

    print(f"Apertura del file {percorso_pdf} in corso...")
    
    dati_estratti = []
    sezione_corrente = "Unknown"
    titolo_corrente = "No Title"
    pattern_titolo = re.compile(r"^([\d\.]+)\s+(.+)")

    try:
        # =====================================================================
        # FASE 1: PDFPLUMBER (Tabelle strutturate e Bounding Box)
        # =====================================================================
        print("Ricerca tabelle con pdfplumber...")
        tabelle_formattate = {} 

        with pdfplumber.open(percorso_pdf) as pdf:
            for num_pagina, pagina_plumber in enumerate(pdf.pages):
                tabelle_trovate = pagina_plumber.find_tables()
                tabelle_formattate[num_pagina] = []

                for tabella in tabelle_trovate:
                    dati_tabella = tabella.extract()
                    
                    if not dati_tabella or len(dati_tabella) < 2:
                        continue # Salta tabelle vuote o finte

                    intestazioni = dati_tabella[0]
                    intestazioni_pulite = [col.replace('\n', ' ') if col else "Dato" for col in intestazioni]

                    testo_strutturato = []
                    for riga in dati_tabella[1:]:
                        chiave_principale = riga[0]
                        if not chiave_principale or chiave_principale.strip() == "":
                            continue
                        chiave_principale = chiave_principale.replace('\n', ' ')

                        frase_dati = []
                        for i in range(1, len(riga)):
                            valore = riga[i]
                            if valore and valore.strip() != "":
                                val_pulito = valore.replace('\n', ' ')
                                frase_dati.append(f"{intestazioni_pulite[i]}: {val_pulito}")

                        testo_strutturato.append(f"Specifications for {chiave_principale} -> " + " | ".join(frase_dati))

                    testo_unito = " ".join(testo_strutturato)
                    
                    # Salva la tabella solo se c'è del testo vero dentro
                    if testo_unito.strip():
                        tabelle_formattate[num_pagina].append({
                            "bbox": tabella.bbox,
                            "testo": testo_unito
                        })

        # =====================================================================
        # FASE 2: PYMUPDF (Testo base, Ereditarietà e Filtro Rumore)
        # =====================================================================
        print("Estrazione testo con PyMuPDF...")
        with fitz.open(percorso_pdf) as doc:
            for num_pagina in range(len(doc)):
                pagina_fitz = doc[num_pagina]
                larghezza = pagina_fitz.rect.width
                altezza = pagina_fitz.rect.height
                
                area_utile = fitz.Rect(0, 80, larghezza, altezza - 60)
                blocchi = pagina_fitz.get_text("blocks", clip=area_utile)
                
                tabelle_inserite = set()
                tabelle_della_pagina = tabelle_formattate.get(num_pagina, [])

                for blocco in blocchi:
                    rect_blocco = fitz.Rect(blocco[:4])
                    sovrapposto = False
                    indice_tab = None
                    
                    # 1. CONTROLLO SOVRAPPOSIZIONE CON TABELLA
                    for idx, tab_dict in enumerate(tabelle_della_pagina):
                        rect_tabella = fitz.Rect(tab_dict["bbox"])
                        if rect_blocco.intersects(rect_tabella):
                            sovrapposto = True
                            indice_tab = idx
                            break
                            
                    if sovrapposto:
                        rect_tabella = fitz.Rect(tabelle_della_pagina[indice_tab]["bbox"])

                        # Salva la parte di testo CHE STA SOPRA la tabella (se esiste)
                        if rect_blocco.y0 < rect_tabella.y0 - 5:
                            area_sopra = fitz.Rect(rect_blocco.x0, rect_blocco.y0, rect_blocco.x1, rect_tabella.y0)
                            testo_sopra = pagina_fitz.get_text("text", clip=area_sopra).strip()
                            testo_sopra = testo_sopra.replace('\n', ' ')
                            testo_sopra = re.sub(r'\s+', ' ', testo_sopra).strip()
                            # Rimuovi didascalie tabella tipo "Table 1. Applied standards"
                            testo_sopra = re.sub(r'\s*Table\s+\d+\..*$', '', testo_sopra).strip()
                            if testo_sopra and len(testo_sopra.split()) >= 8:
                                record_sopra = {
                                    "document_id": document_id,
                                    "file_name": '/'.join(percorso_pdf.split("/")[-2:]),
                                    "page": str(pagina_manuale_int + num_pagina) if sezione_pagina != 's' else f's-{num_pagina + 1}',
                                    "section": sezione_corrente,
                                    "title": titolo_corrente,
                                    "text": testo_sopra
                                }
                                dati_estratti.append(record_sopra)

                        # Salva la parte di testo CHE STA SOTTO la tabella (se esiste)
                        if rect_blocco.y1 > rect_tabella.y1 + 5:
                            area_sotto = fitz.Rect(rect_blocco.x0, rect_tabella.y1, rect_blocco.x1, rect_blocco.y1)
                            testo_sotto = pagina_fitz.get_text("text", clip=area_sotto).strip()
                            testo_sotto = testo_sotto.replace('\n', ' ')
                            testo_sotto = re.sub(r'\s+', ' ', testo_sotto).strip()
                            if testo_sotto and len(testo_sotto.split()) >= 8:
                                record_sotto = {
                                    "document_id": document_id,
                                    "file_name": '/'.join(percorso_pdf.split("/")[-2:]),
                                    "page": str(pagina_manuale_int + num_pagina) if sezione_pagina != 's' else f's-{num_pagina + 1}',
                                    "section": sezione_corrente,
                                    "title": titolo_corrente,
                                    "text": testo_sotto
                                }
                                dati_estratti.append(record_sotto)

                        # Inserisci la tabella ora, così eredita il titolo corretto letto finora!
                        if indice_tab not in tabelle_inserite:
                            record_tab = {
                                "document_id": document_id,
                                "file_name": '/'.join(percorso_pdf.split("/")[-2:]),
                                "page": str(pagina_manuale_int + num_pagina) if sezione_pagina != 's' else f's-{num_pagina + 1}',
                                "section": sezione_corrente,
                                "title": titolo_corrente, 
                                "text": tabelle_della_pagina[indice_tab]["testo"]
                            }
                            dati_estratti.append(record_tab)
                            tabelle_inserite.add(indice_tab)
                        continue

                    # 2. SE NON È TABELLA, ESTRAI E PULISCI IL TESTO
                    testo_blocco = blocco[4].strip()
                    if not testo_blocco:
                        continue
                        
                    testo_pulito = testo_blocco.replace('\n', ' ') 
                    testo_pulito = re.sub(r'\s+', ' ', testo_pulito).strip()

                    # 3. CONTROLLO TITOLI E SEZIONI (prima del filtro rumore!)
                    match = pattern_titolo.search(testo_pulito)
                    if match:
                        sezione_corrente = match.group(1)
                        titolo_corrente = match.group(2)
                        continue 

                    # 4. PULIZIA PREFISSI SPURI PRIMA DI "Fig."
                    # PyMuPDF a volte unisce etichette del diagramma con la didascalia
                    fig_match = re.search(r'\bFig\.', testo_pulito)
                    if fig_match and fig_match.start() > 0:
                        prefisso = testo_pulito[:fig_match.start()].strip()
                        if len(prefisso.split()) < 8:
                            testo_pulito = testo_pulito[fig_match.start():]

                    # 5. FILTRO ANTIRUMORE PER IMMAGINI (Salva Captions, butta Etichette)
                    numero_parole = len(testo_pulito.split())
                    if numero_parole < 8 and not testo_pulito.lower().startswith(("fig", "table", "note", "warning")):
                        continue

                    # 6. CREAZIONE RECORD STANDARD
                    record = {
                        "document_id": document_id,
                        "file_name": '/'.join(percorso_pdf.split("/")[-2:]),
                        "page": str(pagina_manuale_int + num_pagina) if sezione_pagina != 's' else f's-{num_pagina + 1}',
                        "section": sezione_corrente,
                        "title": titolo_corrente,
                        "text": testo_pulito
                    }
                    dati_estratti.append(record)
                
                # RETE DI SICUREZZA: Inserisci tabelle rimaste indietro
                for idx, tab_dict in enumerate(tabelle_della_pagina):
                    if idx not in tabelle_inserite:
                        record_tab = {
                            "document_id": document_id,
                            "file_name": '/'.join(percorso_pdf.split("/")[-2:]),
                            "page": str(pagina_manuale_int + num_pagina) if sezione_pagina != 's' else f's-{num_pagina + 1}',
                            "section": sezione_corrente,
                            "title": titolo_corrente, 
                            "text": tab_dict["testo"]
                        }
                        dati_estratti.append(record_tab)

        # =====================================================================
        # FASE 3: SALVATAGGIO SU FILE
        # =====================================================================
        with open(percorso_output, "w", encoding="utf-8") as file_json:
            json.dump(dati_estratti, file_json, indent=4, ensure_ascii=False)
            
        print(f"Estratto {len(dati_estratti)} blocchi strutturati nel file: {percorso_output}")

    except Exception as e:
        print(f"Ops! Si è verificato un errore: {e}")

if __name__ == "__main__":
    estrai_testo_manuale()
