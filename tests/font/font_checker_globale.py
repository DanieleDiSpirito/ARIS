import pdfplumber
import statistics

def verifica_font_intero_pdf(pdf_path, target_font=None, target_size=None, tolleranza_size=0.2):
    """
    Estrae le righe da TUTTE le pagine del PDF e le filtra in base al font e/o alla dimensione forniti.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"\nInizio analisi completa del file: '{pdf_path}'")
            print(f"Pagine totali da scansionare: {len(pdf.pages)}")
            if target_font: print(f"Filtro Font applicato: contiene '{target_font}'")
            if target_size: print(f"Filtro Size applicato: {target_size}pt (±{tolleranza_size})")
            print("=" * 80)
            
            totale_match_documento = 0
            
            for numero_pagina, pagina in enumerate(pdf.pages, start=1):
                parole = pagina.extract_words(extra_attrs=["fontname", "size"])
                if not parole:
                    continue # Salta le pagine vuote
                    
                # Raggruppamento in righe basato sull'asse Y (tolleranza 3 pixel)
                righe = {}
                for w in parole:
                    y = round(w['top'])
                    trovata = False
                    for existing_y in righe.keys():
                        if abs(y - existing_y) <= 3:
                            righe[existing_y].append(w)
                            trovata = True
                            break
                    if not trovata:
                        righe[y] = [w]
                
                match_count_pagina = 0
                risultati_pagina = []
                
                # Analizziamo le righe dall'alto verso il basso
                for y in sorted(righe.keys()):
                    riga_words = sorted(righe[y], key=lambda x: x['x0'])
                    testo = " ".join([w['text'] for w in riga_words]).strip()
                    
                    if not testo:
                        continue
                    
                    # Calcolo della MODA per font e size
                    dimensioni = [w['size'] for w in riga_words]
                    font_names = [w['fontname'] for w in riga_words]
                    
                    size_moda = round(statistics.mode(dimensioni), 1)
                    font_moda = statistics.mode(font_names)
                    
                    # Logica di filtraggio
                    match_font = True
                    match_size = True
                    
                    if target_font and target_font.lower() not in font_moda.lower():
                        match_font = False
                        
                    if target_size is not None and abs(size_moda - target_size) > tolleranza_size:
                        match_size = False
                        
                    # Se la riga supera i filtri impostati, salvala!
                    if match_font and match_size:
                        match_count_pagina += 1
                        totale_match_documento += 1
                        risultati_pagina.append(f"Pag: {numero_pagina:<3} | Y: {y:<4} | Size: {size_moda:<5} | Font: {font_moda:<25} | Testo: {testo}")
                
                # Stampa i risultati SOLO se ha trovato qualcosa in questa specifica pagina
                if match_count_pagina > 0:
                    print(f"\n--- Trovati {match_count_pagina} match a Pagina {numero_pagina} ---")
                    for res in risultati_pagina:
                        print(res)

            print("\n" + "=" * 80)
            print(f"ANALISI COMPLETATA. Totale righe trovate in tutto il documento: {totale_match_documento}")
            
    except FileNotFoundError:
        print(f"Errore: Il file '{pdf_path}' non è stato trovato. Controlla il percorso.")

# ==========================================
# SEZIONE DI INPUT
# ==========================================
if __name__ == "__main__":
    
    FILE_PDF = "../../data/raw/codici_errore/troubleshooting_alarms.pdf"
    
    # Imposta i filtri: cerchiamo TUTTO ciò che è a 9.0pt, a prescindere dal font
    FONT_DA_CERCARE = "Arial-BoldMT"   
    SIZE_DA_CERCARE = 12
       
    
    verifica_font_intero_pdf(FILE_PDF, target_font=FONT_DA_CERCARE, target_size=SIZE_DA_CERCARE)