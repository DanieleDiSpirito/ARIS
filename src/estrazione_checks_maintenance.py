import fitz  # La libreria PyMuPDF
import json
import re

def estrai_testo_manuale():
    percorso_pdf = "checks_maintenance.pdf"
    percorso_output = "testo_estratto.json"
    
    print(f"Apertura del file {percorso_pdf} in corso...")
    
    try:
        # Apriamo il documento usando 'with' così si chiude da solo alla fine
        with fitz.open(percorso_pdf) as doc:
            dati_estratti = []
            
            # --- 1. LE SCATOLE DELLA MEMORIA ---
            # Inizializziamo le variabili fuori dal ciclo. Conserveranno il valore
            # finché non troveranno un nuovo titolo!
            sezione_corrente = "Sconosciuta"
            titolo_corrente = "Nessun Titolo"
            
            # --- 2. L'ESPRESSIONE REGOLARE (RegEx) ---
            # Spiegazione del pattern:
            # ^           : Inizio della frase
            # (\d+\.\d+)  : Un numero, un punto, un altro numero (es. 2.3) -> Gruppo 1
            # \s+         : Uno o più spazi vuoti
            # (.+)        : Tutto il resto del testo (es. CHECKS AND MAINTENANCE) -> Gruppo 2
            pattern_titolo = re.compile(r"^(\d+\.\d+)\s+(.+)")

            # Iniziamo a sfogliare le pagine
            for num_pagina in range(len(doc)):
                pagina = doc[num_pagina]
                
                # --- 3. IL RITAGLIO (Crop) ---
                # Prendiamo le dimensioni totali della pagina
                larghezza = pagina.rect.width
                altezza = pagina.rect.height
                
                # Tagliamo i primi 80 pixel in alto e gli ultimi 60 in basso per 
                # rimuovere "B-83525EN/07", "MAINTENANCE" e il numero di pagina "- 11 -"
                # (Nota: se vedi che taglia troppo o troppo poco, modifica l'80 e il 60)
                area_utile = fitz.Rect(0, 80, larghezza, altezza - 60)
                
                # --- 4. ESTRAZIONE A BLOCCHI ---
                # Usiamo "blocks" ristretti all'area utile. 
                # Questo evita che "(a) Before operation" venga separato dal suo testo.
                blocchi = pagina.get_text("blocks", clip=area_utile)
                
                for blocco in blocchi:
                    # In PyMuPDF, il testo estratto di un blocco si trova all'indice 4
                    testo_blocco = blocco[4].strip()
                    
                    # Se il blocco è un'immagine vuota o spazi bianchi, lo ignoriamo
                    if not testo_blocco:
                        continue
                        
                    # Puliamo il testo trasformando i vari "A capo" interni in spazi normali
                    # Questo crea paragrafi belli e continui per l'Intelligenza Artificiale
                    testo_pulito = testo_blocco.replace('\n', ' ')
                    
                    # --- 5. LA LOGICA DI RICONOSCIMENTO TITOLI ---
                    match = pattern_titolo.search(testo_pulito)
                    
                    if match:
                        # ABBIAMO TROVATO UN TITOLO! Aggiorniamo la memoria
                        sezione_corrente = match.group(1) # Prende il "2.3"
                        titolo_corrente = match.group(2)  # Prende "CHECKS AND MAINTENANCE"
                        
                        # Visto che questa frase è solo il titolo, saltiamo la creazione 
                        # del blocco di testo e passiamo al paragrafo successivo
                        continue 

                    # --- 6. CREAZIONE DEL RECORD ---
                    # Se non è un titolo, è testo normale. Creiamo il record JSON
                    # associando la sezione e il titolo attualmente salvati in memoria.
                    record = {
                        "document_id": "DOC001",
                        "file_name": percorso_pdf,
                        "page": num_pagina + 1,
                        "section": sezione_corrente,
                        "title": titolo_corrente,
                        "text": testo_pulito
                    }
                    
                    # Aggiungiamo il pezzettino alla lista finale
                    dati_estratti.append(record)
                    
        # --- 7. SALVATAGGIO SU FILE ---
        with open(percorso_output, "w", encoding="utf-8") as file_json:
            json.dump(dati_estratti, file_json, indent=4, ensure_ascii=False)
            
        print(f"MAGIA COMPLETATA! 🎉")
        print(f"Ho estratto {len(dati_estratti)} blocchi strutturati nel file: {percorso_output}")

    except Exception as e:
        print(f"Ops! Si è verificato un errore: {e}")

# Questa riga dice a Python di far partire la funzione quando avvii lo script
if __name__ == "__main__":
    estrai_testo_manuale()