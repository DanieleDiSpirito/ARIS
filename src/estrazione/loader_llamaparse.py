import os
import sys
import csv
import glob
import json
import re
import asyncio
from sys import argv
from dotenv import load_dotenv

# Applica nest_asyncio per evitare collisioni sull'event loop async di LlamaParse
# try:
#     import nest_asyncio
#     nest_asyncio.apply()
# except ImportError:
#     pass

# Configurazione dei path del repository
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

# Carica variabili d'ambiente dal file .env
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.utils.telemetry import misura_performance

# Importazione di LlamaParse
try:
    from llama_parse import LlamaParse
except ImportError:
    print("❌ Errore: Libreria 'llama-parse' non installata. Esegui: pip install llama-parse")
    sys.exit(1)


def leggi_metadati_csv(percorso_csv, nome_file_pdf):
    """
    Legge l'indice documentale per estrarre l'ID e la pagina iniziale del manuale.
    Mantiene la compatibilità assoluta con la logica degli altri loader.
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
    return re.sub(r'^[#\s\-\*]+', '', testo).strip()


def dividi_pagina_in_blocchi_strutturati(markdown_text, document_id, nome_file, page_label, stato_sezione):
    """
    Divide il markdown della pagina in blocchi separati per ciascuna sezione/titolo trovati.
    Ritorna una lista di dizionari conformi allo schema a 6 campi e lo stato di sezione aggiornato.
    """
    blocchi = []
    linee_accumulate = []
    
    sezione_corrente = stato_sezione["sezione"]
    titolo_corrente = stato_sezione["titolo"]
    
    for line in markdown_text.split("\n"):
        line_strip = line.strip()
        
        # Intercetta intestazioni markdown da # a ######
        match_heading = re.match(r"^(#{1,6})\s+(.*)", line_strip)
        if match_heading:
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
            
            # Aggiorna lo stato della gerarchia
            testo_heading = match_heading.group(2).strip()
            testo_heading = re.sub(r'[*_`]', '', testo_heading) # Rimuove marcatori di stile markdown
            
            match_sezione = re.match(r"^([A-Z0-9]\.[\d\.]*)\s+(.*)", testo_heading, re.IGNORECASE)
            if match_sezione:
                sezione_corrente = match_sezione.group(1).strip().rstrip('.')
                titolo_corrente = match_sezione.group(2).strip()
            elif re.match(r"^(\d+)\s+(.*)", testo_heading):
                match_cifra = re.match(r"^(\d+)\s+(.*)", testo_heading)
                sezione_corrente = match_cifra.group(1).strip()
                titolo_corrente = match_cifra.group(2).strip()
            else:
                titolo_corrente = testo_heading
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
        
    # Aggiorna lo stato esterno
    stato_sezione["sezione"] = sezione_corrente
    stato_sezione["titolo"] = titolo_corrente
    
    return blocchi


@misura_performance(metodo="llamaparse")
async def esegui_estrazione_llamaparse(percorso_pdf, percorso_csv):
    """
    Esegue l'estrazione del PDF tramite le API di LlamaParse.
    Parsea il markdown strutturato riga per riga per ricostruire la gerarchia.
    """
    nome_file = os.path.basename(percorso_pdf)
    document_id, pagina_iniziale_int, sezione_speciale = leggi_metadati_csv(percorso_csv, nome_file)
    
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("❌ Errore: LLAMA_CLOUD_API_KEY non trovata nel file .env!")
        
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        language="en",
        verbose=True
    )
    
    print(f"☁️  LlamaParse sta analizzando il PDF: {nome_file}...")
    
    # aload_data esegue il parsing asincrono e restituisce una lista di Document (uno per pagina)
    documents = await parser.aload_data(percorso_pdf)
    
    risultato_json = []
    
    # Stato gerarchico persistente tra le pagine
    stato_sezione = {
        "sezione": "Generale",
        "titolo": "Introduzione"
    }
    
    for idx_pagina, doc in enumerate(documents):
        page_label = calcola_nome_pagina(idx_pagina, pagina_iniziale_int, sezione_speciale)
        
        # Estraiamo i blocchi strutturati dividendo la pagina per ciascuna intestazione
        blocchi_pagina = dividi_pagina_in_blocchi_strutturati(
            markdown_text=doc.text,
            document_id=document_id,
            nome_file=nome_file,
            page_label=page_label,
            stato_sezione=stato_sezione
        )
        
        risultato_json.extend(blocchi_pagina)
        
    return risultato_json


async def main():
    if len(argv) < 2:
        print("❌ Uso: python loader_llamaparse.py <nome_cartella_manuali | all>")
        sys.exit(1)
        
    cartella_target = argv[1]
    csv_metadati = os.path.join(PROJECT_ROOT, "data", "raw", "metadata", "document_index.csv")
    cartella_output = os.path.join(PROJECT_ROOT, "data", "processed", "llamaparse")
    os.makedirs(cartella_output, exist_ok=True)
    
    lista_pdf = []
    
    # 1. Recupero della lista dei PDF da elaborare
    if cartella_target.lower() == "all":
        print(f"📂 [LLAMAPARSE BATCH] Lettura di tutti i PDF definiti nell'indice documentale...")
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
                        print(f"⚠️ File PDF non trovato: {pdf_path}")
        except Exception as e:
            print(f"❌ Errore durante la lettura di document_index.csv: {e}")
            sys.exit(1)
    else:
        percorso_raw = os.path.join(PROJECT_ROOT, "data", "raw", cartella_target)
        lista_pdf = glob.glob(os.path.join(percorso_raw, "*.pdf"))
        
    print(f"🚀 [LLAMAPARSE SOTA] Trovati {len(lista_pdf)} PDF da analizzare.\n")
    
    # 2. Elaborazione dei PDF con strato di cache intelligente salva-crediti
    for pdf_path in lista_pdf:
        nome_pdf = os.path.basename(pdf_path)
        nome_json = nome_pdf.replace(".pdf", ".json")
        percorso_salvataggio = os.path.join(cartella_output, nome_json)
        
        # STRATO DI CACHE SALVA-CREDITI:
        # Se il file è già stato elaborato in precedenza ed è valido, lo saltiamo!
        if os.path.exists(percorso_salvataggio) and os.path.getsize(percorso_salvataggio) > 0:
            print(f"⏭️  [CACHE HIT] {nome_pdf} già elaborato con successo. Salto per preservare i crediti LlamaParse! (File: {percorso_salvataggio})")
            continue
            
        print(f"--- 📄 Inizio Parsing Documentale: {nome_pdf} ---")
        try:
            records_estratti = await esegui_estrazione_llamaparse(pdf_path, csv_metadati)
            
            if records_estratti:
                with open(percorso_salvataggio, "w", encoding="utf-8") as f:
                    json.dump(records_estratti, f, indent=4, ensure_ascii=False)
                print(f"    ✓ Completato. Salvato in data/processed/llamaparse/{nome_json} ({len(records_estratti)} blocchi).\n")
            else:
                print(f"    ⚠️ Nessun blocco estratto da {nome_pdf}\n")
        except Exception as e:
            print(f"❌ Errore durante il parsing di {nome_pdf}: {e}\n")
            
    print("====================================================")
    print("🎉 BATCH PROCESSING CON LLAMAPARSE COMPLETATO! 🎉")
    print("====================================================")

if __name__ == "__main__":
    asyncio.run(main())