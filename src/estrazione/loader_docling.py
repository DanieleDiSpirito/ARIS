import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import sys
import csv
import glob
import json
import re
from sys import argv

# Configurazione dei path per consentire l'importazione della telemetria
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

from src.utils.telemetry import misura_performance

# Import nativi di Docling (IBM)
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import PdfFormatOption
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
except ImportError:
    print("❌ Errore: Libreria 'docling' non installata. Esegui: pip install docling")
    sys.exit(1)


def leggi_metadati_csv(percorso_csv, nome_file_pdf):
    """
    Legge l'indice documentale per estrarre l'ID e la pagina iniziale del manuale.
    Mantiene la compatibilità assoluta con la logica del loader euristico.
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


@misura_performance(metodo="docling")
def esegui_estrazione_docling(percorso_pdf, percorso_csv):
    """
    Esegue l'estrazione del PDF tramite i modelli DLA di Docling.
    Mappa l'albero logico dei nodi Markdown nel formato JSON standard della Knowledge Base.
    Monitorato automaticamente dal decoratore di telemetria.
    """
    nome_file = os.path.basename(percorso_pdf)
    document_id, pagina_iniziale_int, sezione_speciale = leggi_metadati_csv(percorso_csv, nome_file)
    
    # Configurazione della pipeline Docling (Disattiviamo l'OCR se i PDF sono nativi per velocizzare)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # Cambiare in True se si lavora su scansioni d'officina
    pipeline_options.do_table_structure = True  # Forza il Deep Learning sulle tabelle
    # Configurazione più conservativa per evitare crash di memoria
    
    # Configurazione per limitare l'uso di memoria e ottimizzare l'estrazione
    pipeline_options.images_scale = 1.0  # Non scalare le immagini in alta risoluzione
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST  # Usa modalità veloce per le tabelle
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    print(f"🤖 Docling sta analizzando il layout di: {nome_file}...")
    conversion_result = converter.convert(percorso_pdf)
    doc = conversion_result.document
    serializer = MarkdownDocSerializer(doc=doc)
    
    pagine_documento = {}
    
    # Variabili di stato per la gerarchia delle sezioni
    sezione_corrente = "Generale"
    titolo_corrente = "Introduzione"
    
    # Iteriamo programmaticamente su tutti gli elementi strutturali rilevati dall'AI
    for item, _ in doc.iterate_items():
        # Recupera la pagina reale del PDF associata all'elemento (0-indexed in Docling)
        idx_pagina = 0
        if item.prov and len(item.prov) > 0:
            idx_pagina = item.prov[0].page_no - 1  # Riporta a 0-indexed
            
        page_label = calcola_nome_pagina(idx_pagina, pagina_iniziale_int, sezione_speciale)
        
        if page_label not in pagine_documento:
            pagine_documento[page_label] = []
            
        # Se l'elemento è un titolo (heading, section_header, title), aggiorniamo lo stato della gerarchia
        if item.label in ["section_header", "title", "heading"]:
            testo_heading = item.text.strip()
            # Intercetta pattern numerici di sezione (es. 5.6.4 o A.1)
            if re.match(r"^[\d\.]+$", testo_heading):
                sezione_corrente = testo_heading
            elif re.match(r"^([\d\.]+)\s+(.*)", testo_heading):
                match = re.match(r"^([\d\.]+)\s+(.*)", testo_heading)
                sezione_corrente = match.group(1).strip()
                titolo_corrente = match.group(2).strip()
            else:
                titolo_corrente = testo_heading
                
        # Esportiamo l'elemento specifico in formato Markdown (testo o tabelle)
        text_markdown = serializer.serialize(item=item).text.strip()
        
        if text_markdown:
            pagine_documento[page_label].append({
                "section": iper_pulizia_campo(sezione_corrente),
                "title": iper_pulizia_campo(titolo_corrente),
                "text": text_markdown
            })
            
    # Compattiamo i frammenti unendo il testo appartenente alla stessa pagina e sezione
    risultato_json = []
    for page_label, elementi in pagine_documento.items():
        raggruppati = {}
        for elem in elementi:
            chiave_blocco = (elem["section"], elem["title"])
            if chiave_blocco not in raggruppati:
                raggruppati[chiave_blocco] = []
            raggruppati[chiave_blocco].append(elem["text"])
            
        for (sec, tit), testi in raggruppati.items():
            testo_unito = "\n\n".join(testi).strip()
            if testo_unito:
                risultato_json.append({
                    "document_id": document_id,
                    "file_name": nome_file,
                    "page": page_label,
                    "section": sec,
                    "title": tit,
                    "text": testo_unito
                })
                
    return risultato_json


def iper_pulizia_campo(testo):
    """Rimuove eventuali tag markdown residui dai titoli delle sezioni"""
    if not testo:
        return "Generale"
    return re.sub(r'^[#\s\-\*]+', '', testo).strip()


# ==========================================
# ==========================================
# EXECUTOR MASSIVO BATCH
# ==========================================
if __name__ == "__main__":
    if len(argv) < 2:
        print("❌ Uso: python loader_docling.py <nome_cartella_manuali | all>")
        sys.exit(1)
        
    cartella_target = argv[1]
    csv_metadati = os.path.join(PROJECT_ROOT, "data", "raw", "metadata", "document_index.csv")
    cartella_output = os.path.join(PROJECT_ROOT, "data", "processed", "docling")
    os.makedirs(cartella_output, exist_ok=True)
    
    lista_pdf = []
    
    if cartella_target.lower() == "all":
        print(f"📂 [DOCLING BATCH] Lettura di tutti i PDF definiti nell'indice documentale...")
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
    
    print(f"🚀 [DOCLING SOTA] Trovati {len(lista_pdf)} PDF da analizzare.\n")
    
    for pdf_path in lista_pdf:
        nome_pdf = os.path.basename(pdf_path)
        print(f"--- 📄 Inizio Parsing Documentale: {nome_pdf} ---")
        
        records_estratti = esegui_estrazione_docling(pdf_path, csv_metadati)
        
        if records_estratti:
            nome_json = nome_pdf.replace(".pdf", ".json")
            percorso_salvataggio = os.path.join(cartella_output, nome_json)
            
            with open(percorso_salvataggio, "w", encoding="utf-8") as f:
                json.dump(records_estratti, f, indent=4, ensure_ascii=False)
                
            print(f"    ✓ Completato. Salvato in data/processed/docling/{nome_json} ({len(records_estratti)} blocchi).\n")
        else:
            print(f"    ⚠️ Nessun blocco estratto da {nome_pdf}\n")
            
    print("===================================================")
    print("🎉 BATCH PROCESSING CON IBM DOCLING COMPLETATO! 🎉")
    print("===================================================")