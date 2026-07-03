"""
loader.py
=========
Estrattore generico di documenti PDF.
Usa pdf4llm per l'estrazione rapida locale e si integra con LM Studio (VLM locale)
per convertire diagrammi di flusso e schemi logici in Markdown strutturato.
"""

import os
import sys
import csv
import json
import re
import base64
import argparse
import requests
import fitz  # PyMuPDF
import pymupdf4llm

# Riconfigura la codifica standard in UTF-8 se eseguito su Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def leggi_mappatura_csv(percorso_csv: str) -> Dict[str, Dict[str, Any]]:
    """
    Legge il CSV di indicizzazione se presente.
    Colonne attese: id_documento, nome_file, tipo_documento, pagina_manuale
    Ritorna un dizionario indicizzato per 'nome_file'.
    """
    mappatura = {}
    if not os.path.exists(percorso_csv):
        return mappatura

    try:
        with open(percorso_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                nome_f = row['nome_file'].strip()
                mappatura[nome_f] = {
                    "id_documento": row['id_documento'].strip(),
                    "tipo_documento": row.get('tipo_documento', 'generico').strip(),
                    "pagina_manuale": row.get('pagina_manuale', '1').strip()
                }
    except Exception as e:
        print(f"⚠️ Errore durante la lettura del CSV di metadati: {e}")
    
    return mappatura


def calcola_nome_pagina(indice_pagina_pdf: int, pagina_manuale_str: str) -> str:
    """Calcola la stringa della pagina fisica applicando l'offset iniziale."""
    if pagina_manuale_str.startswith('s'):
        # Gestione sezione speciale (es. s-1, s-2...)
        try:
            valore = int(pagina_manuale_str.split('-')[-1])
            return f"s-{valore + indice_pagina_pdf}"
        except Exception:
            return f"{pagina_manuale_str}_{indice_pagina_pdf}"
    else:
        try:
            valore = int(pagina_manuale_str)
            return str(valore + indice_pagina_pdf)
        except ValueError:
            return f"{pagina_manuale_str}_{indice_pagina_pdf}"


def rileva_diagramma(page) -> bool:
    """
    Rileva se una pagina contiene elementi grafici (schemi/diagrammi) 
    e scarso testo lineare.
    """
    drawings = page.get_drawings()
    images = page.get_images()
    text = page.get_text().strip()

    # Heuristic 1: Moltissimi vettori grafici (linee/box tipici dei flowchart) con testo non eccessivo
    if len(drawings) > 30 and len(text) < 2000:
        return True

    # Heuristic 2: Presenza di immagini con pochissimo testo lineare
    if len(images) >= 1 and len(text) < 800:
        return True

    return False


def get_loaded_model(host_vlm: str) -> str:
    """Recupera il nome del modello attualmente caricato su LM Studio."""
    try:
        response = requests.get(f"{host_vlm}/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get("data", [])
            if models:
                return models[0]["id"]
    except Exception:
        pass
    return "default-model"


def estrai_con_vlm(page, host_vlm: str, model_vlm: str) -> Optional[str]:
    """Renderizza la pagina in PNG e la invia a LM Studio (VLM locale)."""
    # 1. Renderizza a 150 DPI per avere una buona leggibilità
    pix = page.get_pixmap(dpi=150)
    png_bytes = pix.tobytes("png")
    base64_image = base64.b64encode(png_bytes).decode("utf-8")

    url = f"{host_vlm}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer lm-studio"
    }

    prompt = (
        "Sei un assistente tecnico esperto. Questo è un diagramma di troubleshooting o uno schema logico/elettrico estratto da un manuale.\n"
        "Analizza attentamente l'immagine e convertila in un testo logico e strutturato in formato Markdown.\n"
        "Se si tratta di un albero decisionale (flowchart), ricrea la logica usando liste nidificate ed evidenziando i rami SÌ/NO "
        "(ad esempio, usando costrutti del tipo 'SE [condizione] ALLORA ... ALTRIMENTI ...' o liste strutturate nidificate).\n"
        "Mantieni intatti tutti i termini tecnici, sigle, connettori e codici d'errore."
    )

    payload = {
        "model": model_vlm,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2048
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        resp_json = response.json()
        return resp_json['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"   ⚠️ Chiamata VLM fallita per pagina {page.number + 1}: {e}")
        return None


def iper_pulizia_campo(testo: str) -> str:
    """Rimuove markdown dai titoli per normalizzare i campi section e title."""
    if not testo:
        return "Generale"
    cleaned = re.sub(r'^[#\s\-\*\_\`\:]+', '', testo)
    cleaned = re.sub(r'[#\s\-\*\_\`\:]+$', '', cleaned)
    return cleaned.strip()


def is_callout_or_header_to_ignore(text: str) -> bool:
    """Filtra intestazioni di pagina ripetitive, numeri o allarmi orfani."""
    cleaned = iper_pulizia_campo(text).upper()
    if not cleaned:
        return True
    
    # Intestazioni standard di capitoli generici o note di sicurezza
    if cleaned in ["NOTE", "WARNING", "CAUTION", "IMPORTANT", "NOTICE", "DANGER", 
                   "PRECAUTION", "ATTENZIONE", "AVVERTENZA", "AVVISO"]:
        return True

    # Numeri di pagina
    if re.match(r'^-\s*\d+\s*-$', cleaned) or re.match(r'^\d+$', cleaned):
        return True
        
    return False


def is_parent_section(new_sec: str, current_sec: str) -> bool:
    """Verifica se una sezione è genitore di quella corrente (per evitare regressioni gerarchiche)."""
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


def dividi_pagina_in_blocchi_strutturati(
    markdown_text: str, 
    document_id: str, 
    nome_file: str, 
    page_label: str, 
    stato_sezione: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Spezza il testo della pagina in blocchi separati per ogni titolo di sezione incontrato.
    Garantisce la propagazione del titolo e della sezione attiva.
    """
    blocchi = []
    linee_accumulate = []
    
    sezione_corrente = stato_sezione["sezione"]
    titolo_corrente = stato_sezione["titolo"]
    
    for line in markdown_text.split("\n"):
        line_strip = line.strip()
        if not line_strip:
            continue
            
        # Rileva intestazioni standard Markdown (#)
        match_heading = re.match(r"^(#{1,6})\s+(.*)", line_strip)
        is_implicit_heading = False
        implicit_text = ""
        
        # Rileva intestazioni implicite non formattate (es: "2.1 EXTERNAL VIEW")
        if not match_heading and len(line_strip) < 80 and not line_strip.startswith(("-", "*", "|", "(", "[", "{", "•")):
            if not line_strip.endswith((".", "?", "!", ";", ",")):
                match_appendix = re.match(r"^(APPENDIX\s+[A-Z0-9])", line_strip, re.IGNORECASE)
                match_multilevel = re.match(r"^([A-Z0-9]\.[\d\.]+)\s+", line_strip, re.IGNORECASE)
                match_single = re.match(r"^([A-Z0-9]\.?)\s+(.*)", line_strip, re.IGNORECASE)
                
                is_valid = False
                if match_appendix or match_multilevel:
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
            
            # Parsing dei pattern delle intestazioni
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
            
    # Salva l'ultimo blocco residuo
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


def elabora_pdf(
    percorso_pdf: str, 
    meta_info: Optional[Dict[str, Any]], 
    abilita_vlm: bool, 
    host_vlm: str, 
    model_vlm: str
) -> List[Dict[str, Any]]:
    """Esegue l'estrazione ibrida (pdf4llm + VLM per pagine grafiche)."""
    nome_file = os.path.basename(percorso_pdf)
    
    # Assegna metadati (dal CSV o automatici)
    if meta_info:
        document_id = meta_info["id_documento"]
        pagina_iniziale_str = meta_info["pagina_manuale"]
    else:
        document_id = os.path.splitext(nome_file)[0]
        pagina_iniziale_str = "1"
        
    print(f"📄 pymupdf4llm sta analizzando il layout di: {nome_file}...")
    md_chunks = pymupdf4llm.to_markdown(percorso_pdf, page_chunks=True, write_images=False)
    
    doc = fitz.open(percorso_pdf)
    risultato_json = []
    
    stato_sezione = {
        "sezione": "Generale",
        "titolo": "Introduzione"
    }
    
    pagine_grafiche_trovate = 0
    pagine_elaborate_vlm = 0
    
    for idx_pagina in range(len(doc)):
        page_label = calcola_nome_pagina(idx_pagina, pagina_iniziale_str)
        
        # Recupera testo standard
        text_page = md_chunks[idx_pagina]["text"]
        
        # Controlla se la pagina è grafica
        pagina = doc[idx_pagina]
        is_visual = rileva_diagramma(pagina)
        
        if is_visual:
            pagine_grafiche_trovate += 1
            if abilita_vlm:
                print(f"   👁️ Rilevato diagramma/schema a pagina {idx_pagina + 1} (Fisica: {page_label}). Chiamata VLM...")
                vlm_text = estrai_con_vlm(pagina, host_vlm, model_vlm)
                if vlm_text:
                    text_page = vlm_text
                    pagine_elaborate_vlm += 1
            else:
                print(f"   👁️ Rilevato diagramma a pagina {idx_pagina + 1} (Fisica: {page_label}) ma VLM disattivato.")
        
        # Suddividi in blocchi strutturati per sezione
        blocchi_pagina = dividi_pagina_in_blocchi_strutturati(
            markdown_text=text_page,
            document_id=document_id,
            nome_file=nome_file,
            page_label=page_label,
            stato_sezione=stato_sezione
        )
        risultato_json.extend(blocchi_pagina)
        
    doc.close()
    
    print(f"   ✓ Completato. Rilevate {pagine_grafiche_trovate} pagine grafiche (di cui {pagine_elaborate_vlm} elaborate con successo via VLM).")
    return risultato_json


def main():
    parser = argparse.ArgumentParser(description="Estrattore PDF generico ed ibrido (pdf4llm + VLM locale)")
    parser.add_argument("--vlm", action="store_true", help="Abilita l'arricchimento tramite VLM locale")
    parser.add_argument("--host_vlm", type=str, default="http://localhost:1234/v1", 
                        help="Indirizzo server API di LM Studio")
    parser.add_argument("--model_vlm", type=str, default="", 
                        help="Nome modello VLM in LM Studio (lasciare vuoto per autorilevamento)")
    args = parser.parse_args()

    # Definiamo i percorsi all'interno del sotto-progetto general_pdf_pipeline
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_folder = os.path.join(base_dir, "input_manuals")
    output_folder = os.path.join(base_dir, "output_data", "processed_json")
    csv_metadati = os.path.join(input_folder, "document_index.csv")

    os.makedirs(output_folder, exist_ok=True)

    # Legge mappatura
    mappatura = leggi_mappatura_csv(csv_metadati)
    
    # Determina modello VLM se necessario
    model_vlm = args.model_vlm
    if args.vlm and not model_vlm:
        print(f"🤖 Interrogazione server LM Studio per identificare il modello caricato...")
        model_vlm = get_loaded_model(args.host_vlm)
        print(f"   ✓ Modello rilevato: '{model_vlm}'")

    lista_pdf = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.pdf')]
    
    if not lista_pdf:
        print(f"❌ Nessun file PDF trovato in '{input_folder}'.")
        sys.exit(1)
        
    print(f"🚀 Avvio pipeline di estrazione su {len(lista_pdf)} file PDF. VLM locale: {'ATTIVO' if args.vlm else 'DISATTIVO'}\n")
    
    for pdf_path in lista_pdf:
        nome_pdf = os.path.basename(pdf_path)
        meta_info = mappatura.get(nome_pdf)
        
        records = elabora_pdf(pdf_path, meta_info, args.vlm, args.host_vlm, model_vlm)
        
        if records:
            nome_json = nome_pdf.replace(".pdf", ".json")
            percorso_salvataggio = os.path.join(output_folder, nome_json)
            
            with open(percorso_salvataggio, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=4, ensure_ascii=False)
                
            print(f"    ✓ Salvato in: {percorso_salvataggio} ({len(records)} blocchi).\n")
            
    print("🎉 BATCH PROCESSING DI ESTRAZIONE COMPLETATO! 🎉")


if __name__ == "__main__":
    main()
