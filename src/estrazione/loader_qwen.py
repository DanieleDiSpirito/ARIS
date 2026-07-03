import os
import sys
import csv
import glob
import json
import re
import time
import base64
import argparse
from dotenv import load_dotenv

# Riconfigura la codifica standard in UTF-8 se eseguito su Windows per evitare UnicodeEncodeError in console
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Disattiva symlink HuggingFace per evitare problemi su Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configurazione dei path del repository
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

# Carica variabili d'ambiente dal file .env
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.utils.telemetry import misura_performance

# Importazione di PyMuPDF (fitz) per il rendering delle pagine PDF
try:
    import fitz
except ImportError:
    print("❌ Errore: Libreria 'PyMuPDF' (fitz) non installata. Esegui: pip install PyMuPDF")
    sys.exit(1)

# Importazione condizionale di PyTorch e Transformers
HAS_TRANSFORMERS = False
try:
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    HAS_TRANSFORMERS = True
except ImportError:
    pass


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
    return re.sub(r'^[#\s\-\*]+', '', testo).strip()


def dividi_pagina_in_blocchi_strutturati(markdown_text, document_id, nome_file, page_label, stato_sezione):
    """
    Divide il markdown della pagina in blocchi separati per ciascuna sezione/titolo trovati.
    Ritorna una lista di dizionari conformi allo schema a 6 campi.
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


def prompt_qwen_estrazione():
    """Ritorna il prompt strutturato ottimizzato per l'estrazione visiva da parte di Qwen2-VL"""
    return (
        "You are an expert technical documentation parser.\n"
        "Analyze the provided image of a page from a Fanuc robot maintenance manual.\n"
        "Extract the entire text and content of the page, maintaining the following rules strictly:\n"
        "1. Keep the exact section numbers (e.g. 3.1, 4.2.1, A.2) and titles you find.\n"
        "2. Structure the content using standard markdown:\n"
        "   - Use '#' for the main page title or major sections.\n"
        "   - Use '##', '###' for sub-sections.\n"
        "   - Convert all tables into clean markdown tables ('| Column 1 | Column 2 |').\n"
        "3. Maintain the order of the text exactly as it appears.\n"
        "4. Do not include page numbers (like '- 93 -') or manual codes (like 'B-83525EN/07') in the output.\n"
        "5. Return ONLY the markdown content. Do not add any introduction or explanations."
    )


class QwenTransformersEngine:
    """Gestore del caricamento ed esecuzione locale di Qwen2-VL tramite HuggingFace transformers"""
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct"):
        if not HAS_TRANSFORMERS:
            raise ImportError("❌ Libreria 'transformers' o 'torch' non installata. Impossibile usare la modalità locale.")
        
        print(f"Loading local VLM model: {model_id} on GPU/CUDA...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else "auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        print(f"Model and Processor loaded successfully on {self.device}.")

    def process_image(self, image_bytes):
        """Elabora l'immagine PNG in bytes ed estrae il markdown tramite VLM locale"""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prompt = prompt_qwen_estrazione()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=image, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1500)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()


class QwenLMStudioEngine:
    """Gestore dell'estrazione tramite l'API locale di LM Studio (compatibile OpenAI Vision)"""
    def __init__(self, host="http://localhost:1234", model_name="qwen2.5-vision"):
        try:
            import requests
        except ImportError:
            print("❌ Errore: La libreria 'requests' è richiesta per la modalità LM Studio. Installa con: pip install requests")
            sys.exit(1)
        self.requests = requests
        self.host = host.rstrip('/')
        self.model_name = model_name
        print(f"Initialized LM Studio VLM Client targeting {self.host} with model: {self.model_name}")

    def process_image(self, image_bytes):
        """Invia l'immagine PNG in Base64 all'endpoint locale di LM Studio (formato OpenAI Vision)"""
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        prompt = prompt_qwen_estrazione()
        
        # Formato standard OpenAI Vision supportato da LM Studio
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 2048
        }
        
        url = f"{self.host}/v1/chat/completions"
        try:
            response = self.requests.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                raise RuntimeError(f"LM Studio API returned error {response.status_code}: {response.text}")
        except Exception as e:
            raise RuntimeError(f"Failed to query LM Studio API at {url}: {e}")


class QwenOllamaEngine:
    """Gestore dell'estrazione tramite l'API HTTP locale di Ollama"""
    def __init__(self, model_name="qwen2.5-vision", host="http://localhost:11434"):
        try:
            import requests
        except ImportError:
            print("❌ Errore: La libreria 'requests' è richiesta per la modalità Ollama. Installa con: pip install requests")
            sys.exit(1)
        self.requests = requests
        self.model_name = model_name
        self.url = f"{host}/api/chat"
        print(f"Initialized Ollama VLM Client targeting model: {self.model_name} at {host}")

    def process_image(self, image_bytes):
        """Invia l'immagine PNG in Base64 all'endpoint locale di Ollama"""
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        prompt = prompt_qwen_estrazione()
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64]
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        
        try:
            response = self.requests.post(self.url, json=payload, timeout=90)
            if response.status_code == 200:
                data = response.json()
                return data["message"]["content"].strip()
            else:
                raise RuntimeError(f"Ollama API returned error {response.status_code}: {response.text}")
        except Exception as e:
            raise RuntimeError(f"Failed to query Ollama API: {e}")


@misura_performance(metodo="qwen")
def esegui_estrazione_qwen(percorso_pdf, percorso_csv, engine):
    """
    Esegue l'estrazione visiva pagina per pagina del PDF tramite Qwen VLM.
    Converte ciascuna pagina in immagine e la invia all'engine selezionato.
    """
    nome_file = os.path.basename(percorso_pdf)
    document_id, pagina_iniziale_int, sezione_speciale = leggi_metadati_csv(percorso_csv, nome_file)
    
    print(f"Rendering PDF pages in-memory using PyMuPDF: {nome_file}...")
    doc = fitz.open(percorso_pdf)
    
    risultato_json = []
    
    # Stato gerarchico persistente tra le pagine
    stato_sezione = {
        "sezione": "Generale",
        "titolo": "Introduzione"
    }
    
    for idx_pagina in range(len(doc)):
        page_label = calcola_nome_pagina(idx_pagina, pagina_iniziale_int, sezione_speciale)
        print(f"  → Elaborazione pagina {idx_pagina + 1}/{len(doc)} (Etichetta manuale: {page_label})...")
        
        # Converte la pagina in PNG ad alta risoluzione (150 DPI) direttamente in memoria
        page = doc.load_page(idx_pagina)
        pix = page.get_pixmap(dpi=150)
        image_bytes = pix.tobytes("png")
        
        # Ottiene l'estrazione visiva in markdown tramite il VLM
        try:
            markdown_estratto = engine.process_image(image_bytes)
            
            # Estraiamo i blocchi strutturati dividendo la pagina per ciascuna intestazione
            blocchi_pagina = dividi_pagina_in_blocchi_strutturati(
                markdown_text=markdown_estratto,
                document_id=document_id,
                nome_file=nome_file,
                page_label=page_label,
                stato_sezione=stato_sezione
            )
            
            risultato_json.extend(blocchi_pagina)
            print(f"    ✓ Estratti {len(blocchi_pagina)} blocchi da pagina {page_label}")
        except Exception as e:
            print(f"  ❌ Errore durante l'elaborazione di pagina {page_label}: {e}")
            
    return risultato_json


def main():
    parser = argparse.ArgumentParser(description="Loader basato su Qwen VLM locale (Transformers / LM Studio / Ollama) per ARIS")
    parser.add_argument("target", help="Nome cartella manuali in raw o 'all' per elaborare tutto")
    
    # Flags per selezionare il motore VLM
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--lm-studio", action="store_true", help="Usa LM Studio locale (compatibile OpenAI Vision API)")
    group.add_argument("--ollama", action="store_true", help="Usa Ollama locale anziché HuggingFace transformers")
    
    # Parametri per LM Studio
    parser.add_argument("--lm-host", default="http://localhost:1234", help="Endpoint API di LM Studio")
    parser.add_argument("--lm-model", default="qwen2.5-vision", help="Nome del modello VLM caricato su LM Studio")
    
    # Parametri per Ollama
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Endpoint API di Ollama")
    parser.add_argument("--ollama-model", default="qwen2.5-vision", help="Nome del modello VLM su Ollama")
    
    # Parametri per Transformers
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct", help="Repository HuggingFace del modello locale")
    
    args = parser.parse_args()

    csv_metadati = os.path.join(PROJECT_ROOT, "data", "raw", "metadata", "document_index.csv")
    cartella_output = os.path.join(PROJECT_ROOT, "data", "processed", "qwen")
    os.makedirs(cartella_output, exist_ok=True)
    
    lista_pdf = []
    
    # 1. Recupero della lista dei PDF da elaborare
    if args.target.lower() == "all":
        print(f"📂 [QWEN BATCH] Lettura di tutti i PDF definiti nell'indice documentale...")
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
        percorso_raw = os.path.join(PROJECT_ROOT, "data", "raw", args.target)
        lista_pdf = glob.glob(os.path.join(percorso_raw, "*.pdf"))
        
    print(f"🚀 [QWEN VLM] Trovati {len(lista_pdf)} PDF da analizzare.\n")
    if not lista_pdf:
        print("Nessun file PDF trovato. Uscita.")
        return

    # 2. Inizializzazione dell'Engine VLM selezionato
    if args.lm_studio:
        print("Engine Mode: LM STUDIO (OpenAI Vision API)")
        engine = QwenLMStudioEngine(host=args.lm_host, model_name=args.lm_model)
    elif args.ollama:
        print("Engine Mode: OLLAMA API")
        engine = QwenOllamaEngine(model_name=args.ollama_model, host=args.ollama_host)
    else:
        print("Engine Mode: LOCAL HUGGINGFACE TRANSFORMERS GPU")
        if not HAS_TRANSFORMERS:
            print("❌ Errore: PyTorch o Transformers non trovati nell'ambiente Conda! Installa le dipendenze o usa la modalità --lm-studio o --ollama")
            sys.exit(1)
        engine = QwenTransformersEngine(model_id=args.model_id)

    # 3. Elaborazione dei PDF con caching salva-GPU
    for pdf_path in lista_pdf:
        nome_pdf = os.path.basename(pdf_path)
        nome_json = nome_pdf.replace(".pdf", ".json")
        percorso_salvataggio = os.path.join(cartella_output, nome_json)
        
        # STRATO DI CACHE:
        if os.path.exists(percorso_salvataggio) and os.path.getsize(percorso_salvataggio) > 0:
            print(f"⏭️  [CACHE HIT] {nome_pdf} già elaborato con successo. Salto per salvare tempo e risorse! (File: {percorso_salvataggio})")
            continue
            
        print(f"\n--- 📄 Inizio Parsing Visivo: {nome_pdf} ---")
        start_time = time.time()
        try:
            records_estratti = esegui_estrazione_qwen(pdf_path, csv_metadati, engine)
            
            if records_estratti:
                with open(percorso_salvataggio, "w", encoding="utf-8") as f:
                    json.dump(records_estratti, f, indent=4, ensure_ascii=False)
                duration = time.time() - start_time
                print(f"    ✓ Completato in {duration:.2f}s. Salvato in data/processed/qwen/{nome_json} ({len(records_estratti)} blocchi).\n")
            else:
                print(f"    ⚠️ Nessun blocco estratto da {nome_pdf}\n")
        except Exception as e:
            print(f"❌ Errore durante il parsing visivo di {nome_pdf}: {e}\n")
            import traceback
            traceback.print_exc()
            
    print("====================================================")
    print("🎉 BATCH PROCESSING CON QWEN VLM COMPLETATO! 🎉")
    print("====================================================")

if __name__ == "__main__":
    main()