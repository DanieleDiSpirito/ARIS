"""
chunker.py
==========
Script generico di chunking.
Carica i file JSON estratti, applica split basati su token (HuggingFace per locale o TikToken per cloud),
garantisce l'iniezione del contesto del titolo/sezione e gestisce le tabelle Markdown spezzate.
"""

import os
import sys
import json
import re
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configura console Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def extract_table_header(text: str) -> Optional[str]:
    """Cerca e restituisce le righe di intestazione di una tabella Markdown se presente."""
    lines = text.split('\n')
    for i in range(len(lines) - 1):
        line1 = lines[i].strip()
        line2 = lines[i+1].strip()
        if line1.startswith('|') and line2.startswith('|') and '-' in line2:
            if re.match(r'^[\s\|:\-]+$', line2):
                return f"{line1}\n{line2}"
    return None


def chunk_file(
    percorso_json: str, 
    output_dir: str, 
    chunk_size: int, 
    chunk_overlap: int, 
    tipo_modello: str, 
    nome_tokenizer: str
) -> None:
    """Carica un singolo file JSON, applica il chunking e lo salva."""
    nome_base = os.path.basename(percorso_json)
    nome_pdf = nome_base.replace(".json", ".pdf")
    
    output_file = os.path.join(output_dir, nome_base.replace(".json", f"_chunks_{tipo_modello}_{chunk_size}.json"))
    
    print(f"🔪 Chunking del file: {nome_base} (Size: {chunk_size}, Overlap: {chunk_overlap}, Env: {tipo_modello})...")
    
    with open(percorso_json, "r", encoding="utf-8") as f:
        dati_grezzi = json.load(f)

    # Inizializzazione dello splitter
    if tipo_modello == "cloud":
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=nome_tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    elif tipo_modello == "locale":
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(nome_tokenizer)
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""]
            )
        except Exception as e:
            print(f"⚠️ Impossibile caricare il tokenizer di HuggingFace ({e}). Fallback su split per caratteri.")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 4,  # Conversione grezza approssimativa token -> caratteri
                chunk_overlap=chunk_overlap * 4,
                separators=["\n\n", "\n", ".", " ", ""]
            )
    else:
        raise ValueError("tipo_modello deve essere 'cloud' o 'locale'")

    chunks_finali = []

    for item in dati_grezzi:
        testo = item.get("text", "").strip()
        if not testo:
            continue
            
        # Iniezione Titolo / Sezione per preservare il contesto (Context-Aware)
        titolo_sezione = f"[Documento: {item.get('file_name', nome_pdf)} | Sezione: {item.get('section', 'Generale')} - {item.get('title', 'Introduzione')}]\n"
        testo_da_dividere = titolo_sezione + testo
        
        frammenti = text_splitter.split_text(testo_da_dividere)
        
        current_table_header = None
        
        for i, frammento in enumerate(frammenti):
            # Gestione delle Tabelle spezzate
            header_in_frammento = extract_table_header(frammento)
            if header_in_frammento:
                current_table_header = header_in_frammento
            elif current_table_header and frammento.strip().startswith('|') and not extract_table_header(frammento[:150]):
                # Se è una riga di tabella ma non ha l'header, lo iniettiamo
                frammento = f"{current_table_header}\n{frammento.strip()}"
                
            # Salta frammenti cortissimi privi di valore semantico
            testo_pulito = frammento.replace(titolo_sezione, "").strip()
            if len(testo_pulito) < 40:
                continue
                
            section_safe = str(item.get('section', '0')).replace('.', '_')
            has_table_flag = "|" in frammento and "---" in frammento
            
            # Rimuove le righe divisorie di Markdown (es. |---|---|) prima del salvataggio nel testo finale del chunk
            testo_senza_divisori = re.sub(r'^[\s\|:\-]+$\n?', '', frammento, flags=re.MULTILINE).strip()
            
            chunk_record = {
                "chunk_id": f"{item['document_id']}_{section_safe}_{item['page']}_{i}_{tipo_modello}",
                "document_id": item["document_id"],
                "file_name": item["file_name"],
                "page": item["page"],
                "section": item["section"],
                "title": item["title"],
                "text": testo_senza_divisori,
                "token_size_target": chunk_size,
                "modello_target": tipo_modello,
                "char_count": len(testo_senza_divisori),
                "has_table": has_table_flag,
                "has_alarm_code": bool(re.search(r'\b[A-Z]{3,4}-\d{3}\b', frammento))
            }
            chunks_finali.append(chunk_record)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks_finali, f, indent=4, ensure_ascii=False)
        
    print(f"   ✓ Completato. Creato {output_file} con {len(chunks_finali)} chunk.\n")


def main():
    parser = argparse.ArgumentParser(description="Chunking generico per pipeline PDF")
    parser.add_argument("--size", type=int, default=500, help="Dimensione target del chunk in token")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap dei chunk in token")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale",
                        help="Ambiente target per determinare il tokenizer")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_folder = os.path.join(base_dir, "output_data", "processed_json")
    output_folder = os.path.join(base_dir, "output_data", "chunks")
    
    os.makedirs(output_folder, exist_ok=True)
    
    tokenizer_name = "BAAI/bge-m3" if args.env == "locale" else "text-embedding-3-small"
    
    json_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.json')]
    
    if not json_files:
        print(f"❌ Nessun file JSON trovato in '{input_folder}'. Esegui prima loader.py.")
        sys.exit(1)
        
    print(f"🚀 Avvio chunking su {len(json_files)} file JSON. Target: {args.size} token (Overlap: {args.overlap})\n")
    
    for file_path in json_files:
        chunk_file(
            percorso_json=file_path,
            output_dir=output_folder,
            chunk_size=args.size,
            chunk_overlap=args.overlap,
            tipo_modello=args.env,
            nome_tokenizer=tokenizer_name
        )
        
    print("🎉 BATCH PROCESSING DI CHUNKING COMPLETATO! 🎉")


if __name__ == "__main__":
    main()
