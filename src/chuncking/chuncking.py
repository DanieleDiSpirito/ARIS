import json
import os
# Richiede: pip install langchain-text-splitters tiktoken transformers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

def genera_dataset_chunks(input_file, chunk_size, chunk_overlap, tipo_modello, nome_modello):
    """
    Legge il JSON pulito, divide il testo usando il tokenizer corretto (Cloud o Locale)
    e salva un nuovo file JSON.
    """
    # Creiamo un nome file che includa se è per Cloud o Locale
    output_file = f"dataset_chunks_{tipo_modello}_{chunk_size}.json"
    
    print(f"🔪 Avvio chunking per {tipo_modello} ({chunk_size} token, Overlap: {chunk_overlap})...")
    
    # 1. Carica il JSON pulito
    with open(input_file, "r", encoding="utf-8") as f:
        dati_puliti = json.load(f)

    # 2. Configura lo Splitter in base al tipo di modello
    if tipo_modello == "cloud":
        # Usiamo tiktoken per OpenAI
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=nome_modello, # es. "text-embedding-3-small"
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    elif tipo_modello == "locale":
        # Usiamo il tokenizer di HuggingFace per il modello locale
        # Prima scarica/carica il tokenizer esatto del modello
        tokenizer = AutoTokenizer.from_pretrained(nome_modello)
        
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    else:
        raise ValueError("tipo_modello deve essere 'cloud' o 'locale'")

    chunks_finali = []

    # 3. Processa ogni record
    for item in dati_puliti:
        testo_da_dividere = item["text"]
        
        # Divide il testo in una lista di frammenti
        frammenti = text_splitter.split_text(testo_da_dividere)
        
        for i, frammento in enumerate(frammenti):
            chunk_record = {
                "chunk_id": f"{item['document_id']}_{item['page']}_{i}_{tipo_modello}",
                "document_id": item["document_id"],
                "file_name": item["file_name"],
                "page": item["page"],
                "section": item["section"],
                "title": item["title"],
                "text": frammento,
                "token_size_target": chunk_size,
                "modello_target": tipo_modello
            }
            chunks_finali.append(chunk_record)

    # 4. Salva il file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks_finali, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Fatto! Creato {output_file} con {len(chunks_finali)} chunks.\n")

if __name__ == "__main__":
    file_input = r"C:\Users\vince\Documents\GitHub\TestEProve2\preprocessing\knowledge_base_finale_V3.json"
    
    # 1. GENERIAMO I DATASET PER L'ESPERIMENTO SUL CLOUD (OpenAI)
    config_esperimenti_cloud = [
        {"size": 300, "overlap": 40},
        {"size": 700, "overlap": 100},
        {"size": 1000, "overlap": 150}
    ]
    
    print("--- ☁️ PREPARAZIONE DATI PER MODELLO CLOUD ---")
    for config in config_esperimenti_cloud:
        genera_dataset_chunks(
            input_file=file_input, 
            chunk_size=config["size"], 
            chunk_overlap=config["overlap"],
            tipo_modello="cloud",
            nome_modello="text-embedding-3-small"
        )
        
    # 2. GENERIAMO I DATASET PER L'ESPERIMENTO LOCALE (BAAI/bge-m3)
    config_esperimenti_locale = [
        {"size": 300, "overlap": 40},
        {"size": 700, "overlap": 100},
        {"size": 1000, "overlap": 150}
    ]
    
    print("\n--- 💻 PREPARAZIONE DATI PER MODELLO LOCALE ---")
    for config in config_esperimenti_locale:
        genera_dataset_chunks(
            input_file=file_input, 
            chunk_size=config["size"], 
            chunk_overlap=config["overlap"],
            tipo_modello="locale",
            nome_modello="BAAI/bge-m3" # HuggingFace scaricherà il tokenizer in automatico!
        )
        
    print("🎉 Tutti i dataset (Cloud e Locali) sono pronti per gli esperimenti!")