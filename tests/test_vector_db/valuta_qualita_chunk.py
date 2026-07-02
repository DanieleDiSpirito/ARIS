import os
import json
import re
import pandas as pd
from typing import Dict, List, Any

# Definizione dei percorsi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)

# Caratteri considerati "puliti" o standard (alfanumerici, spazi, punteggiatura comune, sintassi markdown)
CLEAN_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 \n\r\t.,;:!?()[]{}<>+-*/=_|#@$%^&*\"'\\`~°àèéìòùÀÈÉÌÒÙ"
)

def calcola_garbage_ratio(text: str) -> float:
    """Percentuale di caratteri non standard o potenzialmente rumore."""
    if not text:
        return 0.0
    weird_chars = sum(1 for c in text if c not in CLEAN_CHARS)
    return (weird_chars / len(text)) * 100

def calcola_whitespace_anomaly(text: str) -> float:
    """Percentuale di caratteri all'interno di spazi o newline consecutivi eccessivi."""
    if not text:
        return 0.0
    
    # Trova gruppi di 2 o più spazi consecutivi
    consec_spaces = sum(len(m.group(0)) - 1 for m in re.finditer(r' {2,}', text))
    # Trova gruppi di 3 o più newline consecutive
    consec_newlines = sum(len(m.group(0)) - 1 for m in re.finditer(r'\n{3,}', text))
    
    return ((consec_spaces + consec_newlines) / len(text)) * 100

def calcola_orphan_lines_ratio(text: str) -> float:
    """Percentuale di righe corte di rumore o didascalie orfane (es. [Caption], [Page 12])."""
    if not text:
        return 0.0
    lines = text.split('\n')
    orphan_count = 0
    valid_lines = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        valid_lines += 1
        # Riga cortissima e non puramente alfabetica (es. "1", "[ ]", "-")
        if len(line) <= 4 and not line.isalpha():
            orphan_count += 1
        # Didascalia orfana tipica
        elif line.startswith('[') and line.endswith(']') and len(line) < 25:
            orphan_count += 1
            
    return (orphan_count / valid_lines * 100) if valid_lines > 0 else 0.0

def contiene_tabella_markdown(text: str) -> bool:
    """Rileva la presenza di una tabella strutturata in markdown (riga con divisori |---|)."""
    return bool(re.search(r'\|\s*:?-+:?\s*\|', text))

def conta_codici_allarme(text: str) -> int:
    """Trova il numero di codici allarme nel formato SRVO-xxx o DCS-xxx."""
    return len(re.findall(r'\b[A-Z]{3,4}-\d{3}\b', text))

def analizza_metodo(metodo: str) -> Dict[str, Any]:
    """Analizza il file dei chunk per il metodo specificato."""
    chunk_file = os.path.join(PROJECT_ROOT, "data", "chunks", metodo, "dataset_chunks_locale_700.json")
    
    if not os.path.exists(chunk_file):
        print(f"⚠️ File non trovato per il metodo '{metodo}': {chunk_file}")
        return {}
        
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    n_chunks = len(chunks)
    if n_chunks == 0:
        return {}
        
    tot_len = 0
    tot_garbage = 0.0
    tot_anomaly = 0.0
    tot_orphans = 0.0
    tot_tables = 0
    tot_alarms = 0
    
    for chunk in chunks:
        text = chunk.get("text", "")
        tot_len += len(text)
        tot_garbage += calcola_garbage_ratio(text)
        tot_anomaly += calcola_whitespace_anomaly(text)
        tot_orphans += calcola_orphan_lines_ratio(text)
        if contiene_tabella_markdown(text):
            tot_tables += 1
        tot_alarms += conta_codici_allarme(text)
        
    return {
        "Metodo": metodo,
        "Numero Chunk": n_chunks,
        "Lunghezza Media (Caratteri)": int(tot_len / n_chunks),
        "Garbage Ratio (%)": round(tot_garbage / n_chunks, 2),
        "Spazi Anomali (%)": round(tot_anomaly / n_chunks, 2),
        "Righe Orfane (%)": round(tot_orphans / n_chunks, 2),
        "Chunk con Tabelle (%)": round((tot_tables / n_chunks) * 100, 2),
        "Codici Allarme Rilevati": tot_alarms
    }

def main():
    metodi = ["euristico", "docling", "llamaparse", "qwen", "pdf4llm", "mineru"]
    results = []
    
    print("🔬 Avvio analisi della qualità e pulizia dei chunk...")
    for m in metodi:
        res = analizza_metodo(m)
        if res:
            results.append(res)
            
    if not results:
        print("❌ Nessun dato analizzato.")
        return
        
    df = pd.DataFrame(results)
    
    # Stampa a schermo in formato tabellare
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\n📊 TABELLA COMPARATIVA DI QUALITÀ DEI CHUNK (700 TOKEN):")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)
    
    # Genera report in markdown e salvalo
    metrics_dir = os.path.join(PROJECT_ROOT, "data", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    report_path = os.path.join(metrics_dir, "benchmark_qualita_chunks.md")
    
    md_table = df.to_markdown(index=False)
    
    report_content = f"""# Benchmark di Qualità e Pulizia del Testo dei Chunk (ARIS)

Questo report mette a confronto la pulizia e la qualità testuale dei chunk generati dai **5 diversi metodi di parsing** estratti a **700 token** di dimensione target.

## Metriche di Valutazione:
1. **Garbage Ratio (%):** Percentuale di caratteri di rumore non standard rispetto ai caratteri totali del chunk.
2. **Spazi Anomali (%):** Percentuale di caratteri dovuti a spaziazioni consecutive eccessive o newline multiple non pulite.
3. **Righe Orfane (%):** Percentuale di righe di intestazione/didascalia orfane o molto corte rispetto alle righe totali del chunk.
4. **Chunk con Tabelle (%):** Percentuale di chunk in cui è stata rilevata una tabella Markdown valida.
5. **Codici Allarme Rilevati:** Conteggio totale di allarmi di tipo tecnico (es. `SRVO-xxx`) preservati nel testo per le query di diagnostica.

## Tabella Comparativa

{md_table}

---
*Report generato automaticamente dallo script `valuta_qualita_chunk.py`.*
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"\n✅ Report salvato con successo in: {report_path}")

if __name__ == "__main__":
    main()
