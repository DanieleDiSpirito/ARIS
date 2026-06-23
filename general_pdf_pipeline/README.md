# Pipeline PDF4LLM Generalizzata e Valutazione RAG

Questo sotto-progetto contiene una pipeline generica ed indipendente per estrarre, dividere in chunk, indicizzare e **valutare** l'efficacia del retrieval RAG (Vector, BM25 ed Ibrido) su qualsiasi manuale PDF.

Include una strategia **ibrida** che rileva automaticamente i diagrammi visivi di troubleshooting e ne delega l'interpretazione a un **VLM locale** (tramite LM Studio).

---

## 📂 Struttura del Sotto-Progetto

Tutte le operazioni e i dati sono confinati all'interno di questa directory:
```
general_pdf_pipeline/
├── input_manuals/           # Inserisci qui i PDF e l'eventuale document_index.csv
├── output_data/
│   ├── processed_json/      # Testo grezzo estratto per pagina
│   ├── chunks/              # Chunk generati pronti per Chroma
│   └── benchmark_*.md       # Report finali di valutazione generati
├── vector_db/               # Cartelle dei database Chroma generati
├── tests/
│   └── test_questions.csv   # Domande per la valutazione quantitativa
└── src/
    ├── loader.py            # Estrattore ibrido (pdf4llm + VLM locale LM Studio)
    ├── chunker.py           # Splitter dei chunk (Context-Aware e tabelle)
    ├── vector_store.py      # Generatore embeddings e database Chroma
    ├── evaluate_rag.py      # Valutazione automatica (Vector, BM25, Hybrid)
    ├── retrieval_metrics.py # Modulo di calcolo metriche (Hit Rate, Precision, MRR)
    └── query_rag.py         # CLI interattiva per interrogare il DB
```

---

## ⚙️ Come Utilizzare la Pipeline

Attiva prima l'ambiente conda del progetto (`aris`). Tutte le righe di comando vanno eseguite dalla **root del repository principale** (`ARIS/`).

### Step 1: Caricare i Manuali PDF
Posiziona uno o più manuali PDF all'interno di `general_pdf_pipeline/input_manuals/`.
* *(Opzionale)*: Crea un file `document_index.csv` in `general_pdf_pipeline/input_manuals/` per mappare gli offset delle pagine e gli ID documento. La struttura deve essere:
  ```csv
  id_documento,nome_file,tipo_documento,pagina_manuale
  DOC01,nome_manuale.pdf,manuale_manutenzione,10
  ```

### Step 2: Estrazione del Testo (Ibrida)
Esegui l'estrazione. Se desideri elaborare i diagrammi visivi tramite VLM, assicurati che **LM Studio** sia avviato con un modello Vision (es. `llama3.2-vision` o `qwen2-vl`) e passa il flag `--vlm`:

```bash
# Estrazione standard a costo zero (veloce)
python general_pdf_pipeline/src/loader.py

# Estrazione ibrida (rilevamento diagrammi + descrizione VLM via LM Studio)
python general_pdf_pipeline/src/loader.py --vlm
```
*I file estratti verranno salvati in `general_pdf_pipeline/output_data/processed_json/`.*

### Step 3: Chunking del Testo
Suddividi il testo estratto in chunk logici specificando la taglia dei token (es. 500) e l'ambiente:

```bash
python general_pdf_pipeline/src/chunker.py --size 500 --env locale
```
*I chunk generati verranno salvati in `general_pdf_pipeline/output_data/chunks/`.*

### Step 4: Generazione del Vector DB
Crea gli embeddings (usando `BAAI/bge-m3` in locale) e popola il database vettoriale Chroma:

```bash
python general_pdf_pipeline/src/vector_store.py --size 500 --env locale
```
*Il database Chroma verrà scritto in `general_pdf_pipeline/vector_db/chroma_general_locale_500`.*

---

## 🔬 Valutazione e Ricerca

### A. Eseguire la Valutazione Quantitativa (Vector vs BM25 vs Ibrido)
1. Crea un file delle domande in `general_pdf_pipeline/tests/test_questions.csv` seguendo questa struttura:
   ```csv
   id,question,expected_file,expected_page,category,difficulty
   Q001,Come si risolve il problema X?,nome_manuale.pdf,12,diagnostica,low
   ```
2. Avvia la valutazione automatica:
   ```bash
   python general_pdf_pipeline/src/evaluate_rag.py --size 500 --env locale --k 3
   ```
   Lo script calcolerà le metriche di accuratezza per **Pure Vector**, **Pure BM25** e **Hybrid Search** e salverà un report comparativo in `general_pdf_pipeline/output_data/benchmark_retrieval_locale_500.md`.

### B. Ricerca Interattiva da Terminale (CLI)
Puoi interrogare direttamente il database ed analizzare i chunk estratti in tempo reale per fare prove manuali:

```bash
python general_pdf_pipeline/src/query_rag.py --size 500 --env locale
```

**Comandi speciali nel terminale di ricerca:**
* Scrivi normalmente la query per usare il motore **Ibrido** (Default).
* Scrivi `:v <query>` per usare esclusivamente la **Vector Search** (Semantica).
* Scrivi `:b <query>` per usare esclusivamente la **BM25 Search** (Parole chiave).
* Digita `exit` per uscire.
