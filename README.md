<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyMuPDF-1.27-009688?style=for-the-badge" alt="PyMuPDF"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/RAG-Pipeline-FF6F00?style=for-the-badge" alt="RAG"/>
</p>

<h1 align="center">🤖 ARIS</h1>
<h3 align="center"><i>Assistente per Robot Industriale Smart</i></h3>

<p align="center">
  Un assistente tecnico intelligente basato su <b>Retrieval-Augmented Generation (RAG)</b><br>
  per la manutenzione del controller <b>FANUC R-30iB Mate / R-30iB Mate Plus</b>
</p>

---

## 📋 Indice

- [Panoramica](#-panoramica)
- [Architettura](#-architettura)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Documentazione Tecnica](#-documentazione-tecnica)
- [Installazione](#-installazione)
- [Utilizzo](#-utilizzo)
- [Pipeline di Estrazione](#-pipeline-di-estrazione)
- [Valutazione Sperimentale e Risultati](#-valutazione-sperimentale-e-risultati)
- [Roadmap](#-roadmap)
- [Licenza](#-licenza)

---

## 🔍 Panoramica

**ARIS** è un sistema RAG progettato come progetto di tesi per fornire assistenza tecnica intelligente nella manutenzione di robot industriali FANUC. Il sistema:

1. 📄 **Estrae** informazioni da manuali tecnici PDF (testo + tabelle)
2. 🔎 **Recupera** i passaggi più rilevanti tramite ricerca semantica
3. 🧠 **Genera** risposte tecniche accurate e verificabili tramite LLM
4. 📌 **Cita** le fonti documentali (documento, pagina, sezione)
5. ⚠️ **Rifiuta** di rispondere quando il contesto è insufficiente

### Esempio di Interazione

```
👤 Domanda:  Il controller non si avvia. Quali controlli devo fare?

🤖 Risposta: Secondo la documentazione, i controlli da effettuare sono:
   1. Verificare il circuit breaker del controller
   2. Controllare il power supply unit e la conversione AC/DC
   3. Verificare il cavo del teach pendant per torsioni eccessive
   4. Controllare il controller e i dispositivi periferici per anomalie
   
   📄 Fonte: checks_maintenance.pdf, pag. 11, sez. 2.3
   📄 Fonte: overview_configuration.pdf, pag. 10, sez. 2.1
```

---

## 🏗 Architettura

Il sistema segue una classica pipeline **Retrieval-Augmented Generation**:

```
 ┌─────────────────────────────────────────────────────┐
 │                  📁 DOCUMENTI TECNICI                │
 │   manuali · codici errore · procedure · schede      │
 └──────────────────────┬──────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────┐
 │              ⚙️  PREPROCESSING & ESTRAZIONE          │
 │   PyMuPDF (testo) + pdfplumber (tabelle)            │
 │   Pulizia · Filtro rumore · Riconoscimento sezioni  │
 └──────────────────────┬──────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────┐
 │                  ✂️  CHUNKING                         │
 │   Suddivisione semantica · Overlap · Metadati       │
 └──────────────────────┬──────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────┐
 │              🧬 EMBEDDINGS + VECTOR DB               │
 │   Vettorizzazione dei chunk · Indicizzazione         │
 └──────────────────────┬──────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────┐
 │          💬 QUERY → RETRIEVAL → LLM → RISPOSTA       │
 │   Ricerca semantica top-k · Prompt engineering      │
 │   Generazione risposta con fonti documentali        │
 └──────────────────────────────────────────────────────┘
```

---

## 📂 Struttura del Progetto

```
ARIS/
├── 📁 artifacts/                        # Artifacts del progetto
│   ├── 📄 analysis_chunking.md
│   └── 📄 analysis_knowledge_base.md
│
├── 📁 data/
│   ├── 📁 raw/                          # Documentazione originale (manuali, procedure, cablaggi, ecc.)
│   ├── 📁 processed/                    # Testo strutturato estratto per ciascun parser (JSON)
│   │   ├── 📁 docling/
│   │   ├── 📁 euristico/
│   │   ├── 📁 llamaparse/
│   │   ├── 📁 mineru/
│   │   ├── 📁 pdf4llm/
│   │   └── 📁 qwen/
│   ├── 📁 chunks/                       # Dataset di chunk pronti per l'indicizzazione
│   └── 📁 metrics/                      # Risultati dei benchmark ed esperimenti qualitativi/quantitativi
│       ├── 📄 raw_new_benchmarks.json   # Dati grezzi dei benchmark
│       ├── 📄 report_sperimentale_completo.md # Report unificato delle metriche
│       ├── 📄 questions_quality_it.csv  # Test set di 100 domande in Italiano
│       └── 📄 questions_quality_en.csv  # Test set di 100 domande in Inglese
├── 📁 src/
│   ├── 📁 app/                          # Interfaccia Utente (Streamlit)
│   │   └── 📄 app.py                    # Script principale del chatbot Streamlit
│   ├── 📁 chunking/                     # Logica di chunking semantico
│   │   └── 📄 chunking.py               # Segmentazione del testo context-aware
│   ├── 📁 embeddings/                   # Database vettoriale
│   │   └── 📄 create_vector_db.py       # Creazione e popolamento di ChromaDB
│   ├── 📁 estrazione/                   # Ingestione dei manuali PDF (loader dei vari parser)
│   │   ├── 📄 loader_docling.py
│   │   ├── 📄 loader_euristico.py
│   │   ├── 📄 loader_llamaparse.py
│   │   ├── 📄 loader_mineru.py
│   │   ├── 📄 loader_pdf4llm.py
│   │   └── 📄 loader_qwen.py
│   ├── 📁 preprocessing/                # Pulizia del testo e arricchimento di dominio
│   │   ├── 📄 cleaner_docling.py
│   │   ├── 📄 cleaner_euristico.py
│   │   ├── 📄 cleaner_llamaparse.py
│   │   ├── 📄 cleaner_mineru.py
│   │   ├── 📄 cleaner_pdf4llm.py
│   │   ├── 📄 cleaner_qwen.py
│   │   └── 📄 domain_ernichment.py
│   ├── 📁 rag_pipeline/                 # Motori di retrieval e generazione RAG
│   │   ├── 📄 inspect_kb.py             # Utility di ispezione del database
│   │   ├── 📄 rag_pipeline.py           # Pipeline Puro RAG (Vector Search)
│   │   ├── 📄 rag_pipeline_hybrid.py    # Pipeline Ibrida (BM25 + Vector Search)
│   │   ├── 📄 rag_pipeline_rerank.py    # Pipeline RAG con Reranking (Cross-Encoder)
│   │   ├── 📄 rag_pipeline_graph.py     # Pipeline GraphRAG (Grafo relazionale)
│   │   └── 📄 visualize_graph.py        # Utility di visualizzazione del grafo
│   └── 📁 utils/                        # Utility comuni
│       └── 📄 telemetry.py              # Tracciamento dei tempi e risorse hardware (Peak Working Set)
│
├── 📁 vector_db/                       # Database vettoriali ChromaDB suddivisi per parser, env e chunk size
│   ├── 📁 chroma_locale_300/           # bge-m3 · collection "langchain" · 300 token
│   ├── 📁 chroma_locale_700/           # bge-m3 · collection "langchain" · 700 token
│   ├── 📁 chroma_locale_1000/          # bge-m3 · collection "langchain" · 1000 token
│   ├── 📁 chroma_cloud_300/            # text-embedding-3-small · collection "langchain" · 300 token
│   ├── 📁 chroma_cloud_700/            # text-embedding-3-small · collection "langchain" · 700 token
│   └── 📁 chroma_cloud_1000/           # text-embedding-3-small · collection "langchain" · 1000 token
|
├── 📁 notebooks/                        # Analisi e sperimentazione
├── 📁 tests/                            # Dataset di test e valutazione
├── 📁 thesis_report/                    # Elaborato di tesi
│
├── 📄 .env                              # API Key e variabili d'ambiente
├── 📄 requirements.txt                  # Dipendenze Python
├── 📄 LICENSE                           # MIT License
├── 📄 INFO.md                           # Informazioni sul progetto
└── 📄 README.md                         # <- Sei qui!
```

---

## 📚 Documentazione Tecnica

Il sistema elabora la documentazione ufficiale FANUC per il controller **R-30iB Mate / R-30iB Mate Plus**:

| ID | Documento | Tipo | Contenuto |
|:---:|---|---|---|
| 01 | `safety_precautions.pdf` | 🛡️ Sicurezza | Avvertenze e precauzioni di sicurezza |
| 02 | `overview_configuration.pdf` | 📖 Panoramica | Componenti principali e configurazione |
| 03 | `checks_maintenance.pdf` | 🔧 Manutenzione | Procedure di manutenzione ordinaria |
| 04 | `troubleshooting_alarms.pdf` | 🚨 Errori | Codici errore e allarmi |
| 05 | `troubleshooting_visual.pdf` | 🔍 Diagnostica | Troubleshooting visivo |
| 06 | `circuit_boards_amplifiers.pdf` | ⚡ Schede | Circuiti e amplificatori |
| 07 | `replacing_units.pdf` | 🔄 Procedure | Sostituzione componenti |
| 08 | `connections.pdf` | 🔌 Cablaggi | Schemi e descrizioni cablaggi |

---

## ⚡ Installazione

### Prerequisiti

- Python 3.11+
- [Conda](https://docs.conda.io/) (consigliato)
- [LM Studio](https://lmstudio.ai/) per l'esecuzione locale del LLM

### Setup con Conda

```bash
# Clona il repository
git clone https://github.com/DanieleDiSpirito/ARIS.git
cd ARIS

# Crea e attiva l'ambiente (Python 3.11)
conda create -n aris_311 python=3.11 -y
conda activate aris_311

# Installa le dipendenze
pip install -r requirements.txt
```

### Configurazione `.env`

Il progetto utilizza variabili d'ambiente per configurare gli endpoint locali e le chiavi API dei servizi cloud. Nella root del progetto è presente il file modello [.env.example](.env.example). 

Per configurare l'ambiente:

1. Copia il file di esempio rinominandolo in `.env`:
   ```bash
   cp .env.example .env
   ```
2. Modifica i valori all'interno del file `.env` secondo le tue necessità (ad esempio inserendo la tua chiave API di OpenRouter in `OPENAI_API_KEY` o configurando l'endpoint del LLM locale).


---

## 🚀 Utilizzo

### 1. Estrazione del Testo

Esegui gli script di estrazione dalla directory `src/estrazione/`:

```bash
# Estrae testo e tabelle da overview_configuration.pdf
python src/estrazione/estrazione_overview_configuration.py

# Estrae testo da checks_maintenance.pdf
python src/estrazione/estrazione_checks_maintenance.py
```

I file JSON estratti vengono salvati in `data/processed/`.

### 2. Formato dell'Output

Ogni blocco estratto segue questo schema JSON:

```json
{
    "document_id": "02",
    "file_name": "manuali_manutenzione/overview_configuration.pdf",
    "page": "10",
    "section": "2.1",
    "title": "EXTERNAL VIEW OF THE CONTROLLER",
    "text": "- Main board The main board contains a microprocessor..."
}
```

| Campo | Descrizione |
|---|---|
| `document_id` | ID univoco dal catalogo documentale |
| `file_name` | Percorso relativo al PDF sorgente |
| `page` | Numero di pagina nel manuale originale |
| `section` | Codice sezione (es. 2.1, 2.3) |
| `title` | Titolo della sezione corrente |
| `text` | Testo estratto e pulito |

### 3. Pipeline RAG — Test da Terminale

Gli script si avviano dalla **root del progetto** (`ARIS/`) e supportano i seguenti argomenti da CLI:

| Argomento | Valori | Default | Descrizione |
|---|---|---|---|
| `--env` | `locale`, `cloud` | `locale` | Modello ed embedding (Locale: LM Studio/BGE-M3, Cloud: OpenRouter/text-embedding-3-small) |
| `--chunk_size` | `300`, `700`, `1000` | `700` | Dimensione dei chunk del database da interrogare |
| `--metodo` | `pdf4llm`, `euristico`, `docling`, `llamaparse`, `mineru`, `qwen` | `pdf4llm` | Ingestion parser del database selezionato |
| `--query` | stringa | `"Cosa significa l'allarme SRVO-004?"` | Domanda tecnica da inviare alla pipeline RAG |
| `--debug` | (flag) | (disattivo) | Mostra i log di debug e i metadati dei chunk recuperati |

```bash
# 1. Puro RAG (Vector Search)
python src/rag_pipeline/rag_pipeline.py --env cloud --chunk_size 700 --metodo pdf4llm --query "Specifiche Main board A05B-2650-H001"

# 2. RAG Ibrido (BM25 + Vector Search)
python src/rag_pipeline/rag_pipeline_hybrid.py --env locale --chunk_size 700 --metodo pdf4llm --query "Cosa significa l'allarme SRVO-004?"

# 3. Rerank RAG (Vector + BM25 + Cross-Encoder Reranker)
python src/rag_pipeline/rag_pipeline_rerank.py --env cloud --chunk_size 700 --metodo pdf4llm --query "Procedura per la sostituzione della batteria del controller"

# 4. GraphRAG (Grafo Relazionale + Cross-Encoder Reranker)
python src/rag_pipeline/rag_pipeline_graph.py --env cloud --chunk_size 700 --metodo pdf4llm --query "Quali moduli sono collegati al cabinet del controller?"
```

> **Modelli di Embedding associati:**
> - `--env locale` → `BAAI/bge-m3` (1024 dim, locale HuggingFace)
> - `--env cloud` → `text-embedding-3-small` (1536 dim, OpenAI via OpenRouter)

### 4. Interfaccia Streamlit

```bash
# Attiva l'ambiente conda ed esegui l'app streamlit
conda activate aris_311
streamlit run src/app/app.py
```

L'interfaccia utente Streamlit permette di interagire graficamente con l'assistente chatbot. Supporta lo **streaming dei token in tempo reale** per la visualizzazione progressiva della risposta dell'LLM.

Dalla sidebar è possibile configurare al volo:
- **Motore LLM/Embedding** (Locale o Cloud)
- **Dimensione Chunk** (300, 700, 1000)
- **Parser di Ingestione** (pdf4llm, euristico, docling, llamaparse, mineru, qwen)
- **Algoritmo RAG** (Puro, Ibrido, Rerank, Graph)
- **Debug View** (abilitazione della visualizzazione dei chunk recuperati con sorgenti, punteggi e metadati)

---

## ⚙️ Pipeline di Estrazione

L'estrazione utilizza un approccio **dual-engine** per massimizzare la qualità:

### Engine 1 — pdfplumber (Tabelle)
- Rileva tabelle strutturate nel PDF
- Salva le bounding box per evitare duplicati
- Converte le tabelle in formato leggibile:
  ```
  Specifiche per CE controller -> EMC Standard: EN 55011 | Robot Standard: EN/ISO 10218-1
  ```

### Engine 2 — PyMuPDF (Testo)
- Estrae il testo a blocchi con crop dell'area utile
- Filtra header, footer e numeri di pagina
- Riconosce automaticamente sezioni e titoli tramite regex
- Gestisce l'ereditarietà del contesto (sezione/titolo)
- Filtra rumore da etichette di immagini

### Preprocessing
- ✅ Rimozione newline interni e normalizzazione spazi
- ✅ Preservazione codici tecnici, sigle e unità di misura
- ✅ Mantenimento avvertenze di sicurezza (WARNING, NOTE)
- ✅ Filtro anti-rumore per testi < 5 parole

---

## 📊 Valutazione Sperimentale e Risultati

Il sistema ARIS è stato testato e valutato empiricamente su tutte le fasi della sua pipeline di elaborazione. Di seguito vengono riportati i principali risultati estratti dal [Report Sperimentale Completo](data/metrics/report_sperimentale_completo.md).

I test qualitativi e quantitativi si basano su due test set da **100 domande ciascuno** (in lingua italiana e in lingua inglese), coprendo scenari reali di manutenzione FANUC categorizzati per difficoltà (Low, Medium, Hard) e tipologia (Codici errore, Procedure, Troubleshooting, Consultazione).

### 1. Ingestione dei PDF (Benchmark Risorse e Scalabilità)
Valutazione del tempo di esecuzione e del picco massimo di RAM e VRAM per l'estrazione dei documenti tecnici originali con i vari parser:

| Parser | File | Tempo (s) | RAM (MB) | VRAM (MB) | Esecuzione |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Docling** | `safety_precautions.pdf` (13 pag) | 6.79 | 2765.77 | 1189.70 | Locale (GPU CUDA PyTorch) |
| **LlamaParse** | `safety_precautions.pdf` (13 pag) | 41.03 | 925.14 | 0.00 | Cloud API |
| **MinerU** | `safety_precautions.pdf` (13 pag) | 36.11 | 503.10 | 0.00 | Locale CLI (magic-pdf) |
| **Pdf4llm** | `safety_precautions.pdf` (13 pag) | 3.67 | 876.36 | 0.00 | Locale (CPU PyMuPDF) |
| **Qwen** | `safety_precautions.pdf` (13 pag) | 251.72 | 908.54 | 5456.37 | Locale GPU (Transformers Qwen2-VL) |

> 📌 **Verdetto:** **PDF4LLM** si dimostra il loader CPU più veloce in assoluto, ideale per pipelines leggere. **Docling** offre prestazioni eccellenti su GPU locale. **Qwen2-VL** e **LlamaParse** presentano tempi d'attesa e requisiti hardware significativamente più onerosi.

### 2. Analisi Qualitativa della Pulizia dei Chunk per Layout
Analisi della pulizia del testo segmentato (dimensione target 700 token) in base al layout del manuale originale:

#### Categoria: Layout Tecnico (`troubleshooting_alarms.pdf`)
*Manuali densi di allarmi e diagnostica causa-effetto. Misura la conservazione dei codici critici (es. SRVO-004).*
| Metodo | N. Chunk | Garbage Ratio (%) | Righe Orfane (%) | Chunks con Tabelle (%) | Codici Allarme Rilevati |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **euristico** | 274 | 0.02% | 0.62% | 0.00% | **553** |
| **pdf4llm** | 170 | 0.01% | 1.07% | 0.00% | 309 |
| **docling** | 173 | 0.01% | 10.00% | 0.00% | 295 |
| **llamaparse** | 139 | 0.03% | 11.07% | 0.00% | 249 |
| **mineru** | 52 | 0.20% | 6.91% | 0.00% | 208 |
| **qwen** | 238 | 0.03% | 29.16% | 0.00% | 134 |

---

### 3. Valutazione della Fase di Retrieval (Hit Rate@3 e MRR)
Valutazione del corretto recupero delle informazioni pertinenti sul test set italiano (100 domande).

#### Confronto Motori di Ingestione (FASE A - baseline 700 token, Locale BGE-M3)
| DB Name / Parser | Hit Rate@3 (%) (Vector) | MRR (Vector) | Hit Rate@3 (%) (Hybrid) | MRR (Hybrid) |
| :--- | :---: | :---: | :---: | :---: |
| **pdf4llm** | **71.0%** | **0.6417** | **71.0%** | 0.6167 |
| **euristico** | 68.0% | 0.6067 | 69.0% | **0.6217** |
| **llamaparse** | 67.0% | 0.5950 | 67.0% | 0.5700 |
| **mineru** | 65.0% | 0.5683 | 65.0% | 0.5617 |
| **qwen** | 60.0% | 0.5350 | 59.0% | 0.5067 |
| **docling** | 53.0% | 0.4783 | 55.0% | 0.4833 |

#### Sensibilità al Chunk Size e Modello di Embedding (FASE B & C - pdf4llm)
- **Chunk Size (BGE-M3 Locale):** 700 token si dimostra ottimale (HR@3 = **71.0%**, MRR = **0.6417**) rispetto a 300 token (HR@3 = 69.0%) e 1000 token (HR@3 = 71.0%, MRR = 0.6267).
- **Modello di Embedding (pdf4llm 700):** L'embedding locale **BGE-M3** supera l'embedding cloud **text-embedding-3-small** sia in Hit Rate@3 (**71%** vs **68%**) sia in tempi medi di ricerca semantica (**0.022 s** vs **0.347 s**).

---

### 4. Strategie Avanzate (Cross-Encoder Reranking & GraphRAG)
Introduzione di modelli di re-ranking (Cross-Encoder) ed espansione del contesto relazionale tramite GraphRAG:

| Database Config (700 chunk) | Algoritmo di Retrieval | Hit Rate@3 (%) | MRR | Tempo Medio Retrieval |
| :--- | :--- | :---: | :---: | :---: |
| **pdf4llm Cloud 700** | Puro Vettoriale | 68.0% | 0.5733 | 0.3472 s |
| **pdf4llm Cloud 700** | Rerank (Cross-Encoder) | 76.0% | 0.6373 | 3.0101 s |
| **pdf4llm Cloud 700** | **GraphRAG (Grafo Relazionale)** | **83.0%** | **0.7492** | 3.0390 s |
| **pdf4llm Locale 700** | Puro Vettoriale | 71.0% | 0.6417 | 0.0220 s |
| **pdf4llm Locale 700** | **Rerank (Cross-Encoder)** | **73.0%** | **0.6383** | **0.0962 s** |
| **pdf4llm Locale 700** | GraphRAG (Grafo Relazionale) | 70.0% | 0.6217 | 0.1125 s |

> 💡 **Insights:** L'espansione relazionale del contesto operata da **GraphRAG** ottiene le migliori prestazioni assolute in cloud (**83% di Hit Rate@3**), risolvendo query complesse che collegano concetti distribuiti in più pagine. Il **Reranker** locale rappresenta la scelta ottimale a bassa latenza, incrementando l'Hit Rate locale al **73%** in meno di **100 ms**.

---

### 5. Generazione LLM End-to-End (RAGAS ed LLM-as-a-Judge)
Prestazioni end-to-end registrate combinando le pipeline di retrieval con i modelli di generazione:
- **Cloud LLM:** `openai/gpt-4o-mini` (via OpenRouter)
- **Locale LLM:** `phi3.5` (via LM Studio)

| Env | Lingua | Algoritmo | LLM Model | Accuracy (LLM Judge) | Tempo Risposta | Faithfulness (Fedeltà) | Answer Relevancy | Context Precision |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **cloud** | **it** | **Graph** | GPT-4o-mini | **70.0%** | 10.83 s | 0.4508 | 0.8897 | 0.7222 |
| **cloud** | **it** | **Ibrido** | GPT-4o-mini | **70.0%** | **8.29 s** | 0.7011 | **0.9160** | 0.5026 |
| **cloud** | **it** | **Puro** | GPT-4o-mini | 63.3% | 8.27 s | **0.7376** | 0.8400 | **0.7556** |
| **cloud** | **it** | **Rerank** | GPT-4o-mini | **70.0%** | 9.11 s | 0.7063 | 0.8961 | 0.7333 |
| **locale** | **it** | **Puro** | Phi-3.5 | **50.0%** | 111.43 s | 0.5592 | 0.5872 | 0.7222 |
| **locale** | **it** | **Ibrido** | Phi-3.5 | 33.3% | **90.05 s** | **0.6158** | 0.5624 | 0.5381 |
| **locale** | **it** | **Rerank** | Phi-3.5 | 33.3% | 97.11 s | 0.6151 | 0.5589 | **0.8667** |
| **locale** | **it** | **Graph** | Phi-3.5 | 26.7% | 124.41 s | 0.4102 | **0.7037** | **0.8667** |

---

### 📈 Grafici e Telemetrie
Tutti i grafici analitici ed i plot delle metriche sono salvati e consultabili all'interno della cartella [thesis_report/images/](thesis_report/images/):
- **Precisione RAGAS:** [ragas_strategie.png](thesis_report/images/png/ragas_strategie.png)
- **Accuratezza LLM (SI/NO):** [accuracy_strategie.png](thesis_report/images/png/accuracy_strategie.png)
- **Latenze di Retrieval:** [tempo_medio_retrieval_metodi_it.png](thesis_report/images/png/tempo_medio_retrieval_metodi_it.png) e [tempo_medio_retrieval_en.png](thesis_report/images/png/tempo_medio_retrieval_en.png)
- **Tempi Risposta LLM Cloud:** [tempo_medio_risposta_llm_cloud.png](thesis_report/images/png/tempo_medio_risposta_llm_cloud.png)
- **Hit Rate dei Metodi in Italiano:** [hit_rate_metodo_it.png](thesis_report/images/png/hit_rate_metodo_it.png)

---

## 🗺 Roadmap

- [x] Raccolta e organizzazione documentazione FANUC
- [x] Indice documentale (`document_index.csv`)
- [x] Script estrazione con dual-engine (PyMuPDF + pdfplumber)
- [x] Estrazione completa degli 8 manuali PDF
- [x] Preprocessing avanzato e pulizia testuale (Filtro rumore OCR, label `[Caption]`)
- [x] Chunking semantico context-aware (Iniezione Titoli, Gestione Tabelle Markdown, Metadati)
- [x] Creazione embeddings — locale (`BAAI/bge-m3`) e cloud (`text-embedding-3-small`)
- [x] Popolamento Vector database (ChromaDB · collection `langchain` · chunk da 700 token · locale + cloud)
- [x] Pipeline RAG Puro (`rag_pipeline.py`) — Vector Search con embedding dinamico
- [x] Pipeline RAG Ibrida (`rag_pipeline_hybrid.py`) — BM25 + Vector Search (EnsembleRetriever)
- [x] Parametri CLI flessibili (`--env`, `--chunk_size`, `--query`) con DB path dinamico
- [x] Integrazione LLM locale (LM Studio) e Cloud (OpenRouter)
- [x] Prompt engineering avanzato (System/Human separati, lettura tabelle, formattazione adattiva)
- [x] Interfaccia utente Streamlit (`src/app/app.py`) — sidebar con selezione env e chunk_size
- [x] Debug chunk a schermo (anteprima testi e metadati recuperati per ogni query)
- [x] Configurazione LangSmith per tracing della pipeline (variabili `.env` configurate)
- [x] Configurazione motore Cloud (OpenRouter) come predefinito per il Chatbot
- [x] Streaming in tempo reale delle risposte dell'assistente (real-time chat response streaming)
- [x] Integrazione Reranking (Cross-Encoder) per ottimizzazione recupero (`rag_pipeline_rerank.py`)
- [x] Integrazione GraphRAG per espansione relazionale del contesto (`rag_pipeline_graph.py`)
- [x] Costruzione dataset di test completo (100 domande in Italiano e 100 domande in Inglese)
- [x] Valutazione quantitativa delle fasi di retrieval (Hit Rate, Precision, Recall, MRR, tempi)
- [x] Valutazione end-to-end con metriche RAGAS (Faithfulness, Relevancy, Precision, Recall) e accuratezza LLM

---

## 📄 Licenza

Questo progetto è distribuito sotto licenza **MIT**. Vedi il file [LICENSE](LICENSE) per maggiori dettagli.

---

<p align="center">
  <i>Progetto di tesi — Assistente LLM per Manutenzione Industriale</i><br>
  <b>Daniele Di Spirito, Vincenzo Zeppa</b> · 2026
</p>
