# Catalogo Sperimentale Completo dei Test e delle Metriche (Capitolo 4)

Questo catalogo raccoglie e unifica ogni singolo dato, telemetria ed analisi qualitativa o quantitativa prodotta durante gli esperimenti del **Capitolo 4** per la tesi di laurea ARIS.

---

## 🛠️ SEZIONE 1: Benchmark Hardware e Consumo Risorse (Scaling)

I dati seguenti registrano il tempo di esecuzione ed il vero picco assoluto di RAM e VRAM del processo di caricamento ed estrazione per ciascun parser, valutati su documenti di lunghezza differente per misurarne la scalabilità.

| Parser / Metodo | File PDF (Pagine) | Tempo Elaborazione (s) | Picco RAM (MB) | Picco VRAM (MB) | Tipo di Esecuzione |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Docling** | `test_telemetry.pdf` (1 pag) | 1.66 | 2765.77 | 1006.36 | Locale (GPU CUDA PyTorch) |
| | `checks_maintenance.pdf` (1 pag) | 3.60 | 2064.26 | 442.82 | Locale (GPU CUDA PyTorch) |
| | `safety_precautions.pdf` (13 pag) | 6.79 | 2765.77 | 1189.70 | Locale (GPU CUDA PyTorch) |
| | `overview_configuration.pdf` (33 pag) | 5.42 | 2433.29 | 774.32 | Locale (GPU CUDA PyTorch) |
| **pdf4llm** | `test_telemetry.pdf` (1 pag) | 0.17 | 876.36 | 0.00 | Locale (CPU PyMuPDF) |
| | `checks_maintenance.pdf` (1 pag) | 0.20 | 770.42 | 0.00 | Locale (CPU PyMuPDF) |
| | `safety_precautions.pdf` (13 pag) | 3.67 | 876.36 | 0.00 | Locale (CPU PyMuPDF) |
| | `overview_configuration.pdf` (33 pag) | 3.90 | 855.20 | 0.00 | Locale (CPU PyMuPDF) |
| **LlamaParse** | `test_telemetry.pdf` (1 pag) | 18.83 | 638.82 | 0.00 | Cloud API (LlamaIndex Server) |
| **MinerU** | `test_telemetry.pdf` (1 pag) | 17.98 | 503.10 | 0.00 | Locale CLI (magic-pdf) |
| | `checks_maintenance.pdf` (1 pag) | 44.37 | 503.10 | 0.00 | Locale CLI (magic-pdf) |
| | `safety_precautions.pdf` (13 pag) | 36.11 | 503.10 | 0.00 | Locale CLI (magic-pdf) |
| | `overview_configuration.pdf` (33 pag) | 121.50 | 503.10 | 0.00 | Locale CLI (magic-pdf) |
| **Qwen VLM** | `test_telemetry.pdf` (1 pag) | 26.35 | 4530.01 | 5456.04 | Locale GPU (Transformers Qwen2-VL) |


---

## 📊 SEZIONE 2: Analisi Qualitativa per Tipologia di Layout

## 📊 Categoria: Testo Lineare (Narrativo / Istruzioni di Sicurezza)
**File di riferimento:** `safety_precautions.pdf`

*Documenti composti da paragrafi di testo fluente, elenchi puntati e pochissimi schemi o tabelle. Misura l'abilità del parser di non introdurre caratteri spuri.*

| Metodo     |   Numero Chunk |   Garbage Ratio (%) |   Spazi Anomali (%) |   Righe Orfane (%) |   Chunk con Tabelle (%) |   Codici Allarme Rilevati |
|:-----------|---------------:|--------------------:|--------------------:|-------------------:|------------------------:|--------------------------:|
| docling    |             42 |                0    |                0.05 |               9.71 |                    2.38 |                         7 |
| qwen       |             36 |                0.02 |                0    |              10.55 |                    2.78 |                         6 |
| llamaparse |             38 |                0.03 |                0    |              17.3  |                    2.63 |                         7 |
| pdf4llm    |             40 |                0.03 |                0    |              14.82 |                    2.5  |                         7 |
| mineru     |             26 |                0.08 |                0    |               2.48 |                    3.85 |                         8 |
| euristico  |             60 |                0.11 |                0    |               6.09 |                    3.33 |                        20 |

## 📊 Categoria: Layout Strutturato (Tabelle Pinout / Schemi di Connessione)
**File di riferimento:** `connections.pdf`

*Documentazione ricca di schemi di cablaggio e tabelle pin/segnale. Misura la capacità di ricostruire griglie markdown valide.*

| Metodo     |   Numero Chunk |   Garbage Ratio (%) |   Spazi Anomali (%) |   Righe Orfane (%) |   Chunk con Tabelle (%) |   Codici Allarme Rilevati |
|:-----------|---------------:|--------------------:|--------------------:|-------------------:|------------------------:|--------------------------:|
| qwen       |            394 |                0.04 |                0    |              12.94 |                    1.52 |                         5 |
| docling    |            106 |                0.05 |                0.01 |              15.01 |                    3.77 |                         3 |
| pdf4llm    |            281 |                0.06 |                0    |               8.16 |                    1.78 |                         6 |
| euristico  |            396 |                0.08 |                0    |               7.13 |                    2.02 |                        14 |
| llamaparse |            295 |                0.09 |                0    |              10.1  |                    1.36 |                         6 |
| mineru     |            179 |                0.12 |                0    |               7.61 |                    1.12 |                         6 |

## 📊 Categoria: Layout Tecnico (Troubleshooting / Codici Allarme)
**File di riferimento:** `troubleshooting_alarms.pdf`

*Manuali ricchi di codici diagnostici e tabelle causa-effetto. Misura la preservazione di stringhe chiave (es. SRVO-062) e la rimozione di intestazioni orfane.*

| Metodo     |   Numero Chunk |   Garbage Ratio (%) |   Spazi Anomali (%) |   Righe Orfane (%) |   Chunk con Tabelle (%) |   Codici Allarme Rilevati |
|:-----------|---------------:|--------------------:|--------------------:|-------------------:|------------------------:|--------------------------:|
| docling    |            173 |                0.01 |                0    |              10    |                       0 |                       295 |
| pdf4llm    |            170 |                0.01 |                0    |               1.07 |                       0 |                       309 |
| euristico  |            274 |                0.02 |                0    |               0.62 |                       0 |                       553 |
| llamaparse |            139 |                0.03 |                0.34 |              11.07 |                       0 |                       249 |
| qwen       |            238 |                0.03 |                0    |              29.16 |                       0 |                       134 |
| mineru     |             52 |                0.2  |                0    |               6.91 |                       0 |                       208 |

---

## 🔬 SEZIONE 3: Valutazione delle Fasi di Retrieval (Fasi A, B, C, D)

Le tabelle seguenti sintetizzano l'efficacia del retrieval semantico sul test set di 100 domande in lingua italiana per tutte le fasi sperimentali.

---

## 🔬 FASE A: Confronto dei Motori di Parsing (Baseline: 700 token, Locale BGE-M3)

### 1. Retrieval Vettoriale Puro (Pure Vector)
| DB Name                      |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:-----------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_docling_locale_700    |               53 |             29.33 |             53 | 0.4783 |            0.025  |
| chroma_euristico_locale_700  |               68 |             47.33 |             68 | 0.6067 |            0.0224 |
| chroma_llamaparse_locale_700 |               67 |             32.33 |             67 | 0.595  |            0.0242 |
| chroma_locale_700            |               68 |             47.33 |             68 | 0.6067 |            0.2587 |
| chroma_mineru_locale_700     |               65 |             28.67 |             65 | 0.5683 |            0.2276 |
| chroma_pdf4llm_locale_700    |               71 |             36.67 |             71 | 0.6417 |            0.2434 |
| chroma_qwen_locale_700       |               60 |             27.67 |             60 | 0.535  |            0.2041 |


### 2. Retrieval Ibrido (Hybrid Search)
| DB Name                      |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:-----------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_docling_locale_700    |               55 |             30.67 |             55 | 0.4833 |            0.0338 |
| chroma_euristico_locale_700  |               69 |             31    |             69 | 0.6217 |            0.0274 |
| chroma_llamaparse_locale_700 |               67 |             32.67 |             67 | 0.57   |            0.0266 |
| chroma_locale_700            |               68 |             47.33 |             68 | 0.6067 |            0.2451 |
| chroma_mineru_locale_700     |               65 |             30    |             65 | 0.5617 |            0.2694 |
| chroma_pdf4llm_locale_700    |               71 |             36.67 |             71 | 0.6167 |            0.2645 |
| chroma_qwen_locale_700       |               59 |             26.67 |             59 | 0.5067 |            0.0294 |


---

## 📈 FASE B: Analisi di Sensibilità al Chunk Size (Parser di riferimento: pdf4llm, Locale BGE-M3)

### 1. Retrieval Vettoriale Puro
| DB Name                    |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:---------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_pdf4llm_locale_300  |               69 |             36.67 |             69 | 0.6067 |            0.0298 |
| chroma_pdf4llm_locale_700  |               71 |             36.67 |             71 | 0.6417 |            0.022  |
| chroma_pdf4llm_locale_1000 |               71 |             35.67 |             71 | 0.6267 |            0.0241 |


### 2. Retrieval Ibrido
| DB Name                    |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:---------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_pdf4llm_locale_300  |               68 |             37    |             68 | 0.6017 |            0.0266 |
| chroma_pdf4llm_locale_700  |               71 |             36.67 |             71 | 0.6167 |            0.0232 |
| chroma_pdf4llm_locale_1000 |               70 |             36    |             70 | 0.61   |            0.0263 |


---

## ☁️ FASE C: Impatto del Modello di Embedding (Locale vs Cloud, pdf4llm, 700 token)

### 1. Retrieval Vettoriale Puro (Cloud OpenAI)
| DB Name                  |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:-------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_pdf4llm_cloud_700 |               68 |             34.67 |             68 | 0.5733 |            0.3472 |


### 2. Retrieval Ibrido (Cloud OpenAI)
| DB Name                  |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:-------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_pdf4llm_cloud_700 |               68 |                35 |             68 | 0.5867 |            0.2077 |


---

## 🔀 FASE D: Confronto delle Strategie di Ricerca (pdf4llm, 700 token, Locale BGE-M3)
| Strategia | Hit Rate@3 (%) | Precision@3 (%) | Recall@3 (%) | MRR | Tempo Medio (s) |
|:---|---:|---:|---:|---:|---:|
| **PURE_BM25 (Lessicale)** | 30.0 | 13.0 | 30.0 | 0.22 | 0.0015 |
| **PURE_VECTOR (Vettoriale BGE-M3)** | 71 | 36.67 | 71 | 0.6417 | 0.2434 |
| **HYBRID_SEARCH (Ibrido)** | 71 | 36.67 | 71 | 0.6167 | 0.2645 |


---
*Report compilato in automatico il 01 July 2026 dallo script di benchmark.*

---
*Catalogo unificato generato in automatico il 01 July 2026 in data/metrics/report_sperimentale_completo.md.*
