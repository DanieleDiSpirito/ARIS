# Catalogo Sperimentale Completo dei Test e delle Metriche (Capitolo 4)

Questo catalogo raccoglie e unifica ogni singolo dato, telemetria ed analisi quantitativa o qualitativa prodotta durante gli esperimenti del **Capitolo 4** per la tesi di laurea ARIS (focalizzato sulla parte di Ingestione, Indicizzazione e Retrieval dello Studente 1).

---

## 🛠️ SEZIONE 1: Benchmark Hardware e Consumo Risorse (Scaling)

I dati seguenti registrano il tempo di esecuzione ed il vero picco assoluto di RAM e VRAM del processo di caricamento ed estrazione per ciascun parser, valutati su documenti di lunghezza differente per misurarne la scalabilità (misurazioni effettuate con il nuovo metodo di tracciamento Peak Working Set).

| Parser     | File                                  |   Tempo (s) |   RAM (MB) |   VRAM (MB) | Tipo di Esecuzione                 |
|:-----------|:--------------------------------------|------------:|-----------:|------------:|:-----------------------------------|
| Docling    | `checks_maintenance.pdf` (1 pag)      |        3.6  |    2064.26 |      442.82 | Locale (GPU CUDA PyTorch)          |
| Docling    | `overview_configuration.pdf` (33 pag) |        5.42 |    2433.29 |      774.32 | Locale (GPU CUDA PyTorch)          |
| Docling    | `safety_precautions.pdf` (13 pag)     |        6.79 |    2765.77 |     1189.7  | Locale (GPU CUDA PyTorch)          |
| Docling    | `test_telemetry.pdf` (1 pag)          |        1.66 |    2765.77 |     1006.36 | Locale (GPU CUDA PyTorch)          |
| Llamaparse | `checks_maintenance.pdf` (1 pag)      |        3.64 |     925.14 |        0    | Cloud API                          |
| Llamaparse | `overview_configuration.pdf` (33 pag) |       31.3  |     925.14 |        0    | Cloud API                          |
| Llamaparse | `safety_precautions.pdf` (13 pag)     |       41.03 |     925.14 |        0    | Cloud API                          |
| Llamaparse | `test_telemetry.pdf` (1 pag)          |        5.25 |     925.14 |        0    | Cloud API                          |
| Mineru     | `checks_maintenance.pdf` (1 pag)      |       44.37 |     503.1  |        0    | Locale CLI (magic-pdf)             |
| Mineru     | `overview_configuration.pdf` (33 pag) |      121.5  |     503.1  |        0    | Locale CLI (magic-pdf)             |
| Mineru     | `safety_precautions.pdf` (13 pag)     |       36.11 |     503.1  |        0    | Locale CLI (magic-pdf)             |
| Mineru     | `test_telemetry.pdf` (1 pag)          |       17.98 |     503.1  |        0    | Locale CLI (magic-pdf)             |
| Pdf4llm    | `checks_maintenance.pdf` (1 pag)      |        0.2  |     770.42 |        0    | Locale (CPU PyMuPDF)               |
| Pdf4llm    | `overview_configuration.pdf` (33 pag) |        3.9  |     855.2  |        0    | Locale (CPU PyMuPDF)               |
| Pdf4llm    | `safety_precautions.pdf` (13 pag)     |        3.67 |     876.36 |        0    | Locale (CPU PyMuPDF)               |
| Pdf4llm    | `test_telemetry.pdf` (1 pag)          |        0.17 |     876.36 |        0    | Locale (CPU PyMuPDF)               |
| Qwen       | `checks_maintenance.pdf` (1 pag)      |       13.19 |     904.04 |        0    | Locale GPU (Transformers Qwen2-VL) |
| Qwen       | `overview_configuration.pdf` (33 pag) |      100    |     925.14 |        0    | Locale GPU (Transformers Qwen2-VL) |
| Qwen       | `safety_precautions.pdf` (13 pag)     |      251.72 |     908.54 |        0    | Locale GPU (Transformers Qwen2-VL) |
| Qwen       | `test_telemetry.pdf` (1 pag)          |       18.52 |     901.6  |        0    | Locale GPU (Transformers Qwen2-VL) |

---

## 📊 SEZIONE 2: Analisi Qualitativa per Tipologia di Layout

### Categoria: Testo Lineare (Narrativo / Istruzioni di Sicurezza)
**File di riferimento:** `safety_precautions.pdf`

| Metodo     |   Numero Chunk |   Garbage Ratio (%) |   Spazi Anomali (%) |   Righe Orfane (%) |   Chunk con Tabelle (%) |   Codici Allarme Rilevati |
|:-----------|---------------:|--------------------:|--------------------:|-------------------:|------------------------:|--------------------------:|
| docling    |             42 |                0    |                0.05 |               9.71 |                    2.38 |                         7 |
| qwen       |             36 |                0.02 |                0    |              10.55 |                    2.78 |                         6 |
| llamaparse |             38 |                0.03 |                0    |              17.3  |                    2.63 |                         7 |
| pdf4llm    |             40 |                0.03 |                0    |              14.82 |                    2.5  |                         7 |
| mineru     |             26 |                0.08 |                0    |               2.48 |                    3.85 |                         8 |
| euristico  |             60 |                0.11 |                0    |               6.09 |                    3.33 |                        20 |

### Categoria: Layout Strutturato (Tabelle Pinout / Schemi di Connessione)
**File di riferimento:** `connections.pdf`

| Metodo     |   Numero Chunk |   Garbage Ratio (%) |   Spazi Anomali (%) |   Righe Orfane (%) |   Chunk con Tabelle (%) |   Codici Allarme Rilevati |
|:-----------|---------------:|--------------------:|--------------------:|-------------------:|------------------------:|--------------------------:|
| qwen       |            394 |                0.04 |                0    |              12.94 |                    1.52 |                         5 |
| docling    |            106 |                0.05 |                0.01 |              15.01 |                    3.77 |                         3 |
| pdf4llm    |            281 |                0.06 |                0    |               8.16 |                    1.78 |                         6 |
| euristico  |            396 |                0.08 |                0    |               7.13 |                    2.02 |                        14 |
| llamaparse |            295 |                0.09 |                0    |              10.1  |                    1.36 |                         6 |
| mineru     |            179 |                0.12 |                0    |               7.61 |                    1.12 |                         6 |

### Categoria: Layout Tecnico (Troubleshooting / Codici Allarme)
**File di riferimento:** `troubleshooting_alarms.pdf`

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

Le tabelle seguenti sintetizzano l'efficacia del retrieval semantico sul test set di 100 domande in lingua italiana per le fasi sperimentali di baseline.

### FASE A: Confronto dei Motori di Parsing (Baseline: 700 token, Locale BGE-M3)

#### 1. Retrieval Vettoriale Puro (Pure Vector)
| DB Name                      |   Hit Rate@3 (%) |   Precision@3 (%) |   Recall@3 (%) |    MRR |   Tempo Medio (s) |
|:-----------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_docling_locale_700    |               53 |             29.33 |             53 | 0.4783 |            0.025  |
| chroma_euristico_locale_700  |               68 |             47.33 |             68 | 0.6067 |            0.0224 |
| chroma_llamaparse_locale_700 |               67 |             32.33 |             67 | 0.595  |            0.0242 |
| chroma_locale_700            |               68 |             47.33 |             68 | 0.6067 |            0.2587 |
| chroma_mineru_locale_700     |               65 |             28.67 |             65 | 0.5683 |            0.2276 |
| chroma_pdf4llm_locale_700    |               71 |             36.67 |             71 | 0.6417 |            0.2434 |
| chroma_qwen_locale_700       |               60 |             27.67 |             60 | 0.535  |            0.2041 |

#### 2. Retrieval Ibrido (Hybrid Search)
| DB Name                      |   Hit Rate@3 (%) |   Precision@3 (%) |   Recall@3 (%) |    MRR |   Tempo Medio (s) |
|:-----------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_docling_locale_700    |               55 |             30.67 |             55 | 0.4833 |            0.0338 |
| chroma_euristico_locale_700  |               69 |             31    |             69 | 0.6217 |            0.0274 |
| chroma_llamaparse_locale_700 |               67 |             32.67 |             67 | 0.57   |            0.0266 |
| chroma_locale_700            |               68 |             47.33 |             68 | 0.6067 |            0.2451 |
| chroma_mineru_locale_700     |               65 |             30    |             65 | 0.5617 |            0.2694 |
| chroma_pdf4llm_locale_700    |               71 |             36.67 |             71 | 0.6167 |            0.2645 |
| chroma_qwen_locale_700       |               59 |             26.67 |             59 | 0.5067 |            0.0294 |

---

### FASE B: Analisi di Sensibilità al Chunk Size (Parser: pdf4llm, Locale BGE-M3)

#### 1. Retrieval Vettoriale Puro
| DB Name                    |   Hit Rate@3 (%) |   Precision@3 (%) |   Recall@3 (%) |    MRR |   Tempo Medio (s) |
|:---------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_pdf4llm_locale_300  |               69 |             36.67 |             69 | 0.6067 |            0.0298 |
| chroma_pdf4llm_locale_700  |               71 |             36.67 |             71 | 0.6417 |            0.022  |
| chroma_pdf4llm_locale_1000 |               71 |             35.67 |             71 | 0.6267 |            0.0241 |

#### 2. Retrieval Ibrido
| DB Name                    |   Hit Rate@3 (%) |   Precision@3 (%) |   Recall@3 (%) |    MRR |   Tempo Medio (s) |
|:---------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_pdf4llm_locale_300  |               68 |             37    |             68 | 0.6017 |            0.0266 |
| chroma_pdf4llm_locale_700  |               71 |             36.67 |             71 | 0.6167 |            0.0232 |
| chroma_pdf4llm_locale_1000 |               70 |             36    |             70 | 0.61   |            0.0263 |

---

### FASE C: Impatto del Modello di Embedding (Locale vs Cloud, pdf4llm, 700 token)

#### 1. Retrieval Vettoriale Puro
| DB Name                  |   Hit Rate@3 (%) |   Precision@3 (%) |   Recall@3 (%) |    MRR |   Tempo Medio (s) |
|:-------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_pdf4llm_locale_700|               71 |             36.67 |             71 | 0.6417 |            0.022  |
| chroma_pdf4llm_cloud_700 |               68 |             34.67 |             68 | 0.5733 |            0.3472 |

#### 2. Retrieval Ibrido
| DB Name                  |   Hit Rate@3 (%) |   Precision@3 (%) |   Recall@3 (%) |    MRR |   Tempo Medio (s) |
|:-------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_pdf4llm_locale_700|               71 |             36.67 |             71 | 0.6167 |            0.0232 |
| chroma_pdf4llm_cloud_700 |               68 |                35 |             68 | 0.5867 |            0.2077 |

---

### FASE D: Confronto delle Strategie di Ricerca (pdf4llm, 700 token, Locale BGE-M3)

| Strategia | Hit Rate@3 (%) | Precision@3 (%) | Recall@3 (%) | MRR | Tempo Medio (s) |
|:---|---:|---:|---:|---:|---:|
| **PURE_BM25 (Lessicale)** | 30.0 | 13.0 | 30.0 | 0.2200 | 0.0015 |
| **PURE_VECTOR (Vettoriale BGE-M3)** | 71.0 | 36.67 | 71.0 | 0.6417 | 0.0220 |
| **HYBRID_SEARCH (Ibrido)** | 71.0 | 36.67 | 71.0 | 0.6167 | 0.0232 |

---

## 🔬 SEZIONE 4: Strategie di Retrieval Avanzate (Reranking)

I dati seguenti confrontano l'efficacia del **Re-ranking (Cross-Encoder)** applicato su database estratti con i diversi metodi di parsing (dimensione chunk fissa a 700 token).

### FASE E: Valutazione Reranking (Locale vs Cloud, k=3)

#### 1. Reranking su DB Cloud (text-embedding-3-small + BM25 + Cross-Encoder)
| DB Name                    |   Hit Rate@3 (%) |   Precision@3 (%) |   Recall@3 (%) |    MRR |   Tempo Medio (s) |
|:---------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_euristico_cloud_700 |               75 |              25.8 |             75 | 0.6427 |            1.6356 |
| chroma_pdf4llm_cloud_700   |               76 |              27.2 |             76 | 0.6373 |            3.0101 |

#### 2. Reranking su DB Locali (BGE-M3 + BM25 + Cross-Encoder)
| DB Name                      |   Hit Rate@3 (%) |   Precision@3 (%) |   Recall@3 (%) |    MRR |   Tempo Medio (s) |
|:-----------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_euristico_locale_700  |               69 |             34    |             69 | 0.635  |            0.0658 |
| chroma_pdf4llm_locale_700    |               73 |             37.67 |             73 | 0.6383 |            0.087  |
| chroma_docling_locale_700    |               53 |             27.67 |             53 | 0.4633 |            0.0887 |
| chroma_llamaparse_locale_700 |               68 |             33    |             68 | 0.5833 |            0.0972 |
| chroma_mineru_locale_700     |               69 |             30.33 |             69 | 0.605  |            0.0925 |
| chroma_qwen_locale_700       |               57 |             26.67 |             57 | 0.5183 |            0.084  |

---

## 🌎 SEZIONE 5: Valutazione Multilingua (Test Set Inglese - EN)

Per verificare le prestazioni in un contesto monolingua nativo, le strategie di retrieval sono state valutate anche sul test set composto da 30 domande formulate direttamente in lingua inglese (dimensione chunk fissa a 700 token).

### FASE F: Benchmark di Retrieval in Lingua Inglese (30 domande, k=3)

| DB Name                     | Strategia   |   Hit Rate@3 (%) |   Precision@3 (%) |   Recall@3 (%) |    MRR |   Tempo Medio (s) |
|:----------------------------|:------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_pdf4llm_locale_700   | Vector      |               75 |             38    |             75 | 0.6667 |            0.0374 |
| chroma_pdf4llm_locale_700   | Hybrid      |               75 |             37.67 |             75 | 0.6833 |            0.0469 |
| chroma_pdf4llm_locale_700   | Rerank      |               79 |             41.33 |             79 | 0.7183 |            0.1074 |
| chroma_pdf4llm_cloud_700    | Vector      |               73 |             38.67 |             73 | 0.6417 |            0.4277 |
| chroma_pdf4llm_cloud_700    | Hybrid      |               72 |             39    |             72 | 0.6483 |            0.3674 |
| chroma_pdf4llm_cloud_700    | Rerank      |               83 |             43.33 |             83 | 0.75   |            0.5424 |
| chroma_euristico_locale_700 | Vector      |               69 |             49    |             69 | 0.6183 |            0.0199 |
| chroma_euristico_locale_700 | Hybrid      |               73 |             34    |             73 | 0.6917 |            0.0241 |
| chroma_euristico_locale_700 | Rerank      |               78 |             37.33 |             78 | 0.715  |            0.0622 |

---
*Catalogo unificato generato in automatico il 02 July 2026 in data/metrics/report_sperimentale_completo.md.*
