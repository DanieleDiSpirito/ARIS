# Benchmark Qualitativo Dettagliato per Tipologia di Layout (ARIS)

Questo report suddivide l'analisi di pulizia e qualità dei chunk in base al **tipo di layout del documento originale** per evidenziare i punti di forza e debolezza di ciascun parser.

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

