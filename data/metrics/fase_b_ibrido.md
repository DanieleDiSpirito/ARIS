# Benchmark Retrieval Ibrido (ARIS)

Questo report riassume le metriche di accuratezza del **Retrieval Ibrido (BM25 + Vettoriale)** per i database estratti con i diversi metodi di parsing, valutati sul test set di domande.

## Tabella Comparativa

| DB Name                    |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:---------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_pdf4llm_locale_300  |               68 |             37    |             68 | 0.6017 |            0.0266 |
| chroma_pdf4llm_locale_700  |               71 |             36.67 |             71 | 0.6167 |            0.0232 |
| chroma_pdf4llm_locale_1000 |               70 |             36    |             70 | 0.61   |            0.0263 |

---
*Report generato automaticamente dallo script `valuta_rag_ibrido.py` il 20 Giugno 2026.*
