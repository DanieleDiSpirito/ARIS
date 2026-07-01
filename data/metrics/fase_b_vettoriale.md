# Benchmark Retrieval Vettoriale (ARIS)

Questo report riassume le metriche di accuratezza del **Retrieval Vettoriale** per i database estratti con i diversi metodi di parsing, valutati sul test set di domande.

## Tabella Comparativa

| DB Name                    |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:---------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_pdf4llm_locale_300  |               69 |             36.67 |             69 | 0.6067 |            0.0298 |
| chroma_pdf4llm_locale_700  |               71 |             36.67 |             71 | 0.6417 |            0.022  |
| chroma_pdf4llm_locale_1000 |               71 |             35.67 |             71 | 0.6267 |            0.0241 |

---
*Report generato automaticamente dallo script `valuta_rag.py` il 20 Giugno 2026.*
