# Benchmark Retrieval Vettoriale (ARIS)

Questo report riassume le metriche di accuratezza del **Retrieval Vettoriale** per i database estratti con i diversi metodi di parsing, valutati sul test set di domande.

## Tabella Comparativa

| DB Name                  |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:-------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_mineru_locale_300 |               78 |                40 |             78 | 0.715  |            0.0227 |
| chroma_mineru_locale_700 |               77 |                37 |             77 | 0.7067 |            0.0219 |

---
*Report generato automaticamente dallo script `valuta_rag.py` il 20 Giugno 2026.*
