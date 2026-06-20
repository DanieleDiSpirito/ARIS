# Benchmark Retrieval Vettoriale (ARIS)

Questo report riassume le metriche di accuratezza del **Retrieval Vettoriale** per i database estratti con i diversi metodi di parsing, valutati sul test set di domande.

## Tabella Comparativa

| DB Name                      |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:-----------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_docling_locale_700    |               62 |             32.33 |             62 | 0.5717 |            0.0246 |
| chroma_euristico_locale_700  |               76 |             54    |             76 | 0.6933 |            0.0243 |
| chroma_llamaparse_locale_700 |               79 |             39.67 |             79 | 0.695  |            0.0219 |
| chroma_locale_700            |               77 |             54.67 |             77 | 0.7033 |            0.2193 |
| chroma_pdf4llm_locale_700    |               82 |             40.67 |             82 | 0.7433 |            0.2497 |
| chroma_qwen_locale_700       |               72 |             36    |             72 | 0.64   |            0.2558 |

---
*Report generato automaticamente dallo script `valuta_rag.py` il 20 Giugno 2026.*
