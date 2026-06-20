# Benchmark Retrieval Ibrido (ARIS)

Questo report riassume le metriche di accuratezza del **Retrieval Ibrido (BM25 + Vettoriale)** per i database estratti con i diversi metodi di parsing, valutati sul test set di domande.

## Tabella Comparativa

| DB Name                      |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:-----------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_docling_locale_700    |               62 |             33.67 |             62 | 0.5683 |            0.0299 |
| chroma_euristico_locale_700  |               82 |             38.67 |             82 | 0.6933 |            0.0272 |
| chroma_llamaparse_locale_700 |               80 |             40.33 |             80 | 0.6817 |            0.0243 |
| chroma_locale_700            |               77 |             54.67 |             77 | 0.7033 |            0.2276 |
| chroma_pdf4llm_locale_700    |               82 |             41.67 |             82 | 0.7217 |            0.2212 |
| chroma_qwen_locale_700       |               74 |             36    |             74 | 0.6267 |            0.2407 |

---
*Report generato automaticamente dallo script `valuta_rag_ibrido.py` il 20 Giugno 2026.*
