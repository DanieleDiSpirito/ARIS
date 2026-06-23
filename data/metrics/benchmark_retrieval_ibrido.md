# Benchmark Retrieval Ibrido (ARIS)

Questo report riassume le metriche di accuratezza del **Retrieval Ibrido (BM25 + Vettoriale)** per i database estratti con i diversi metodi di parsing, valutati sul test set di domande.

## Tabella Comparativa

| DB Name                      |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:-----------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_docling_locale_700    |               63 |             33.67 |             63 | 0.5683 |            0.0203 |
| chroma_euristico_locale_700  |               82 |             38.67 |             82 | 0.6867 |            0.0206 |
| chroma_llamaparse_locale_700 |               80 |             40.33 |             80 | 0.6817 |            0.0197 |
| chroma_locale_700            |               77 |             54.67 |             77 | 0.7033 |            0.0183 |
| chroma_mineru_locale_700     |               77 |             37    |             77 | 0.6933 |            0.0183 |
| chroma_pdf4llm_locale_700    |               82 |             41.67 |             82 | 0.7217 |            0.0218 |
| chroma_qwen_locale_700       |               74 |             36    |             74 | 0.6267 |            0.0222 |

---
*Report generato automaticamente dallo script `valuta_rag_ibrido.py` il 20 Giugno 2026.*
