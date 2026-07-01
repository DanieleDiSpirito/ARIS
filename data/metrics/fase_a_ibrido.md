# Benchmark Retrieval Ibrido (ARIS)

Questo report riassume le metriche di accuratezza del **Retrieval Ibrido (BM25 + Vettoriale)** per i database estratti con i diversi metodi di parsing, valutati sul test set di domande.

## Tabella Comparativa

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
*Report generato automaticamente dallo script `valuta_rag_ibrido.py` il 20 Giugno 2026.*
