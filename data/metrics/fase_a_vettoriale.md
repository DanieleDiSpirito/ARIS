# Benchmark Retrieval Vettoriale (ARIS)

Questo report riassume le metriche di accuratezza del **Retrieval Vettoriale** per i database estratti con i diversi metodi di parsing, valutati sul test set di domande.

## Tabella Comparativa

| DB Name                      |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:-----------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_docling_locale_700    |               53 |             29.33 |             53 | 0.4783 |            0.025  |
| chroma_euristico_locale_700  |               68 |             47.33 |             68 | 0.6067 |            0.0224 |
| chroma_llamaparse_locale_700 |               67 |             32.33 |             67 | 0.595  |            0.0242 |
| chroma_locale_700            |               68 |             47.33 |             68 | 0.6067 |            0.2587 |
| chroma_mineru_locale_700     |               65 |             28.67 |             65 | 0.5683 |            0.2276 |
| chroma_pdf4llm_locale_700    |               71 |             36.67 |             71 | 0.6417 |            0.2434 |
| chroma_qwen_locale_700       |               60 |             27.67 |             60 | 0.535  |            0.2041 |

---
*Report generato automaticamente dallo script `valuta_rag.py` il 20 Giugno 2026.*
