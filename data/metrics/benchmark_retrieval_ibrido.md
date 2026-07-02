# Benchmark Retrieval Ibrido (ARIS)

Questo report riassume le metriche di accuratezza del **Retrieval Ibrido (BM25 + Vettoriale)** per i database estratti con i diversi metodi di parsing, valutati sul test set di domande.

## Tabella Comparativa

| DB Name                  |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:-------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_pdf4llm_cloud_700 |               68 |                35 |             68 | 0.5867 |            0.2077 |

---
*Report generato automaticamente dallo script `valuta_rag_ibrido.py`.*
