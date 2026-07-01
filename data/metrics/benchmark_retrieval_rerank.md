# Benchmark Retrieval con Re-ranking (ARIS)

Questo report riassume le metriche di accuratezza del **Retrieval con Re-ranking (BM25 + Vettoriale + Cross-Encoder)** per i database estratti con i diversi metodi di parsing, valutati sul test set di domande.

## Tabella Comparativa

| DB Name                    |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:---------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_euristico_cloud_700 |               75 |              25.8 |             75 | 0.6427 |            1.6356 |
| chroma_pdf4llm_cloud_700   |               76 |              27.2 |             76 | 0.6373 |            3.0101 |

---
*Report generato automaticamente dallo script `valuta_rag_rerank.py`.*
