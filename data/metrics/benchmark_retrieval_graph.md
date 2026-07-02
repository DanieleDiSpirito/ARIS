# Benchmark Retrieval GraphRAG (ARIS)

Questo report riassume le metriche di accuratezza del **Retrieval con Grafo Relazionale (GraphRAG)** per i database estratti con i diversi metodi di parsing, valutati sul test set di domande.

## Tabella Comparativa

| DB Name                    |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:---------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_euristico_cloud_700 |               79 |              29.2 |             79 | 0.719  |            2.9036 |
| chroma_pdf4llm_cloud_700   |               83 |              33.6 |             83 | 0.7492 |            3.039  |

---
*Report generato automaticamente dallo script `valuta_rag_graph.py`.*
