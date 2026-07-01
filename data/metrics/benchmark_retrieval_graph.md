# Benchmark Retrieval GraphRAG (ARIS)

Questo report riassume le metriche di accuratezza del **Retrieval con Grafo Relazionale (GraphRAG)** per i database estratti con i diversi metodi di parsing, valutati sul test set di domande.

## Tabella Comparativa

| DB Name                    |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |
|:---------------------------|-----------------:|------------------:|---------------:|-------:|------------------:|
| chroma_euristico_cloud_700 |               72 |              26.6 |             72 | 0.5812 |            7.5424 |
| chroma_pdf4llm_cloud_700   |               72 |              29.2 |             72 | 0.5862 |            7.844  |

---
*Report generato automaticamente dallo script `valuta_rag_graph.py`.*
