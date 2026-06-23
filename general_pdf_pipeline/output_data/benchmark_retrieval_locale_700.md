# Benchmark di Retrieval Generico (RAG)

Questo report riassume le metriche di accuratezza del **Retrieval RAG** (Vector, BM25, Ibrido) calcolate sul manuale di test.

## Parametri di Esecuzione
- **Ambiente**: LOCALE
- **Dimensione Chunk**: 700 token
- **Numero Domande**: 9
- **Top-k selezionati**: 3
- **Tolleranza pagina**: ±1

## Tabella Comparativa

| Strategia     |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |    MRR |   Tempo Medio (s) |   Tempo Totale (s) |
|:--------------|-----------------:|------------------:|---------------:|-------:|------------------:|-------------------:|
| PURE_VECTOR   |              100 |             66.67 |            100 | 0.9444 |            0.0572 |               0.52 |
| PURE_BM25     |              100 |             62.96 |            100 | 1      |            0.0005 |               0.01 |
| HYBRID_SEARCH |              100 |             74.07 |            100 | 1      |            0.0166 |               0.15 |

---
*Report generato automaticamente il 22-06-2026 15:33:56.*
