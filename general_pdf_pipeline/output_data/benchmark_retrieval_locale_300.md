# Benchmark di Retrieval Generico (RAG)

Questo report riassume le metriche di accuratezza del **Retrieval RAG** (Vector, BM25, Ibrido) calcolate sul manuale di test.

## Parametri di Esecuzione
- **Ambiente**: LOCALE
- **Dimensione Chunk**: 300 token
- **Numero Domande**: 3
- **Top-k selezionati**: 2
- **Tolleranza pagina**: ±0

## Tabella Comparativa

| Strategia     |   Hit Rate@k (%) |   Precision@k (%) |   Recall@k (%) |   MRR |   Tempo Medio (s) |   Tempo Totale (s) |
|:--------------|-----------------:|------------------:|---------------:|------:|------------------:|-------------------:|
| PURE_VECTOR   |              100 |               100 |            100 |     1 |            0.1401 |               0.43 |
| PURE_BM25     |              100 |               100 |            100 |     1 |            0.0002 |               0    |
| HYBRID_SEARCH |              100 |               100 |            100 |     1 |            0.0173 |               0.05 |

---
*Report generato automaticamente il 22-06-2026 13:47:03.*
