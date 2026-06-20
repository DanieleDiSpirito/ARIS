# Benchmark di Qualità e Pulizia del Testo dei Chunk (ARIS)

Questo report mette a confronto la pulizia e la qualità testuale dei chunk generati dai **5 diversi metodi di parsing** estratti a **700 token** di dimensione target.

## Metriche di Valutazione:
1. **Garbage Ratio (%):** Percentuale di caratteri di rumore non standard rispetto ai caratteri totali del chunk.
2. **Spazi Anomali (%):** Percentuale di caratteri dovuti a spaziazioni consecutive eccessive o newline multiple non pulite.
3. **Righe Orfane (%):** Percentuale di righe di intestazione/didascalia orfane o molto corte rispetto alle righe totali del chunk.
4. **Chunk con Tabelle (%):** Percentuale di chunk in cui è stata rilevata una tabella Markdown valida.
5. **Codici Allarme Rilevati:** Conteggio totale di allarmi di tipo tecnico (es. `SRVO-xxx`) preservati nel testo per le query di diagnostica.

## Tabella Comparativa

| Metodo     |   Numero Chunk |   Lunghezza Media (Caratteri) |   Garbage Ratio (%) |   Spazi Anomali (%) |   Righe Orfane (%) |   Chunk con Tabelle (%) |   Codici Allarme Rilevati |
|:-----------|---------------:|------------------------------:|--------------------:|--------------------:|-------------------:|------------------------:|--------------------------:|
| euristico  |            870 |                           687 |                0.06 |                0    |               4.68 |                    1.38 |                       605 |
| docling    |            410 |                           460 |                0.02 |                0.01 |              12.22 |                    1.46 |                       314 |
| llamaparse |            583 |                           517 |                0.07 |                0.08 |              11.81 |                    1.03 |                       271 |
| qwen       |            789 |                           367 |                0.03 |                0    |              17.75 |                    1.01 |                       154 |
| pdf4llm    |            590 |                           528 |                0.04 |                0    |               6.16 |                    1.19 |                       331 |

---
*Report generato automaticamente dallo script `valuta_qualita_chunk.py` il 20 Giugno 2026.*
