# Analisi del Formato della Knowledge Base e della Pipeline di Estrazione

---

## 1. Cosa fa `document_loader.py` (Estrattore)

Questo script è il **cuore dell'estrazione** dai PDF. Si compone di 3 moduli:

### Modulo 1 — `estrai_elementi_pagina()` (Estrattore Unificato)
- Usa **pdfplumber** per estrarre sia testo che tabelle da ogni pagina
- **Radar Tabelle**: trova tutte le tabelle e le converte in **formato Markdown** (con `|` e `---`)
- **Scudo Anti-Doppioni**: le parole che cadono dentro l'area geometrica di una tabella vengono saltate, per evitare duplicazione
- **Ghigliottina Spaziale**: scarta header (y < 75) e footer (y > 780) della pagina
- Raggruppa le parole per riga (coordinata Y) e le ordina da sinistra a destra
- Calcola metadati per ogni riga: `size_max`, `size_moda`, `font_moda`
- Alla fine ordina tutto (testo + tabelle) dall'alto in basso per mantenere l'ordine di lettura

### Modulo 2 — `classifica_riga()` (Classificatore)
Assegna un **tag semantico** a ogni riga in base a font/dimensione:

| Tag | Logica di riconoscimento |
|---|---|
| `TRIGGER_CAPITOLO` | Bold + size ≥ 19.5 |
| `TRIGGER_PARAGRAFO` | Bold + size 14.5–22.5 |
| `TRIGGER_ERRORE` | Bold + size 11.5–12.5 + match regex `SRVO-xxx`, `SYST-xxx`, ecc. |
| `TESTO_TABELLA` | Font fittizio "TabellaMarkdown" |
| `TESTO_PUNTATO` | Font Mincho |
| `TESTO_Caption` | Bold + size ≤ 9.5 |
| `TESTO_WARNING` | Bold + UPPERCASE + inizia con WARNING/CAUTION/NOTE |
| `TESTO_NORMALE` | Tutto il resto |

### Modulo 3 — `crea_json_definitivo()` (Orchestratore)
- Legge il CSV dei metadati per ottenere `document_id` e pagina iniziale
- Implementa una **macchina a stati**: accumula testo in un buffer e lo salva come record JSON quando:
  - Si cambia pagina
  - Si incontra un TRIGGER (nuovo capitolo, paragrafo, o codice errore)
- Output: **un record JSON per blocco logico** con `document_id`, `file_name`, `page`, `section`, `title`, `text`

---

## 2. Cosa fa `data_cleaner.py` (Preprocessore)

Questo script prende i **JSON singoli** prodotti dal loader e li unifica/pulisce:

### Operazioni eseguite (in ordine):

| Step | Operazione | Dettaglio |
|---|---|---|
| A | Filtro indice | Scarta record con title = "INDEX" o "REVISION RECORD" |
| B | Filtro didascalie brevi | Scarta record con solo `[Caption]:` e < 40 caratteri |
| C | `clean_text_content()` | Rimuove caratteri giapponesi, ricostruisce elenchi, compatta lettere singole, rimuove righe vuote di tabelle markdown, normalizza spazi |
| D | `is_garbage_line()` | Rimuove righe con troppi caratteri speciali rispetto alle lettere (rumore da diagrammi OCR) |
| E | Deduplicazione | Se il testo inizia con lo stesso testo del titolo, lo rimuove |
| F | Fix gerarchia | Corregge sezioni tipo "A.1.2" nei titoli delle appendici |
| G | Mappa allarmi | Unisce testi di allarmi SRVO-xxx che si estendono su più pagine |
| Post | Cross-reference | Se un allarme dice "same actions as SRVO-xxx", inietta il testo dell'allarme referenziato |

### Output finale:
- Un unico file `knowledge_base.json` con **420 record** totali

---

## 3. Valutazione del Formato Attuale

### ✅ Cosa va bene

1. **Struttura base corretta**: I 6 campi (`document_id`, `file_name`, `page`, `section`, `title`, `text`) sono esattamente quelli richiesti da INFO.md
2. **Tabelle in Markdown**: le tabelle con `|` sono leggibili dall'LLM e preservano la struttura
3. **Codici errore SRVO ben estratti**: Explanation + Action sono chiari e completi
4. **Cross-reference allarmi**: funzionalità utile per arricchire il contesto
5. **WARNING/CAUTION/NOTE preservati**: fondamentale per la sicurezza industriale

### ⚠️ Problemi trovati

#### Problema 1 — Rumore da didascalie di diagrammi
Molti record contengono blocchi come questo che sono **puro rumore**:
```
[Caption]: X ) X )
[Caption]: KN ) TT KN )
[Caption]: KN ) M ) X / X /
[Caption]: OLIRRLI
[Caption]: AR LIC 2 ( 1 ( A 1 ( 2 (
```
Queste sono **label di diagrammi OCR** dove pdfplumber legge il testo dei componenti in un'immagine schematica. Non hanno valore semantico per il RAG, anzi **inquinano i chunk** e riducono la qualità del retrieval.

> [!WARNING]
> Il filtro `is_garbage_line()` cattura alcune di queste righe, ma molte passano perché contengono abbastanza lettere (es. "AR LIC 2 ( 1 ( A 1 ( 2 (").

#### Problema 2 — Record con solo didascalie
Alcuni record contengono **esclusivamente didascalie** senza contenuto testuale utile:
```json
{
    "title": "A20B-2103-0170)",
    "text": "[Caption]: (R-30 iB Mate A20B-2005-0150) \n[Caption]: (R-30 iB Mate Plus A20B-2103-0170) \n[Caption]: Fig.4.2 Emergency stop board"
}
```
Questi record sprecano spazio nel vector DB e possono confondere il retrieval.

#### Problema 3 — Testo di connettori/pinout come testo libero
I dati di pinout e cablaggio vengono estratti come testo lineare caotico:
```
JRS27 1 RD (RXDA) 11 SD (TXDA) Honda Tsushin Kogyo 2 SG (0V) 12 SG (0V) CONNECTOR: PCR-E20FS
```
Questo era in origine una tabella nel PDF ma il testo viene "appiattito". Un LLM potrebbe confondersi leggendo questi dati.

#### Problema 4 — Testo invertito/speculare dal Terminal Converter Board
Il record della sezione 4.3.3 contiene testo **specchiato**:
```
1MOCIDS 1VCBT 101ID 201ID 301ID...
```
Questo è un artefatto dell'OCR su un diagramma ruotato/specchiato. È **rumore puro**.

#### Problema 5 — Titoli troncati o sporchi
Alcuni titoli sono incompleti:
- `"title": "A20B-2103-0170)"` — il titolo è un codice componente con parentesi
- `"title": "AND EE INTERFACES"` — manca "PERIPHERAL DEVICE" davanti
- `"title": "Connection between RS232C interface and I/O device"` — OK ma il title originale è più lungo

#### Problema 6 — Record molto lunghi
Alcuni record (es. connections.pdf pagina 168, sezione 4.3.3) hanno testo di **~3000+ caratteri** con tabelle complesse. Questi sono troppo lunghi per chunk efficaci nel RAG.

---

## 4. Raccomandazione: Nuovo Script di Post-Processing

> [!IMPORTANT]
> **NON conviene modificare `document_loader.py`** — fa già un buon lavoro di estrazione base.
> **NON conviene modificare `data_cleaner.py`** — la logica di pulizia e merge è corretta.
>
> **Conviene creare un nuovo script** `prepare_for_rag.py` che prende la knowledge_base.json e la trasforma nel formato ottimale per il chunking + embedding.

### Perché un nuovo script separato?

1. **Separazione delle responsabilità**: estrazione → pulizia → preparazione RAG sono 3 fasi distinte
2. **Non si rischiano regressioni** negli script che funzionano già
3. **Iterabilità**: puoi modificare la preparazione RAG senza riestrarre 8 PDF ogni volta
4. I filtri aggressivi (es. eliminare record rumorosi) non servono per l'estrazione base ma servono per il RAG

### Cosa dovrebbe fare `prepare_for_rag.py`:

| Step | Operazione |
|---|---|
| 1 | Caricare `knowledge_base.json` |
| 2 | **Filtrare record rumorosi**: eliminare record dove il testo è composto per >60% da didascalie |
| 3 | **Filtrare testo invertito/speculare**: eliminare blocchi con pattern tipo `1MOCIDS`, `DLOHX`, `TESER` |
| 4 | **Pulire didascalie OCR rumorose**: rimuovere righe `[Caption]:` che contengono solo sigle/parentesi frammentate |
| 5 | **Mantenere didascalie utili**: `[Caption]: Fig.4.2 Emergency stop board` è utile |
| 6 | **Aggiungere campo `document_type`**: "alarm", "procedure", "specification", "connection", ecc. (utile per metadata filtering nel vector DB) |
| 7 | **Aggiungere campo `content_type`**: "error_code", "table", "procedure", "description", "wiring" |
| 8 | Salvare `knowledge_base_rag.json` — file unico pronto per il chunking |

### Struttura output proposta:

```json
{
    "chunk_id": "04_SRVO-023_p43",
    "document_id": "04",
    "file_name": "troubleshooting_alarms.pdf",
    "page": "43",
    "section": "3.5",
    "title": "SRVO-023 Stop error excess (G:i A:j)",
    "text": "(Explanation) ...",
    "document_type": "alarm_codes",
    "content_type": "error_code"
}
```

---

## 5. Conviene avere un file unico senza rumore?

> [!TIP]
> **Sì, assolutamente.** Per il RAG serve un unico file "pulito" dove:
> - Ogni record ha un contenuto **semanticamente significativo**
> - Non ci sono record con solo didascalie di diagrammi
> - Non c'è testo invertito/specchiato
> - I metadati sono arricchiti per il filtering

La `knowledge_base.json` attuale (420 record) dopo la pulizia aggressiva dovrebbe scendere a circa **350–380 record**, eliminando quelli che sono puro rumore visuale.

---

## 6. Riepilogo

| Domanda | Risposta |
|---|---|
| Il formato attuale va bene? | La struttura base (6 campi) è corretta, ma c'è rumore da pulire |
| Modificare lo script di estrazione? | **No**, `document_loader.py` funziona bene |
| Modificare `data_cleaner.py`? | **No**, fa il suo lavoro di pulizia base |
| Creare un nuovo script? | **Sì**, un `prepare_for_rag.py` che filtra il rumore e arricchisce i metadati |
| File unico senza rumore? | **Sì**, `knowledge_base_rag.json` — pronto per il chunking |
