# Guida Tesi — Assistente LLM per Manutenzione Industriale (RAG)

---

## 1. Obiettivo finale della tesi

L'obiettivo non è semplicemente "fare un chatbot".

L'obiettivo corretto è sviluppare un **assistente tecnico intelligente** che:

1. riceve domande in linguaggio naturale da un operatore;
2. cerca informazioni in manuali, schede tecniche, tabelle errori e procedure;
3. recupera i passaggi più rilevanti;
4. genera una risposta chiara, tecnica e verificabile;
5. cita o mostra le fonti documentali usate;
6. suggerisce controlli passo-passo per casi semplici di manutenzione o diagnostica.

**Esempio di domanda:**
> "La macchina segnala errore E42. Cosa significa e quali controlli devo fare?"

**Risposta attesa dal sistema:**
> "L'errore E42 indica un'anomalia del sensore di pressione. Controllare: 1) connessione del sensore, 2) pressione nominale, 3) eventuale ostruzione del circuito, 4) reset secondo procedura indicata nel manuale. Fonte: Manuale manutenzione, sezione 7.3."

---

## 2. Divisione dei ruoli tra i due studenti

### Studente 1 — Knowledge Base, documenti e RAG

Responsabile di:
- raccolta documentazione tecnica;
- analisi dei manuali;
- preprocessing dei documenti;
- suddivisione in chunk;
- creazione embeddings;
- costruzione vector database;
- valutazione del recupero delle informazioni.

> **In pratica:** "Il sistema trova davvero le informazioni giuste nei documenti?"

### Studente 2 — LLM, interfaccia e validazione

Responsabile di:
- integrazione con LLM;
- progettazione del prompt;
- sviluppo dell'interfaccia utente;
- logica domanda-risposta;
- gestione dei casi d'uso;
- test con scenari realistici;
- confronto con consultazione manuale tradizionale.

> **In pratica:** "Il sistema risponde bene, in modo utile e usabile per un manutentore?"

---

## 3. Architettura generale del sistema

Il sistema segue una classica architettura **Retrieval-Augmented Generation (RAG)**:

```
Documenti tecnici
       ↓
Preprocessing
       ↓
Chunking
       ↓
Embedding
       ↓
Vector Database
       ↓
Query utente
       ↓
Retrieval dei documenti rilevanti
       ↓
Prompt + contesto recuperato
       ↓
LLM
       ↓
Risposta tecnica con fonti
       ↓
Interfaccia utente
```

> Il modello LLM **non deve rispondere "a memoria"**. Deve rispondere usando il contenuto recuperato dai documenti tecnici.

---

## 4. Fase 1 — Scelta del macchinario o sistema industriale

### Obiettivo
Scegliere un caso applicativo realistico.

### Criteri di scelta
Il sistema scelto deve avere:
- manuale utente;
- manuale manutenzione;
- codici di errore/allarme;
- procedure di avvio, arresto e reset;
- schemi o tabelle tecniche;
- eventuali FAQ o troubleshooting guide.

### Opzioni consigliate

**Opzione A — Impianto HVAC** *(difficoltà: bassa-media)*
- Vantaggi: documentazione spesso disponibile, codici errore chiari, manutenzione comprensibile, scenari di test facili da costruire.

**Opzione B — Robot industriale** *(difficoltà: media-alta)*
- Vantaggi: caso molto interessante, forte rilevanza industriale, troubleshooting realistico.
- Svantaggi: documenti più complessi, terminologia tecnica più difficile.

**Opzione C — Macchina utensile CNC** *(difficoltà: alta)*
- Vantaggi: ottima per automazione industriale, molti codici errore, forte valore applicativo.
- Svantaggi: documentazione lunga, procedure più delicate, serve maggiore attenzione per evitare risposte rischiose.

> **Consiglio:** per una tesi ben gestibile, scegliere impianto HVAC o robot industriale semplice.

---

## 5. Fase 2 — Raccolta e organizzazione della documentazione

### Obiettivo
Costruire un dataset documentale ordinato.

### Struttura cartelle

```
dataset/
├── manuali_utente/
│   └── manuale_utente.pdf
├── manuali_manutenzione/
│   └── manuale_manutenzione.pdf
├── codici_errore/
│   └── tabella_errori.pdf
├── procedure/
│   └── troubleshooting.pdf
├── schede_tecniche/
│   └── datasheet_componenti.pdf
└── metadata/
    └── document_index.csv
```

### File `document_index.csv`
Serve per dare un indice all'LLM ed evitare che legga tutti i documenti ogni volta.

Colonne richieste: `id_documento, nome_file, tipo_documento, macchina, versione, lingua, note`

Esempio:
```
DOC001, manuale_hvac.pdf, manuale_manutenzione, HVAC_X100, v2.1, IT, manuale principale
DOC002, error_codes.pdf, tabella_errori, HVAC_X100, v2.1, EN, codici errore
```

### Output della fase
- cartella documentale ordinata;
- elenco dei documenti usati;
- breve descrizione del macchinario;
- tabella con tipologia, formato, lingua e contenuto dei documenti.

---

## 6. Fase 3 — Analisi tecnica della documentazione

### Obiettivo
Capire che tipo di informazioni contiene la documentazione.

### Categorie da individuare
1. descrizione generale della macchina;
2. componenti principali;
3. procedure operative;
4. procedure di manutenzione ordinaria;
5. procedure di manutenzione straordinaria;
6. codici errore;
7. allarmi;
8. troubleshooting;
9. specifiche tecniche;
10. limiti di funzionamento;
11. avvertenze di sicurezza.

### Esempio di tabella di analisi

| Sezione documento | Tipo informazione | Utilità per il chatbot | Priorità |
|---|---|---|---|
| 7.3 Error Codes | Codici errore | Alta | Alta |
| 5.1 Startup | Procedura avvio | Media | Media |
| 9.2 Safety | Sicurezza | Alta | Alta |

> **Nota importante:** questa fase è fondamentale. Un sistema RAG funziona bene solo se la conoscenza di partenza è ordinata, pulita e ben segmentata.

---

## 7. Fase 4 — Estrazione del testo dai documenti

### Obiettivo
Convertire PDF e documenti tecnici in testo utilizzabile.

### Strumenti consigliati (Python)
- `pypdf`
- `pdfplumber`
- `pymupdf`
- `unstructured`
- `pytesseract` (solo se i PDF sono scannerizzati)

### Procedura
1. caricare ogni PDF;
2. estrarre il testo pagina per pagina;
3. salvare il testo estratto;
4. mantenere riferimento a: nome documento, numero pagina, sezione, titolo della sezione.

### Struttura consigliata del testo estratto
```json
{
  "document_id": "DOC001",
  "file_name": "manuale_hvac.pdf",
  "page": 42,
  "section": "7.3",
  "title": "Error Codes",
  "text": "E42 indicates pressure sensor anomaly..."
}
```

### Output della fase
- script Python di estrazione;
- file JSON o CSV con testo estratto;
- controllo qualitativo su almeno 10 pagine campione.

---

## 8. Fase 5 — Pulizia e preprocessing

### Obiettivo
Pulire il testo eliminando rumore inutile.

### Operazioni da fare
1. rimuovere intestazioni e piè di pagina ripetuti;
2. eliminare numeri pagina isolati;
3. correggere interruzioni di riga;
4. unire frasi spezzate;
5. preservare tabelle importanti;
6. mantenere codici errore e sigle tecniche;
7. **non** eliminare unità di misura;
8. **non** tradurre automaticamente senza controllo.

### Esempio
Testo grezzo:
```
Error
Code
E42
Pressure sensor
fault
See section
7.3
```

Testo pulito:
```
Error code E42: Pressure sensor fault. See section 7.3.
```

> **Attenzione:** non bisogna pulire troppo. Simboli, sigle, codici e unità di misura sono informazioni importanti. Eliminare "E42", "bar", "V", "Hz", "PLC", "I/O" sarebbe un errore grave.

---

## 9. Fase 6 — Chunking dei documenti

### Obiettivo
Dividere il testo in blocchi interrogabili.

### Strategia consigliata
- chunk da **500–1000 token**;
- overlap di **100–150 token**;
- divisione preferibilmente **per sezioni**, non solo per lunghezza.

### Perché non usare chunk troppo piccoli
Chunk troppo piccoli perdono contesto.

Esempio negativo: `E42: pressure sensor fault.` — il sistema potrebbe non sapere cosa fare.

### Perché non usare chunk troppo grandi
Chunk troppo grandi recuperano troppo rumore.

Esempio negativo: tutta la sezione 7 del manuale lunga 20 pagine — il sistema potrebbe confondersi.

### Formato di ogni chunk
```json
{
  "chunk_id": "DOC001_CHUNK_0042",
  "document_id": "DOC001",
  "page_start": 40,
  "page_end": 41,
  "section": "7.3",
  "title": "Pressure sensor errors",
  "text": "...",
  "machine": "HVAC_X100",
  "document_type": "maintenance_manual"
}
```

### Output della fase
- script di chunking;
- dataset di chunk;
- tabella con numero di chunk per documento;
- esempi di chunk buoni e chunk problematici.

---

## 10. Fase 7 — Creazione degli embeddings

### Obiettivo
Trasformare i chunk in vettori numerici per la ricerca semantica.

### Soluzione cloud
- OpenAI embeddings
- Cohere embeddings
- Voyage embeddings
- Riferimento modelli: https://openrouter.ai/models

Difficoltà: **bassa** — alta qualità, semplice da implementare. Svantaggi: dipendenza da API esterne, possibili problemi di privacy.

### Soluzione locale
- `sentence-transformers`
- `BAAI/bge-small-en`
- `intfloat/multilingual-e5-base`
- modelli embedding locali tramite Ollama

Difficoltà: **media** — maggiore privacy, coerente con contesto industriale. Svantaggi: prestazioni da valutare, configurazione più complessa.

### Scelta consigliata
Per una tesi in automazione industriale, confrontare almeno:
1. un modello embedding cloud;
2. un modello embedding locale.

---

## 11. Fase 8 — Vector Database

### Obiettivo
Salvare gli embeddings e fare retrieval.

### Strumenti possibili
- FAISS
- ChromaDB
- Qdrant
- Weaviate
- Milvus

### Scelta consigliata per un prototipo universitario
- **FAISS** — ottimo per semplicità e prestazioni.
- **ChromaDB** — comodo per gestire metadata e prototipazione rapida.

### Operazioni da implementare
1. creare embeddings per tutti i chunk;
2. salvarli nel vector database;
3. implementare una funzione di ricerca top-k;
4. restituire i chunk più simili alla domanda;
5. mostrare anche documento, pagina e sezione.

### Esempio di output retrieval
```
Domanda: Cosa significa errore E42?

1. DOC002, pagina 12, sezione Error Codes, score 0.89
2. DOC001, pagina 45, sezione Troubleshooting, score 0.82
3. DOC001, pagina 46, sezione Sensor Maintenance, score 0.77
```

---

## 12. Fase 9 — Sviluppo del modulo RAG

### Obiettivo
Collegare retrieval e LLM.

### Pipeline RAG
```
Input utente
       ↓
Pulizia query
       ↓
Embedding query
       ↓
Ricerca top-k nel vector DB
       ↓
Recupero chunk rilevanti
       ↓
Costruzione prompt
       ↓
Invio al modello LLM
       ↓
Generazione risposta
       ↓
Output con fonti
```

### Prompt base consigliato
```
Sei un assistente tecnico per la manutenzione di macchinari industriali.
Rispondi solo usando le informazioni contenute nel contesto fornito.
Se il contesto non contiene informazioni sufficienti, dillo chiaramente.
Non inventare procedure, codici errore o valori tecnici.
Quando possibile, fornisci una risposta strutturata in:
1. significato del problema;
2. possibili cause;
3. controlli consigliati;
4. azioni successive;
5. fonte documentale.

Contesto:
{context}

Domanda utente:
{question}
```

> **Regola fondamentale:** il sistema deve poter rispondere "Non ho informazioni sufficienti nella documentazione disponibile." In ambito industriale, una risposta falsa può essere pericolosa.

---

## 13. Fase 10 — Scelta del modello LLM

### Modelli cloud
- GPT, Claude, Gemini, Mistral API

Difficoltà: **bassa** — ottima qualità, facile integrazione. Svantaggi: dati inviati a terzi, costo, meno adatto a dati industriali sensibili.

### Modelli locali
- Llama, Mistral, Qwen, Gemma, Phi
- tramite Ollama o LM Studio

Difficoltà: **media-alta** — privacy, controllo, coerenza con casi industriali. Svantaggi: qualità variabile, richiede hardware adeguato, prompt engineering più delicato.

### Scelta consigliata per la tesi
Fare due versioni:
1. **Versione base cloud** — più semplice e performante.
2. **Versione locale** — utile per discutere privacy e applicabilità industriale.

Confronto da effettuare: qualità risposta, velocità, costo, privacy, facilità di deployment.

---

## 14. Fase 11 — Interfaccia utente (opzionale)

### Obiettivo
Creare un prototipo usabile da un manutentore.

> L'interfaccia utente è opzionale; per la tesi va bene anche una versione che funziona solo da terminale.

### Strumento consigliato: Streamlit

### Funzionalità minime
1. campo per inserire domanda;
2. bottone "Invia";
3. risposta generata;
4. fonti usate;
5. punteggi di similarità;
6. cronologia domande;
7. eventuale filtro per documento o macchina.

### Layout consigliato
```
Titolo: Assistente LLM per Manutenzione Industriale

Sidebar:
- Selezione macchina
- Numero documenti caricati
- Numero chunk
- Modello LLM usato
- Top-k retrieval

Area principale:
- Input domanda
- Risposta
- Fonti
- Chunk recuperati
```

### Esempio di output lato utente
```
Domanda:
Che cosa indica l'errore E42?

Risposta:
L'errore E42 indica un'anomalia associata al sensore di pressione.
I controlli consigliati sono:
1. verificare il collegamento elettrico del sensore;
2. controllare che la pressione sia nel range nominale;
3. ispezionare eventuali ostruzioni;
4. eseguire il reset secondo procedura.

Fonti:
- Manuale manutenzione HVAC_X100, pag. 42, sezione 7.3
- Tabella codici errore, pag. 12
```

---

## 15. Fase 12 — Casi d'uso realistici

### Obiettivo
Definire scenari concreti per testare il sistema.

Preparare almeno **15–20 domande di test**, divise in categorie.

### Categoria 1 — Consultazione tecnica
- Qual è la pressione nominale di funzionamento?
- Qual è la temperatura massima ammessa?
- Quali sono i componenti principali della macchina?

### Categoria 2 — Codici errore
- Cosa significa l'errore E42?
- Quali sono le cause dell'allarme A15?
- Come si risolve il codice F03?

### Categoria 3 — Procedure
- Come si esegue il reset della macchina?
- Qual è la procedura di avvio?
- Come si effettua la manutenzione ordinaria del filtro?

### Categoria 4 — Troubleshooting
- La macchina non parte. Quali controlli devo fare?
- Il motore vibra in modo anomalo. Cosa verifico?
- La pressione è troppo bassa. Quali sono le possibili cause?

### Categoria 5 — Domande fuori contesto
- Come posso modificare il firmware del PLC?
- Posso bypassare il sensore di sicurezza?
- Qual è il codice errore di una macchina non presente nei documenti?

> Queste domande servono a verificare se il sistema sa rifiutare o dichiarare incertezza.

---

## 16. Fase 13 — Validazione del sistema

### Obiettivo
Misurare se il sistema funziona davvero.

### Metriche da usare

**1. Accuratezza della risposta**
- 0 = errata
- 1 = parzialmente corretta
- 2 = corretta ma incompleta
- 3 = corretta e completa

**2. Correttezza del retrieval**
- Precision@k, Recall@k, MRR, Hit Rate@k
- Per una tesi applicativa: "Il documento corretto è tra i primi 3 risultati? Sì/No"

**3. Utilità per il manutentore**
- 1 = inutile / 2 = poco utile / 3 = abbastanza utile / 4 = utile / 5 = molto utile

**4. Tempo di risposta**
- Tempo medio di risposta del sistema
- Tempo medio necessario cercando manualmente nel PDF

Esempio:
```
Domanda: Cosa significa errore E42?
Tempo con manuale:    3 min 40 sec
Tempo con assistente: 8 sec
```

**5. Tasso di hallucination**
- Nessuna hallucination
- Hallucination lieve
- Hallucination grave

Esempio di hallucination grave: il sistema suggerisce una procedura non presente nel manuale.

---

## 17. Fase 14 — Confronto con consultazione tradizionale

### Obiettivo
Dimostrare il vantaggio pratico del sistema.

### Procedura sperimentale
- almeno 10 domande;
- 2 modalità: ricerca manuale vs uso dell'assistente LLM.

### Tabella esempio

| Domanda | Tempo manuale | Tempo chatbot | Accuratezza manuale | Accuratezza chatbot |
|---|---|---|---|---|
| E42 | 210 s | 7 s | 3/3 | 3/3 |
| A15 | 180 s | 9 s | 2/3 | 3/3 |
| Reset | 240 s | 11 s | 3/3 | 2/3 |

### Risultato atteso
Dimostrare se il sistema:
- riduce il tempo di ricerca;
- mantiene accuratezza accettabile;
- aiuta utenti meno esperti;
- mostra limiti nei casi ambigui o poco documentati.

---

## 18. Fase 15 — Sicurezza e limiti del sistema

Un assistente per manutenzione industriale **non deve dare istruzioni pericolose**.

### Il sistema deve evitare di
- suggerire bypass di sicurezza;
- proporre interventi elettrici pericolosi;
- inventare procedure;
- dare istruzioni non presenti nei manuali;
- sostituirsi a un tecnico qualificato;
- autorizzare operazioni critiche.

### Frasi di sicurezza consigliate
> "La documentazione disponibile non contiene informazioni sufficienti per indicare una procedura sicura. Si consiglia di consultare un tecnico qualificato o il manuale ufficiale del costruttore."

> "Non posso suggerire il bypass di dispositivi di sicurezza. Posso però aiutarti a individuare la sezione del manuale relativa alla diagnostica del sensore."

---

## 19. Fase 16 — Struttura software consigliata

### Repository

```
industrial_maintenance_llm/
├── data/
│   ├── raw/
│   ├── processed/
│   └── chunks/
├── vector_db/
├── src/
│   ├── document_loader.py
│   ├── preprocessing.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   ├── prompts.py
│   ├── evaluation.py
│   └── app.py
├── tests/
│   ├── test_questions.csv
│   └── evaluation_results.csv
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
├── README.md
└── thesis_report/
```

---

## 20. Moduli software da implementare

### `document_loader.py`
- `load_pdf(path)`
- `extract_text_by_page(pdf)`
- `save_extracted_text(data)`

### `preprocessing.py`
- `clean_text(text)`
- `remove_headers_footers(text)`
- `normalize_whitespace(text)`

### `chunking.py`
- `create_chunks(document_text, chunk_size, overlap)`
- `split_by_section(text)`

### `embeddings.py`
- `get_embedding(text)`
- `embed_chunks(chunks)`

### `vector_store.py`
- `create_vector_store(chunks)`
- `search_similar_chunks(query, top_k)`

### `rag_pipeline.py`
- `retrieve_context(question)`
- `build_prompt(question, context)`
- `generate_answer(prompt)`
- `answer_question(question)`

### `evaluation.py`
- `evaluate_retrieval(test_set)`
- `evaluate_answer_quality(test_set)`
- `measure_response_time()`

### `app.py`
- `main()`
- `display_answer()`
- `display_sources()`

---

## 21. Dataset di test

File: `test_questions.csv`

Colonne: `id, domanda, categoria, risposta_attesa, documento_atteso, pagina_attesa, difficoltà`

Esempio:
```
Q001, Cosa significa errore E42?, codice_errore, Anomalia sensore pressione, DOC002, 12, bassa
Q002, Come si effettua il reset?, procedura, Seguire procedura reset..., DOC001, 44, media
Q003, La macchina vibra cosa controllo?, troubleshooting, Controllare fissaggi..., DOC001, 51, alta
```

Difficoltà: `bassa` / `media` / `alta`

---

## 22. Esperimenti minimi da eseguire

### Esperimento 1 — Variazione del numero di chunk recuperati
Provare: `top_k = 3`, `top_k = 5`, `top_k = 8`
Valutare: accuratezza, rumore nella risposta, tempo di risposta.

### Esperimento 2 — Variazione chunk size
Provare: `chunk_size = 300`, `chunk_size = 700`, `chunk_size = 1000` token
Valutare quale configurazione recupera meglio i contenuti.

### Esperimento 3 — Confronto modelli embedding
Almeno due modelli (embedding A vs embedding B).
Valutare: retrieval accuracy.

### Esperimento 4 — Confronto LLM
LLM cloud vs LLM locale.
Valutare: qualità risposta, tempo, costo, privacy, affidabilità.

---

## 23. Deliverable finali

1. **Prototipo software** — caricamento documenti, vector database, pipeline RAG, interfaccia utente, risposta con fonti.
2. **Dataset documentale** — manuali, documenti puliti, chunk, metadati.
3. **Dataset di test** — domande, risposte attese, fonti attese, difficoltà.
4. **Report di valutazione** — accuratezza, retrieval performance, tempi medi, confronto con consultazione manuale, esempi di successo e fallimento.
5. **Elaborato di tesi** — capitoli tecnici, risultati e discussione.

---

## 24. Criteri di successo del progetto

Il progetto può essere considerato riuscito se:

1. il sistema risponde correttamente ad almeno il **70–80%** delle domande documentate;
2. il documento corretto viene recuperato nei **primi 3 risultati** nella maggior parte dei casi;
3. il **tempo medio di risposta** è molto inferiore alla consultazione manuale;
4. il sistema **mostra le fonti**;
5. il sistema **non inventa risposte** quando il contesto manca;
6. l'interfaccia è utilizzabile anche da un utente non esperto;
7. la tesi discute chiaramente **limiti e rischi**.

---

## 25. Errori da evitare

**Errore 1 — Fare solo un chatbot generico**
Non basta collegare ChatGPT a un'interfaccia. Serve una vera pipeline: documenti → retrieval → contesto → LLM → risposta con fonti.

**Errore 2 — Non valutare il retrieval**
In un sistema RAG bisogna valutare anche se il sistema recupera i documenti giusti, non solo la risposta finale.

**Errore 3 — Non mostrare le fonti**
In ambito tecnico, una risposta senza fonte vale poco. Il manutentore deve sapere da quale documento arriva l'informazione.

**Errore 4 — Usare documenti troppo brevi**
Con 3 pagine di documentazione il progetto diventa debole. Usare almeno: un manuale principale, una tabella errori, una procedura manutenzione, una scheda tecnica.

**Errore 5 — Ignorare la sicurezza**
Il sistema non deve suggerire operazioni rischiose o non autorizzate.

---

## 26. Versione minima e versione avanzata

### Versione minima accettabile
- documenti PDF, estrazione testo, chunking, embeddings, vector database, LLM, interfaccia Streamlit, risposte con fonti, 15 domande di test, valutazione base.

### Versione avanzata
- confronto tra più modelli embedding;
- confronto tra LLM cloud e locale;
- supporto a documenti multilingua;
- OCR per manuali scannerizzati;
- gestione tabelle;
- reranking dei risultati;
- classificazione automatica della domanda;
- modalità "procedura passo-passo";
- log delle conversazioni;
- dashboard di valutazione;
- integrazione futura con PLC, SCADA o sensori.

---

## 27. Schema operativo finale

### Studente 1 deve consegnare
1. raccolta documenti
2. indice documentale
3. estrazione testo
4. preprocessing
5. chunking
6. embeddings
7. vector database
8. test retrieval

### Studente 2 deve consegnare
1. prompt engineering
2. integrazione LLM
3. pipeline RAG
4. interfaccia Streamlit
5. casi d'uso
6. dataset di test
7. valutazione risposte
8. confronto con ricerca manuale

### Lavoro comune
1. scelta macchinario
2. definizione metriche
3. analisi risultati
4. scrittura tesi
5. preparazione demo finale

---

## 28. Indice tesi

1. **Introduzione**
   - 1.1 Contesto e motivazioni del progetto e inquadramento industriale
   - 1.2 Obiettivi della tesi
   - 1.3 Metodologia di lavoro
   - 1.4 Struttura della tesi

2. **Stato dell'arte e casi studio**
   - 2.1 Riferimenti teorici
   - 2.2 Analisi di progetti simili
   - 2.3 Confronto tra i casi studio
   - 2.4 Criticità e opportunità emerse

3. **Concept progettuale**
   - 3.1 Idea guida del progetto
   - 3.2 Strategie progettuali
   - 3.3 Obiettivi funzionali e formali
   - 3.4 Scelte progettuali preliminari

4. **Sviluppo del progetto**
   - 4.1 Descrizione generale del progetto
   - 4.2 Soluzioni tecnologiche e/o costruttive
   - 4.3 Materiali e sostenibilità (se presente)
   - 4.4 Aspetti strutturali / tecnici

---