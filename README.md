<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyMuPDF-1.27-009688?style=for-the-badge" alt="PyMuPDF"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/RAG-Pipeline-FF6F00?style=for-the-badge" alt="RAG"/>
</p>

<h1 align="center">🤖 ARIS</h1>
<h3 align="center"><i>Assistente per Robot Industriale Smart</i></h3>

<p align="center">
  Un assistente tecnico intelligente basato su <b>Retrieval-Augmented Generation (RAG)</b><br>
  per la manutenzione del controller <b>FANUC R-30iB Mate / R-30iB Mate Plus</b>
</p>

---

## 📋 Indice

- [Panoramica](#-panoramica)
- [Architettura](#-architettura)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Documentazione Tecnica](#-documentazione-tecnica)
- [Installazione](#-installazione)
- [Utilizzo](#-utilizzo)
- [Pipeline di Estrazione](#-pipeline-di-estrazione)
- [Roadmap](#-roadmap)
- [Licenza](#-licenza)

---

## 🔍 Panoramica

**ARIS** è un sistema RAG progettato come progetto di tesi per fornire assistenza tecnica intelligente nella manutenzione di robot industriali FANUC. Il sistema:

1. 📄 **Estrae** informazioni da manuali tecnici PDF (testo + tabelle)
2. 🔎 **Recupera** i passaggi più rilevanti tramite ricerca semantica
3. 🧠 **Genera** risposte tecniche accurate e verificabili tramite LLM
4. 📌 **Cita** le fonti documentali (documento, pagina, sezione)
5. ⚠️ **Rifiuta** di rispondere quando il contesto è insufficiente

### Esempio di Interazione

```
👤 Domanda:  Il controller non si avvia. Quali controlli devo fare?

🤖 Risposta: Secondo la documentazione, i controlli da effettuare sono:
   1. Verificare il circuit breaker del controller
   2. Controllare il power supply unit e la conversione AC/DC
   3. Verificare il cavo del teach pendant per torsioni eccessive
   4. Controllare il controller e i dispositivi periferici per anomalie
   
   📄 Fonte: checks_maintenance.pdf, pag. 11, sez. 2.3
   📄 Fonte: overview_configuration.pdf, pag. 10, sez. 2.1
```

---

## 🏗 Architettura

Il sistema segue una classica pipeline **Retrieval-Augmented Generation**:

```
 ┌─────────────────────────────────────────────────────┐
 │                  📁 DOCUMENTI TECNICI                │
 │   manuali · codici errore · procedure · schede      │
 └──────────────────────┬──────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────┐
 │              ⚙️  PREPROCESSING & ESTRAZIONE          │
 │   PyMuPDF (testo) + pdfplumber (tabelle)            │
 │   Pulizia · Filtro rumore · Riconoscimento sezioni  │
 └──────────────────────┬──────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────┐
 │                  ✂️  CHUNKING                         │
 │   Suddivisione semantica · Overlap · Metadati       │
 └──────────────────────┬──────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────┐
 │              🧬 EMBEDDINGS + VECTOR DB               │
 │   Vettorizzazione dei chunk · Indicizzazione         │
 └──────────────────────┬──────────────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────┐
 │          💬 QUERY → RETRIEVAL → LLM → RISPOSTA       │
 │   Ricerca semantica top-k · Prompt engineering      │
 │   Generazione risposta con fonti documentali        │
 └──────────────────────────────────────────────────────┘
```

---

## 📂 Struttura del Progetto

```
ARIS/
├── 📁 data/
│   ├── 📁 raw/                          # Documentazione originale
│   │   ├── 📁 manuali_manutenzione/     # Manuali PDF del controller
│   │   ├── 📁 codici_errore/            # Tabelle errori e troubleshooting
│   │   ├── 📁 procedure/               # Procedure di sostituzione
│   │   ├── 📁 schede_tecniche/          # Schemi circuiti e cablaggi
│   │   └── 📁 metadata/                # Indice documentale (CSV)
│   ├── 📁 processed/                    # Testo estratto (JSON)
│   └── 📁 chunks/                       # Chunk pronti per il RAG
│
├── 📁 src/
│   └── 📁 estrazione/                   # Script di estrazione PDF
│       ├── 📄 estrazione_overview_configuration.py
│       └── 📄 estrazione_checks_maintenance.py
│
├── 📁 vector_db/                        # Database vettoriale
├── 📁 notebooks/                        # Analisi e sperimentazione
├── 📁 tests/                            # Dataset di test e valutazione
├── 📁 thesis_report/                    # Elaborato di tesi
│
├── 📄 requirements.txt                  # Dipendenze Python
├── 📄 LICENSE                           # MIT License
└── 📄 README.md                         # Questo file
```

---

## 📚 Documentazione Tecnica

Il sistema elabora la documentazione ufficiale FANUC per il controller **R-30iB Mate / R-30iB Mate Plus**:

| ID | Documento | Tipo | Contenuto |
|:---:|---|---|---|
| 01 | `safety_precautions.pdf` | 🛡️ Sicurezza | Avvertenze e precauzioni di sicurezza |
| 02 | `overview_configuration.pdf` | 📖 Panoramica | Componenti principali e configurazione |
| 03 | `checks_maintenance.pdf` | 🔧 Manutenzione | Procedure di manutenzione ordinaria |
| 04 | `troubleshooting_alarms.pdf` | 🚨 Errori | Codici errore e allarmi |
| 05 | `troubleshooting_visual.pdf` | 🔍 Diagnostica | Troubleshooting visivo |
| 06 | `circuit_boards_amplifiers.pdf` | ⚡ Schede | Circuiti e amplificatori |
| 07 | `replacing_units.pdf` | 🔄 Procedure | Sostituzione componenti |
| 08 | `connections.pdf` | 🔌 Cablaggi | Schemi e descrizioni cablaggi |

---

## ⚡ Installazione

### Prerequisiti

- Python 3.10+
- [Conda](https://docs.conda.io/) (consigliato) o pip

### Setup con Conda

```bash
# Clona il repository
git clone https://github.com/DanieleDiSpirito/ARIS.git
cd ARIS

# Crea e attiva l'ambiente
conda create -n aris python=3.10 -y
conda activate aris

# Installa le dipendenze
pip install -r requirements.txt
```

### Setup con pip

```bash
git clone https://github.com/DanieleDiSpirito/ARIS.git
cd ARIS
pip install -r requirements.txt
```

---

## 🚀 Utilizzo

### 1. Estrazione del Testo

Esegui gli script di estrazione dalla directory `src/estrazione/`:

```bash
cd src/estrazione

# Estrae testo e tabelle da overview_configuration.pdf
python estrazione_overview_configuration.py

# Estrae testo da checks_maintenance.pdf
python estrazione_checks_maintenance.py
```

I file JSON estratti vengono salvati in `data/processed/`.

### 2. Formato dell'Output

Ogni blocco estratto segue questo schema JSON:

```json
{
    "document_id": "02",
    "file_name": "manuali_manutenzione/overview_configuration.pdf",
    "page": "10",
    "section": "2.1",
    "title": "EXTERNAL VIEW OF THE CONTROLLER",
    "text": "- Main board The main board contains a microprocessor..."
}
```

| Campo | Descrizione |
|---|---|
| `document_id` | ID univoco dal catalogo documentale |
| `file_name` | Percorso relativo al PDF sorgente |
| `page` | Numero di pagina nel manuale originale |
| `section` | Codice sezione (es. 2.1, 2.3) |
| `title` | Titolo della sezione corrente |
| `text` | Testo estratto e pulito |

---

## ⚙️ Pipeline di Estrazione

L'estrazione utilizza un approccio **dual-engine** per massimizzare la qualità:

### Engine 1 — pdfplumber (Tabelle)
- Rileva tabelle strutturate nel PDF
- Salva le bounding box per evitare duplicati
- Converte le tabelle in formato leggibile:
  ```
  Specifiche per CE controller -> EMC Standard: EN 55011 | Robot Standard: EN/ISO 10218-1
  ```

### Engine 2 — PyMuPDF (Testo)
- Estrae il testo a blocchi con crop dell'area utile
- Filtra header, footer e numeri di pagina
- Riconosce automaticamente sezioni e titoli tramite regex
- Gestisce l'ereditarietà del contesto (sezione/titolo)
- Filtra rumore da etichette di immagini

### Preprocessing
- ✅ Rimozione newline interni e normalizzazione spazi
- ✅ Preservazione codici tecnici, sigle e unità di misura
- ✅ Mantenimento avvertenze di sicurezza (WARNING, NOTE)
- ✅ Filtro anti-rumore per testi < 5 parole

---

## 🗺 Roadmap

- [x] Raccolta e organizzazione documentazione FANUC
- [x] Indice documentale (`document_index.csv`)
- [x] Script estrazione con dual-engine (PyMuPDF + pdfplumber)
- [x] Estrazione `overview_configuration.pdf`
- [x] Estrazione `checks_maintenance.pdf`
- [ ] Estrazione dei rimanenti 6 PDF
- [ ] Preprocessing avanzato e pulizia testo
- [ ] Chunking semantico con overlap
- [ ] Creazione embeddings (cloud + locale)
- [ ] Vector database (FAISS / ChromaDB)
- [ ] Pipeline RAG completa
- [ ] Integrazione LLM (cloud + locale)
- [ ] Prompt engineering
- [ ] Interfaccia utente (Streamlit)
- [ ] Dataset di test (15-20 domande)
- [ ] Validazione e confronto con consultazione manuale

---

## 📄 Licenza

Questo progetto è distribuito sotto licenza **MIT**. Vedi il file [LICENSE](LICENSE) per maggiori dettagli.

---

<p align="center">
  <i>Progetto di tesi — Assistente LLM per Manutenzione Industriale</i><br>
  <b>Daniele Di Spirito</b> · 2026
</p>
