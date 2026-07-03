import os
import argparse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

COLLECTION_NAME = "langchain"

# Mappa env+chunk_size → cartella del vector DB
def get_db_path(env: str, chunk_size: int, metodo: str = "pdf4llm") -> str:
    """Restituisce il percorso corretto del Vector DB in base all'ambiente, alla dimensione dei chunk e al metodo di estrazione."""
    return os.path.join("vector_db", f"chroma_{metodo}_{env}_{chunk_size}")


def get_embeddings(env: str):
    """Seleziona il modello di embedding coerente con quello usato durante l'ingestion.

    - locale → BAAI/bge-m3  (1024 dimensioni, HuggingFace, locale)
    - cloud  → text-embedding-3-small  (1536 dimensioni, OpenAI)
    """
    if env == "locale":
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif env == "cloud":
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"env non valido: '{env}'. Scegli tra 'locale' o 'cloud'.")


def build_prompt():
    """Costruisce il prompt con System+Human separati.
    Gestisce tre situazioni:
      1. Messaggi di chat generica (saluti, ringraziamenti, follow-up)
      2. Riferimenti alla conversazione precedente
      3. Domande tecniche sui manuali Fanuc
    """
    system_template = """Sei ARIS, un assistente tecnico esperto per robot Fanuc, amichevole e disponibile.

Hai accesso a:
- Un "Contesto tecnico recuperato" dai manuali Fanuc
- Una "Cronologia conversazione" con i messaggi precedenti

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGOLA 1 — MESSAGGI DI CHAT GENERICA
Se il messaggio dell'operatore è un saluto (es. "ciao", "buongiorno", "grazie"), 
una presentazione, o una domanda non tecnica (es. "cosa sai fare?", "chi sei?"),
rispondi in modo naturale e cordiale SENZA usare il contesto tecnico.
Presentati brevemente come assistente tecnico per robot Fanuc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGOLA 2 — RIFERIMENTI ALLA CONVERSAZIONE PRECEDENTE
Se il messaggio fa riferimento a uno scambio precedente (es. "quella risposta non va bene",
"puoi approfondire?", "e per l'asse 2?", "come dicevi prima..."), usa la Cronologia 
conversazione per capire il contesto, poi rispondi o correggi di conseguenza.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGOLA 3 — DOMANDE TECNICHE SUI MANUALI
Se la domanda è tecnica (allarmi, procedure, specifiche hardware, connessioni):
- Rispondi ESCLUSIVAMENTE usando il "Contesto tecnico recuperato".
- Il contesto contiene tabelle con "|": analizza riga per riga per trovare la corrispondenza.
- Solo se sei ASSOLUTAMENTE CERTO che il dato non esista nella documentazione, scrivi:
  "La documentazione disponibile non contiene informazioni sufficienti. Si consiglia di consultare un tecnico qualificato."
- Non inventare mai procedure, codici errore, valori tecnici o bypass di sicurezza.

FORMATTAZIONE per domande tecniche:
- ALLARME / ERRORE / GUASTO → elenco numerato: 1. Significato 2. Possibili cause 3. Controlli 4. Azioni 5. Fonte documentale.
- SPECIFICHE / COMPONENTI → risposta discorsiva.

IMPORTANTE: Devi SEMPRE includere la fonte alla fine di QUALSIASI risposta tecnica. La pagina DEVE essere specificata sempre.
Se usi informazioni provenienti da più pagine o file diversi, elencali tutti separandoli con una virgola.
Scrivi ESATTAMENTE in questo formato su una nuova riga alla fine:
"Fonte documentale: [nome_file.pdf] (Pagina: [numero_pagina]), [altro_file.pdf] (Pagina: [altro_numero])"
"""

    human_template = """Cronologia conversazione (ultimi scambi):
{history}

Contesto tecnico recuperato:
{context}

Messaggio dell'operatore:
{question}"""

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])


def format_docs_with_sources(docs):
    if not docs:
        return "Nessun dato trovato nel contesto."

    print(f"\n--- 🔎 CHUNK INVIATI ALL'LLM — Ibrido BM25+Vector ({len(docs)} totali) ---")
    formatted_chunks = []
    for i, doc in enumerate(docs):
        file_name = doc.metadata.get("file_name", "Documento Sconosciuto")
        page = doc.metadata.get("original_source_page", doc.metadata.get("page", "N/A"))

        print(f"  [{i+1}] {file_name} | Pag: {page} | Testo: {doc.page_content[:100].replace(chr(10), ' ')}...")

        chunk_str = f"--- INIZIO FONTE: {file_name} (Pagina: {page}) ---\n{doc.page_content}\n--- FINE FONTE ---\n"
        formatted_chunks.append(chunk_str)

    print("---------------------------------------------------\n")
    return "\n".join(formatted_chunks)


def setup_rag_chain(retriever, env="locale", model_name=None):
    """Configura la pipeline RAG collegando retriever, prompt e LLM.

    La chain accetta un dict: {"question": str, "history": str}
    - question : domanda corrente dell'operatore
    - history  : ultimi scambi formattati come stringa (può essere vuota)
    """
    if env == "locale":
        model = model_name if model_name else os.getenv("LOCAL_LLM_MODEL", None)
        print(f"🤖 LLM: Locale (server su localhost:1234) | Modello: {model}")
        base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
        llm_classifier = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key="lm-studio",
            temperature=0.0,
            streaming=False,
            max_tokens=1024
        )
        llm_technical = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key="lm-studio",
            temperature=0.0,
            streaming=True,
            max_tokens=2048
        )
        llm_general = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key="lm-studio",
            temperature=0.7,
            streaming=True,
            max_tokens=2048
        )
    elif env == "cloud":
        model = model_name if model_name else "openai/gpt-4o-mini"
        print(f"☁️ LLM: Cloud (OpenRouter) | Modello: {model}")
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("❌ ERRORE: Variabile OPENAI_API_KEY non trovata nel file .env")
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.getenv("OPENAI_API_KEY")
        llm_classifier = ChatOpenAI(
            model=model,
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=0.0,
            streaming=False
        )
        llm_technical = ChatOpenAI(
            model=model,
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=0.0,
            streaming=True
        )
        llm_general = ChatOpenAI(
            model=model,
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=0.7,
            streaming=True
        )
    else:
        raise ValueError("Il parametro env deve essere 'locale' o 'cloud'")

    # Normalizza l'input in un dizionario con chiavi 'question' e 'history'
    normalize_input = RunnableLambda(
        lambda x: {"question": x, "history": ""} if isinstance(x, str) else {
            "question": x.get("question", ""),
            "history": x.get("history", "")
        }
    )

    # Classifica l'intento della domanda
    def classify_intent(inputs):
        question = inputs.get("question", "")
        
        # Controllo rapido a regole per evitare chiamate LLM su saluti semplici
        q_clean = question.strip().lower().rstrip("!?.,")
        greetings = {
            "ciao", "buongiorno", "buonasera", "salve", "hello", "hi", "hey",
            "grazie", "grazie mille", "thank you", "thanks", "prego"
        }
        if q_clean in greetings:
            return "GENERAL"
            
        # Classificazione semantica tramite LLM
        classification_prompt = (
            "Classifica la seguente domanda dell'operatore per un assistente di manutenzione di robot Fanuc.\n"
            "Rispondi ESCLUSIVAMENTE con una delle due parole: 'TECHNICAL' o 'GENERAL'.\n\n"
            "- TECHNICAL: domande su allarmi (es. SRVO-004), cablaggi, specifiche hardware, procedure, diagnostica o manutenzione.\n"
            "- GENERAL: saluti, domande di cortesia, presentazioni, o argomenti non inerenti ai robot (es. ricette, meteo, sport, opinioni).\n\n"
            f"Domanda dell'operatore: {question}\n"
            "Risposta:"
        )
        try:
            res = llm_classifier.invoke(classification_prompt)
            intent = res.content.strip().upper()
            return "TECHNICAL" if "TECHNICAL" in intent else "GENERAL"
        except Exception as e:
            print(f"⚠️ Errore durante la classificazione dell'intento: {e}. Fallback su TECHNICAL.")
            return "TECHNICAL"

    def translate_query_if_needed(question):
        translation_prompt = (
            "Traduci la seguente domanda in inglese se è in italiano, altrimenti restituiscila identica senza alcun commento, spiegazione o introduzione.\n"
            f"Domanda: {question}\n"
            "Traduzione:"
        )
        try:
            res = llm_classifier.invoke(translation_prompt)
            translated = res.content.strip()
            if translated and translated.lower() != question.lower():
                print(f"🌐 Query tradotta in inglese per il retrieval: '{translated}' (originale: '{question}')")
                return translated
        except Exception as e:
            print(f"⚠️ Errore durante la traduzione della query: {e}. Uso la query originale.")
        return question

    # Decide se invocare il retriever o saltarlo
    def retrieve_or_skip(inputs):
        if inputs.get("intent") == "GENERAL":
            print("🔀 Query non pertinente rilevata: Salto il retrieval dei chunk.")
            return []
        
        question = inputs["question"]
        search_query = translate_query_if_needed(question)
        return retriever.invoke(search_query)

    prompt = build_prompt()

    chain_technical = prompt | llm_technical | StrOutputParser()
    chain_general = prompt | llm_general | StrOutputParser()

    def route_by_intent(inputs):
        if inputs.get("intent") == "GENERAL":
            return chain_general
        else:
            return chain_technical

    # Pipeline RAG con Query Routing:
    rag_chain = (
        normalize_input
        | RunnablePassthrough.assign(intent=RunnableLambda(classify_intent))
        | {
            "context":  RunnableLambda(retrieve_or_skip) | format_docs_with_sources,
            "question": lambda x: x["question"],
            "history":  lambda x: x["history"],
            "intent":   lambda x: x["intent"]
        }
        | RunnableLambda(route_by_intent)
    )

    return rag_chain


def answer_question(rag_chain, question):
    """Invia la domanda alla catena e gestisce eventuali errori di connessione."""
    try:
        return rag_chain.invoke(question)
    except Exception as e:
        return f"❌ Errore di connessione o generazione. Dettagli: {str(e)}"


def build_hybrid_retriever(db: Chroma, k: int = 3) -> EnsembleRetriever:
    """Costruisce un retriever ibrido combinando BM25 (keyword) e Chroma (semantic).

    Args:
        db: istanza Chroma già caricata.
        k: numero di documenti da recuperare per ciascun retriever.

    Returns:
        EnsembleRetriever con peso 50% BM25 + 50% Vector Search.
    """
    # 1. Retriever denso (Vector Search)
    chroma_retriever = db.as_retriever(search_kwargs={"k": k})

    # 2. Retriever sparso (BM25 / Keyword Search)
    # Carichiamo tutti i documenti dal Chroma DB per inizializzare BM25.
    # NOTA: per dataset molto grandi, in produzione si userebbe un indice BM25
    # persistito separatamente (es. Elasticsearch/OpenSearch).
    all_data = db.get()
    if not all_data or not all_data.get("documents"):
        raise RuntimeError("❌ Nessun documento trovato nel database vettoriale. Impossibile costruire BM25.")

    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
        if text
    ]
    print(f"✅ BM25 inizializzato su {len(docs)} chunk...")
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    # 3. Ensemble: combina i due retriever con pesi uguali
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]   # 50% keyword + 50% semantic
    )
    return ensemble


class DebugHybridRetriever:
    """Retriever wrapper che mostra informazioni di debug dettagliate per la ricerca ibrida."""
    def __init__(self, ensemble_retriever, db, k: int = 3, debug: bool = False):
        self.ensemble_retriever = ensemble_retriever
        self.db = db
        self.k = k
        self.debug = debug

    def invoke(self, query: str) -> list:
        if not self.debug:
            return self.ensemble_retriever.invoke(query)

        # 1. Vector Search manualmente
        chroma_results = self.db.similarity_search_with_score(query, k=self.k)
        
        # 2. BM25 Search manualmente
        bm25_retriever = None
        for r in self.ensemble_retriever.retrievers:
            if hasattr(r, 'vectorizer') and hasattr(r, 'docs'):
                bm25_retriever = r
                break
        
        bm25_results = []
        if bm25_retriever:
            bm25 = bm25_retriever.vectorizer
            tokens = bm25_retriever.preprocess_func(query)
            bm25_scores = bm25.get_scores(tokens)
            
            def compute_contrib(bm25, token, doc_idx):
                k1 = bm25.k1
                b = bm25.b
                avgdl = bm25.avgdl
                doc_len = bm25.doc_len[doc_idx]
                q_freq = bm25.doc_freqs[doc_idx].get(token, 0)
                q_idf = bm25.idf.get(token, 0)
                denom = (q_freq + k1 * (1 - b + b * doc_len / avgdl))
                return (q_idf * q_freq * (k1 + 1)) / denom if denom != 0 else 0.0

            scored_docs = []
            for doc_idx, doc in enumerate(bm25_retriever.docs):
                score = bm25_scores[doc_idx]
                if score > 0:
                    contribs = []
                    for t in tokens:
                        c = compute_contrib(bm25, t, doc_idx)
                        if c > 0:
                            contribs.append((t, c))
                    contribs = sorted(contribs, key=lambda x: x[1], reverse=True)
                    scored_docs.append((score, doc, contribs))
            bm25_results = sorted(scored_docs, key=lambda x: x[0], reverse=True)[:self.k]

        print("\n🔎 [Debug Ibrido] --- VECTOR SEARCH (Chroma) ---")
        for idx, (doc, score) in enumerate(chroma_results):
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            file_name = doc.metadata.get("file_name", "unknown")
            page = doc.metadata.get("page", "N/A")
            print(f"  [{idx+1}] Distanza: {score:.4f} | ID: {chunk_id} | File: {file_name} | Pagina: {page}")
            print(f"      Snippet: {doc.page_content[:80].replace(chr(10), ' ')}...")

        print("\n🔎 [Debug Ibrido] --- KEYWORD SEARCH (BM25) ---")
        for idx, (score, doc, contribs) in enumerate(bm25_results):
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            file_name = doc.metadata.get("file_name", "unknown")
            page = doc.metadata.get("page", "N/A")
            contribs_str = ", ".join([f"'{w}': {c:.3f}" for w, c in contribs[:3]])
            print(f"  [{idx+1}] Score BM25: {score:.4f} | ID: {chunk_id} | File: {file_name} | Pagina: {page}")
            print(f"      Contribuzioni parole chiave: {contribs_str}")
            print(f"      Snippet: {doc.page_content[:80].replace(chr(10), ' ')}...")

        fused_docs = self.ensemble_retriever.invoke(query)
        print("\n🔎 [Debug Ibrido] --- RISULTATO FUSO (Reciprocal Rank Fusion) ---")
        for idx, doc in enumerate(fused_docs):
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            file_name = doc.metadata.get("file_name", "unknown")
            page = doc.metadata.get("page", "N/A")
            print(f"  [{idx+1}] ID: {chunk_id} | File: {file_name} | Pagina: {page}")
            print(f"      Snippet: {doc.page_content[:80].replace(chr(10), ' ')}...")
        print("--------------------------------------------------\n")

        return fused_docs


def load_questions_from_csv(csv_path: str, question_indices: list) -> dict:
    """Carica le domande selezionate per ID dal file CSV delle domande di test."""
    import csv
    questions = {}
    target_ids = {f"Q{idx:03d}" for idx in question_indices}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File non trovato: {csv_path}")
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("id")
            if qid in target_ids:
                try:
                    num = int(qid[1:])
                    questions[num] = row.get("question")
                except ValueError:
                    pass
    return questions


def main():
    parser = argparse.ArgumentParser(description="Testa la RAG Pipeline Ibrida (BM25 + Vector Search) dal terminale")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale",
                        help="Scegli tra 'locale' (LM Studio) o 'cloud' (OpenRouter)")
    parser.add_argument("--chunk_size", type=int, default=700,
                        help="Dimensione dei chunk usata durante l'ingestion (default: 700). "
                             "Determina quale cartella del Vector DB viene caricata.")
    parser.add_argument("--metodo", type=str, default="pdf4llm",
                        choices=["euristico", "docling", "llamaparse", "qwen", "pdf4llm", "mineru"],
                        help="Metodo di estrazione da utilizzare (default: pdf4llm)")
    parser.add_argument("--query", type=str, default="Cosa significa l'allarme SRVO-004?",
                        help="La domanda da porre al sistema")
    parser.add_argument("--question", type=str, default=None,
                        help="Indice o lista di indici di domande (da 1 a 100, es: 1 o 1,2,5) dal file tests/test_questions_it.csv")
    parser.add_argument("--debug", action="store_true",
                        help="Mostra il debug del retrieval ibrido (distanza vector search, keyword BM25 e contributo parole chiave)")
    args = parser.parse_args()

    db_path = get_db_path(args.env, args.chunk_size, args.metodo)

    print(f"🔄 Avvio test pipeline RAG (Ibrido) — env: {args.env.upper()} | metodo: {args.metodo} | chunk_size: {args.chunk_size}")

    if not os.path.exists(db_path):
        print(f"❌ ERRORE: Database vettoriale non trovato in '{db_path}'.")
        print("   👉 Assicurati di eseguire lo script dalla root del progetto (ARIS/).")
        return

    print(f"🗄️ Caricamento Vector DB da: {db_path}")
    embeddings = get_embeddings(args.env)
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # Costruisce il retriever ibrido BM25 + Vector
    print("🤝 Costruzione Retriever Ibrido (50% BM25 + 50% Vector Search)...")
    try:
        retriever = build_hybrid_retriever(db, k=3)
        retriever = DebugHybridRetriever(retriever, db, k=3, debug=args.debug)
    except RuntimeError as e:
        print(e)
        return

    try:
        chain = setup_rag_chain(retriever, env=args.env)
    except ValueError as e:
        print(e)
        return

    # Caricamento domande se specificato --question
    questions_to_run = []
    if args.question:
        try:
            indices = [int(x.strip()) for x in args.question.split(",") if x.strip()]
            invalid_indices = [idx for idx in indices if idx < 1 or idx > 100]
            if invalid_indices:
                print(f"❌ ERRORE: Gli indici delle domande devono essere compresi tra 1 e 100. Indici non validi: {invalid_indices}")
                return
            csv_path = os.path.join("tests", "test_questions_it.csv")
            loaded_questions = load_questions_from_csv(csv_path, indices)
            for idx in indices:
                if idx in loaded_questions:
                    questions_to_run.append((idx, loaded_questions[idx]))
                else:
                    print(f"⚠️ Domanda con indice {idx} non trovata nel file CSV.")
        except ValueError:
            print("❌ ERRORE: Formato di --question non valido. Usa numeri separati da virgole (es. --question 1,2,3)")
            return
    else:
        questions_to_run = [(None, args.query)]

    for idx, query in questions_to_run:
        if idx is not None:
            print(f"\n================ DOMANDA {idx} ================")
        else:
            print(f"\n🗣️ Domanda: {query}")
        print(f"🗣️ Testo Domanda: {query}")
        print("⏳ Generazione risposta in corso...\n")

        risposta = answer_question(chain, query)

        print("================ RISPOSTA ================")
        print(risposta)
        print("==========================================")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    os.chdir(project_root)

    main()
