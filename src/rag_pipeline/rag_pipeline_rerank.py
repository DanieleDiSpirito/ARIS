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

def get_db_path(env: str, chunk_size: int, metodo: str = "pdf4llm") -> str:
    """Restituisce il percorso corretto del Vector DB in base all'ambiente, alla dimensione dei chunk e al metodo di estrazione."""
    return os.path.join("vector_db", f"chroma_{metodo}_{env}_{chunk_size}")


def get_embeddings(env: str):
    """Seleziona il modello di embedding coerente con quello usato durante l'ingestion."""
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


class HybridRerankRetriever:
    """Retriever Custom che unisce BM25 + Vector Search (k=12) e applica Cross-Encoder Reranking (top_n=3)."""
    def __init__(self, ensemble_retriever, db=None, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", top_n: int = 3, debug: bool = False):
        from sentence_transformers import CrossEncoder
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ensemble_retriever = ensemble_retriever
        self.db = db
        print(f"🧠 Caricamento del Cross-Encoder Reranker: {model_name} su {device.upper()}...")
        self.model = CrossEncoder(model_name, device=device)
        self.top_n = top_n
        self.debug = debug

    def invoke(self, query: str) -> list:
        if self.debug:
            # 1. Vector Search manualmente (k=6, since ensemble has k=6 for each)
            chroma_results = []
            if self.db is not None:
                chroma_results = self.db.similarity_search_with_score(query, k=6)
            
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
                bm25_results = sorted(scored_docs, key=lambda x: x[0], reverse=True)[:6]

            if self.db is not None:
                print("\n🔎 [Debug Rerank] --- VECTOR SEARCH (Chroma) ---")
                for idx, (doc, score) in enumerate(chroma_results):
                    chunk_id = doc.metadata.get("chunk_id", "unknown")
                    file_name = doc.metadata.get("file_name", "unknown")
                    page = doc.metadata.get("page", "N/A")
                    print(f"  [{idx+1}] Distanza: {score:.4f} | ID: {chunk_id} | File: {file_name} | Pagina: {page}")
                    print(f"      Snippet: {doc.page_content[:80].replace(chr(10), ' ')}...")
            else:
                print("\n🔎 [Debug Rerank] --- VECTOR SEARCH (Chroma) non disponibile (nessun db passato) ---")

            print("\n🔎 [Debug Rerank] --- KEYWORD SEARCH (BM25) ---")
            for idx, (score, doc, contribs) in enumerate(bm25_results):
                chunk_id = doc.metadata.get("chunk_id", "unknown")
                file_name = doc.metadata.get("file_name", "unknown")
                page = doc.metadata.get("page", "N/A")
                contribs_str = ", ".join([f"'{w}': {c:.3f}" for w, c in contribs[:3]])
                print(f"  [{idx+1}] Score BM25: {score:.4f} | ID: {chunk_id} | File: {file_name} | Pagina: {page}")
                print(f"      Contribuzioni parole chiave: {contribs_str}")
                print(f"      Snippet: {doc.page_content[:80].replace(chr(10), ' ')}...")

        # 1. Recupera i chunk candidati (k=6 per ciascuno)
        docs = self.ensemble_retriever.invoke(query)
        if not docs:
            return []
            
        # Rimuove duplicati basati sul contenuto testuale
        seen = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
                
        # 2. Esegui il Re-ranking
        pairs = [[query, doc.page_content] for doc in unique_docs]
        scores = self.model.predict(pairs)
        
        # Associa i punteggi ai documenti e ordina in modo decrescente
        scored_docs = sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)
        
        if self.debug:
            print("\n🔎 [Debug Rerank] --- CLASSIFICA RERANK (Cross-Encoder) ---")
            for idx, (score, doc) in enumerate(scored_docs):
                chunk_id = doc.metadata.get("chunk_id", "unknown")
                file_name = doc.metadata.get("file_name", "unknown")
                page = doc.metadata.get("page", "N/A")
                status = "SELEZIONATO (top_n)" if idx < self.top_n else "Escluso"
                print(f"  [{idx+1}] Score: {score:.4f} | ID: {chunk_id} | File: {file_name} | Pagina: {page} | Stato: {status}")
                print(f"      Snippet: {doc.page_content[:80].replace(chr(10), ' ')}...")
            print("--------------------------------------------------\n")

        # Restituisci i primi top_n
        reranked_docs = [doc for _, doc in scored_docs[:self.top_n]]
        return reranked_docs


def build_prompt():
    """Costruisce il prompt con System+Human separati."""
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

    print(f"\n--- 🔎 CHUNK INVIATI ALL'LLM — Re-ranking (Cross-Encoder) ({len(docs)} totali) ---")
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
    """Configura la pipeline RAG collegando retriever, prompt e LLM."""
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

    normalize_input = RunnableLambda(
        lambda x: {"question": x, "history": ""} if isinstance(x, str) else {
            "question": x.get("question", ""),
            "history": x.get("history", "")
        }
    )

    def classify_intent(inputs):
        question = inputs.get("question", "")
        q_clean = question.strip().lower().rstrip("!?.,")
        greetings = {
            "ciao", "buongiorno", "buonasera", "salve", "hello", "hi", "hey",
            "grazie", "grazie mille", "thank you", "thanks", "prego"
        }
        if q_clean in greetings:
            return "GENERAL"
            
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


def build_hybrid_retriever(db: Chroma, k: int = 6) -> EnsembleRetriever:
    """Costruisce un retriever ibrido (BM25 + Vector Search) con un k più alto (default: 6 per retriever = 12 totali)."""
    chroma_retriever = db.as_retriever(search_kwargs={"k": k})

    all_data = db.get()
    if not all_data or not all_data.get("documents"):
        raise RuntimeError("❌ Nessun documento trovato nel database vettoriale. Impossibile costruire BM25.")

    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
        if text
    ]
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble


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
    parser = argparse.ArgumentParser(description="Testa la RAG Pipeline Ibrida con Re-ranking dal terminale")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale",
                        help="Scegli tra 'locale' (LM Studio) o 'cloud' (OpenRouter)")
    parser.add_argument("--chunk_size", type=int, default=700,
                        help="Dimensione dei chunk usata durante l'ingestion.")
    parser.add_argument("--metodo", type=str, default="pdf4llm",
                        choices=["euristico", "docling", "llamaparse", "qwen", "pdf4llm", "mineru"],
                        help="Metodo di estrazione da utilizzare (default: pdf4llm)")
    parser.add_argument("--query", type=str, default="Cosa significa l'allarme SRVO-004?",
                        help="La domanda da porre al sistema")
    parser.add_argument("--question", type=str, default=None,
                        help="Indice o lista di indici di domande (da 1 a 100, es: 1 o 1,2,5) dal file tests/test_questions_it.csv")
    parser.add_argument("--debug", action="store_true",
                        help="Mostra il debug dettagliato (distanze Chroma, score e parole chiave BM25, score Reranker)")
    args = parser.parse_args()

    db_path = get_db_path(args.env, args.chunk_size, args.metodo)

    print(f"🔄 Avvio test pipeline RAG (Re-ranking) — env: {args.env.upper()} | metodo: {args.metodo} | chunk_size: {args.chunk_size}")

    if not os.path.exists(db_path):
        print(f"❌ ERRORE: Database vettoriale non trovato in '{db_path}'.")
        return

    embeddings = get_embeddings(args.env)
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # Crea il retriever ensemble ibrido con k=6 (restituirà fino a 12 chunk candidati)
    ensemble_retriever = build_hybrid_retriever(db, k=6)
    
    # Applica il RerankRetriever (top_n=3)
    retriever = HybridRerankRetriever(ensemble_retriever, db, top_n=3, debug=args.debug)

    chain = setup_rag_chain(retriever, env=args.env)

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
