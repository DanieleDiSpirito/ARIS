import os
import argparse
import re
import networkx as nx
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


import hashlib

def _get_node_key(doc: Document) -> str:
    chunk_id = doc.metadata.get("chunk_id", "unknown")
    text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:8]
    return f"{chunk_id}_{text_hash}"


class LightweightGraphRAG:
    """Mappa le relazioni di vicinanza e le entità tecniche condivise (codici allarmi, part number, connettori) tra i chunk."""
    def __init__(self, chunks: list, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        from sentence_transformers import CrossEncoder
        import torch
        self.G = nx.Graph()
        self.chunks = chunks
        self.chunk_dict = {}
        self.build_graph()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Caricamento del Cross-Encoder per GraphRAG: {model_name} su {device.upper()}...")
        self.reranker = CrossEncoder(model_name, device=device)

    def build_graph(self):
        # 1. Aggiungi i nodi (i chunk) con chiavi univoche
        for doc in self.chunks:
            node_key = _get_node_key(doc)
            self.G.add_node(node_key, doc=doc)
            self.chunk_dict[node_key] = doc
            
            # Estrai le entità dal testo
            entities = self.extract_entities(doc.page_content)
            doc.metadata["entities"] = entities
            
        # 2. Raggruppa i nodi per entità per creare archi
        entity_to_chunks = {}
        for doc in self.chunks:
            node_key = _get_node_key(doc)
            for ent in doc.metadata.get("entities", []):
                entity_to_chunks.setdefault(ent, []).append(node_key)
                
        # 3. Aggiungi archi basati sulle entità condivise
        for ent, node_keys in entity_to_chunks.items():
            if len(node_keys) > 1:
                for i in range(len(node_keys)):
                    for j in range(i + 1, len(node_keys)):
                        c1, c2 = node_keys[i], node_keys[j]
                        if self.G.has_edge(c1, c2):
                            self.G[c1][c2]["weight"] += 1.0
                            if ent not in self.G[c1][c2]["entities"]:
                                self.G[c1][c2]["entities"].append(ent)
                        else:
                            self.G.add_edge(c1, c2, weight=1.0, entities=[ent])
                            
        # 4. Aggiungi archi per adiacenza sequenziale (stesso file, pagina adiacente)
        file_groups = {}
        for doc in self.chunks:
            file_name = doc.metadata.get("file_name")
            if file_name:
                file_groups.setdefault(file_name, []).append(doc)
                
        for file_name, file_docs in file_groups.items():
            def get_page_num(d):
                p = d.metadata.get("page", "0")
                m = re.search(r'\d+', str(p))
                return int(m.group()) if m else 0
            sorted_docs = sorted(file_docs, key=get_page_num)
            for i in range(len(sorted_docs) - 1):
                d1 = sorted_docs[i]
                d2 = sorted_docs[i+1]
                p1 = get_page_num(d1)
                p2 = get_page_num(d2)
                if abs(p1 - p2) <= 1:
                    c1 = _get_node_key(d1)
                    c2 = _get_node_key(d2)
                    if self.G.has_edge(c1, c2):
                        self.G[c1][c2]["weight"] += 0.5
                    else:
                        self.G.add_edge(c1, c2, weight=0.5, entities=["adjacency"])

    def extract_entities(self, text: str) -> list:
        entities = []
        # Error codes: SRVO-004, OHAL-001, ecc.
        error_codes = re.findall(r'\b[A-Z]{4}-\d{3}\b', text)
        entities.extend([code.upper() for code in error_codes])
        
        # Part numbers: A20B-8200-0790, A05B-2650-H001, ecc.
        part_numbers = re.findall(r'\bA\d{2}B-\d{4}-\d{4}\b|\bA\d{2}B-\d{3}-\d{4}\b|\bA\d{2}B-\d{4}-[A-Z]\d{3}\b|\bA05B-\d{4}-[A-Z]\d{3}\b|\bA06B-\d{4}-[A-Z]\d{3}\b', text)
        entities.extend([pn.upper() for pn in part_numbers])
        
        # Connectors: CRMA52A, JD1A, ecc.
        connectors = re.findall(r'\b(?:CRMA\d{2}[A-Z]?|JD\d{1}[A-Z]|CNJ[xXyYzZ\d]+|CRR\d{2})\b', text)
        entities.extend([conn.upper() for conn in connectors])
        
        return list(set(entities))

class GraphExpandedRetriever:
    """Wrapper Retriever LangChain compatibile che implementa l'espansione dei chunk via Grafo."""
    def __init__(self, base_retriever, graph_rag, k_base: int = 12, top_n: int = 4, debug: bool = False):
        self.base_retriever = base_retriever
        self.graph_rag = graph_rag
        self.k_base = k_base
        self.top_n = top_n
        self.debug = debug

    def invoke(self, query: str) -> list:
        # 1. Recupera un pool più ampio dall'ibrido base (es. k_base = 12)
        base_docs = self.base_retriever.invoke(query)[:self.k_base]
        
        if self.debug:
            print("\n🔎 [Debug GraphRAG] Chunk presi in maniera IBRIDA:")
            for idx, doc in enumerate(base_docs):
                chunk_id = doc.metadata.get("chunk_id", "unknown")
                file_name = doc.metadata.get("file_name", "unknown")
                page = doc.metadata.get("page", "N/A")
                print(f"  [{idx+1}] ID: {chunk_id} | File: {file_name} | Pagina: {page} | Snippet: {doc.page_content[:60].replace(chr(10), ' ')}...")
        
        if not base_docs:
            return []
            
        # Rimuove duplicati basati sul contenuto testuale
        seen = set()
        unique_base_docs = []
        for doc in base_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_base_docs.append(doc)
                
        # 2. Primo Rerank sui chunk base per selezionare i top_n base docs
        pairs = [[query, doc.page_content] for doc in unique_base_docs]
        scores_base = self.graph_rag.reranker.predict(pairs)
        scored_base_docs = sorted(zip(scores_base, unique_base_docs), key=lambda x: x[0], reverse=True)
        
        # Selezioniamo fino a self.top_n chunk base
        selected_base = [doc for score, doc in scored_base_docs[:self.top_n]]
        
        if self.debug:
            print("\n🔎 [Debug GraphRAG] --- RERANK 1 (Su tutti i chunk base) ---")
            for idx, (score, doc) in enumerate(scored_base_docs):
                chunk_id = doc.metadata.get("chunk_id", "unknown")
                file_name = doc.metadata.get("file_name", "unknown")
                page = doc.metadata.get("page", "N/A")
                is_selected = "SELEZIONATO per espansione" if idx < len(selected_base) else "Escluso"
                print(f"  [{idx+1}] Score: {score:.4f} | ID: {chunk_id} | File: {file_name} | Pagina: {page} | Stato: {is_selected}")
                print(f"      Snippet: {doc.page_content[:80].replace(chr(10), ' ')}...")
                
        # 3. Espansione tramite il Grafo per tutti i selected_base
        candidate_pool = list(selected_base)
        final_keys = {_get_node_key(doc) for doc in selected_base}
        
        if self.debug:
            print("\n🔎 [Debug GraphRAG] --- ESPANSIONE VICINI NEL GRAFO ---")

        for doc in selected_base:
            rk = _get_node_key(doc)
            if not self.graph_rag.G.has_node(rk):
                if self.debug:
                    print(f"  * Chunk {rk} non trovato nel grafo.")
                continue
            
            # Trova i vicini
            neighbors_with_weights = []
            for neighbor in self.graph_rag.G.neighbors(rk):
                if neighbor not in final_keys:
                    edge_data = self.graph_rag.G.get_edge_data(rk, neighbor)
                    weight = edge_data.get("weight", 1.0)
                    neighbors_with_weights.append((neighbor, weight))
            
            # Trova il miglior vicino
            best_neighbor = None
            if neighbors_with_weights:
                best_neighbor = sorted(neighbors_with_weights, key=lambda x: x[1], reverse=True)[0][0]
                
            if self.debug:
                doc_rk = self.graph_rag.chunk_dict[rk]
                print(f"  * Relazioni per il chunk {rk} ({doc_rk.metadata.get('file_name')} | Pag: {doc_rk.metadata.get('page')}):")
                all_neighbors = list(self.graph_rag.G.neighbors(rk))
                if not all_neighbors:
                    print("    (Nessun collegamento nel grafo)")
                else:
                    for neighbor in all_neighbors:
                        edge_data = self.graph_rag.G.get_edge_data(rk, neighbor)
                        weight = edge_data.get("weight", 1.0)
                        entities = edge_data.get("entities", [])
                        n_doc = self.graph_rag.chunk_dict[neighbor]
                        if neighbor in final_keys and neighbor != best_neighbor:
                            status = "Già presente nel pool candidati"
                        elif neighbor == best_neighbor:
                            status = "SELEZIONATO come vicino (migliore)"
                        else:
                            status = "Escluso (non il migliore)"
                        print(f"    - collegato a -> {neighbor} ({n_doc.metadata.get('file_name')} | Pag: {n_doc.metadata.get('page')})")
                        print(f"      Peso: {weight} | Entità in comune: {entities} | Stato: {status}")

            if best_neighbor:
                final_keys.add(best_neighbor)
                candidate_pool.append(self.graph_rag.chunk_dict[best_neighbor])
                
        # 4. Secondo Rerank sul pool combinato (base + vicini) per selezionare i top_n finali
        # Rimuove eventuali ulteriori duplicati per sicurezza
        seen_pool = set()
        unique_candidate_pool = []
        for doc in candidate_pool:
            if doc.page_content not in seen_pool:
                seen_pool.add(doc.page_content)
                unique_candidate_pool.append(doc)
                
        pairs_pool = [[query, doc.page_content] for doc in unique_candidate_pool]
        scores_pool = self.graph_rag.reranker.predict(pairs_pool)
        scored_pool = sorted(zip(scores_pool, unique_candidate_pool), key=lambda x: x[0], reverse=True)
        
        if self.debug:
            print("\n🔎 [Debug GraphRAG] --- RERANK 2 (Su pool combinato: base + vicini) ---")
            for idx, (score, doc) in enumerate(scored_pool):
                chunk_id = doc.metadata.get("chunk_id", "unknown")
                file_name = doc.metadata.get("file_name", "unknown")
                page = doc.metadata.get("page", "N/A")
                is_selected = "SELEZIONATO (top_n finale)" if idx < self.top_n else "Escluso"
                print(f"  [{idx+1}] Score: {score:.4f} | ID: {chunk_id} | File: {file_name} | Pagina: {page} | Stato: {is_selected}")
                print(f"      Snippet: {doc.page_content[:80].replace(chr(10), ' ')}...")
            print("--------------------------------------------------\n")
            
        return [doc for score, doc in scored_pool[:self.top_n]]


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

    print(f"\n--- 🔎 CHUNK INVIATI ALL'LLM — GraphRAG ({len(docs)} totali) ---")
    formatted_chunks = []
    for i, doc in enumerate(docs):
        file_name = doc.metadata.get("file_name", "Documento Sconosciuto")
        page = doc.metadata.get("original_source_page", doc.metadata.get("page", "N/A"))
        ents = doc.metadata.get("entities", [])

        print(f"  [{i+1}] {file_name} | Pag: {page} | Entità: {ents} | Testo: {doc.page_content[:90].replace(chr(10), ' ')}...")

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
            streaming=False
        )
        llm_technical = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key="lm-studio",
            temperature=0.0,
            streaming=True
        )
        llm_general = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key="lm-studio",
            temperature=0.7,
            streaming=True
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


def build_hybrid_retriever(db: Chroma, k: int = 3) -> EnsembleRetriever:
    """Costruisce un retriever ibrido base (BM25 + Vector Search)."""
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


def build_graph_retriever(db: Chroma, k_base: int = 12, top_n: int = 4, debug: bool = False) -> GraphExpandedRetriever:
    """Inizializza il retriever espanso con il Grafo di Conoscenza relazionale."""
    ensemble = build_hybrid_retriever(db, k=k_base)
    
    all_data = db.get()
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
        if text
    ]
    
    print(f"🕸️ Costruzione Grafo di Conoscenza (Lightweight) con {len(docs)} chunk...")
    graph_rag = LightweightGraphRAG(docs)
    print(f"✅ Grafo costruito: {graph_rag.G.number_of_nodes()} nodi, {graph_rag.G.number_of_edges()} archi.")
    
    return GraphExpandedRetriever(ensemble, graph_rag, k_base=k_base, top_n=top_n, debug=debug)


def main():
    parser = argparse.ArgumentParser(description="Testa la RAG Pipeline con Grafo (GraphRAG) dal terminale")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="locale",
                        help="Scegli tra 'locale' (LM Studio) o 'cloud' (OpenRouter)")
    parser.add_argument("--chunk_size", type=int, default=700,
                        help="Dimensione dei chunk usata durante l'ingestion.")
    parser.add_argument("--metodo", type=str, default="pdf4llm",
                        choices=["euristico", "docling", "llamaparse", "qwen", "pdf4llm", "mineru"],
                        help="Metodo di estrazione da utilizzare (default: pdf4llm)")
    parser.add_argument("--query", type=str, default="Cosa significa l'allarme SRVO-004?",
                        help="La domanda da porre al sistema")
    parser.add_argument("--debug", action="store_true",
                        help="Mostra il debug del GraphRAG (chunk ibridi, collegamenti del grafo, classifica rerank)")
    parser.add_argument("--question", type=str, default=None,
                        help="Indice o lista di indici di domande (da 1 a 100, es: 1 o 1,2,5) dal file tests/test_questions_it.csv")
    args = parser.parse_args()

    db_path = get_db_path(args.env, args.chunk_size, args.metodo)

    print(f"🔄 Avvio test pipeline GraphRAG — env: {args.env.upper()} | metodo: {args.metodo} | chunk_size: {args.chunk_size}")

    if not os.path.exists(db_path):
        print(f"❌ ERRORE: Database vettoriale non trovato in '{db_path}'.")
        return

    embeddings = get_embeddings(args.env)
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    retriever = build_graph_retriever(db, k_base=12, top_n=4, debug=args.debug)

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
