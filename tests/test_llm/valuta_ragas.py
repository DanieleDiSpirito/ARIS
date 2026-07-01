import os
import sys
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
import asyncio
from datasets import Dataset
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings, HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
# Disabilita il tracing di LangSmith per evitare errori di limite di quota nei benchmark
os.environ["LANGCHAIN_TRACING_V2"] = "false"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)

# Aggiungi src/rag_pipeline al path
sys.path.append(os.path.join(PROJECT_ROOT, "src", "rag_pipeline"))

import rag_pipeline_hybrid
import rag_pipeline_rerank
import rag_pipeline_graph
from rag_pipeline_hybrid import get_db_path, get_embeddings, COLLECTION_NAME


def run_ragas_evaluation(env: str, lang: str, chunk_size: int, limit: int = None, k: int = 3, model: str = None, rag_type: str = "ibrido", metodo: str = "pdf4llm"):
    test_file = os.path.join(TESTS_DIR, f"test_questions_{lang}.csv")
    if not os.path.exists(test_file):
        print(f"❌ File di test non trovato: {test_file}")
        return

    df = pd.read_csv(test_file)
    if limit:
        # Campiona casualmente in modo riproducibile e ordina per indice originale
        df = df.sample(n=min(limit, len(df)), random_state=42).sort_index()

    db_path = os.path.join(PROJECT_ROOT, get_db_path(env, chunk_size, metodo))
    if not os.path.exists(db_path):
        print(f"❌ Database non trovato in: {db_path}")
        return

    print(f"🗄️ Caricamento DB e Retriever ({env}, chunk: {chunk_size}, lingua: {lang}, RAG: {rag_type})...")
    embedder = get_embeddings(env)
    lc_chroma = Chroma(
        persist_directory=db_path,
        collection_name=COLLECTION_NAME,
        embedding_function=embedder
    )
    if rag_type == "puro":
        retriever = lc_chroma.as_retriever(search_kwargs={"k": k})
        rag_chain = rag_pipeline_hybrid.setup_rag_chain(retriever, env=env, model_name=model)
    elif rag_type == "ibrido":
        retriever = rag_pipeline_hybrid.build_hybrid_retriever(lc_chroma, k=k)
        rag_chain = rag_pipeline_hybrid.setup_rag_chain(retriever, env=env, model_name=model)
    elif rag_type == "rerank":
        # Per Rerank, recuperiamo più candidati dal retriever (es. k * 2) per poi filtrarli col reranker
        ensemble = rag_pipeline_rerank.build_hybrid_retriever(lc_chroma, k=max(k * 2, 6))
        retriever = rag_pipeline_rerank.HybridRerankRetriever(ensemble, top_n=k)
        rag_chain = rag_pipeline_rerank.setup_rag_chain(retriever, env=env, model_name=model)
    elif rag_type == "graph":
        retriever = rag_pipeline_graph.build_graph_retriever(lc_chroma, k_base=max(k - 1, 2), top_n=k)
        rag_chain = rag_pipeline_graph.setup_rag_chain(retriever, env=env, model_name=model)
    else:
        raise ValueError(f"rag_type non valido: '{rag_type}'")

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    print(f"\n🚀 Generazione risposte e contesti per RAGAS su {len(df)} domande...\n")
    for index, row in df.iterrows():
        question = row['question']
        expected_answer = row['expected_answer']
        
        print(f"[{index+1}/{len(df)}] Q: {question}")
        
        # 1. Recupera i contesti
        try:
            retrieved_docs = retriever.invoke(question)
            contexts = [doc.page_content for doc in retrieved_docs]
        except Exception as e:
            print(f"   ⚠️ Errore di retrieval: {e}")
            contexts = []

        # 2. Genera la risposta
        try:
            answer = rag_chain.invoke({"question": question, "history": ""})
        except Exception as e:
            print(f"   ⚠️ Errore di generazione: {e}")
            answer = f"ERRORE: {str(e)}"

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(expected_answer)

        print(f"   ✅ Completata.")

    print("\n⚖️ Configurazione dei modelli RAGAS Judge...")
    # Configurazione LLM ed Embeddings per il Judge di Ragas
    if env == "cloud":
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("❌ ERRORE: Variabile OPENAI_API_KEY non trovata nel file .env")
        
        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        
        judge_llm = llm_factory(
            model="openai/gpt-4o-mini",
            client=client,
            temperature=0.0,
            max_tokens=3000
        )
        judge_embeddings = RagasOpenAIEmbeddings(
            client=client,
            model="text-embedding-3-small"
        )
    else:
        client = AsyncOpenAI(
            api_key="lm-studio",
            base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
        )
        judge_llm = llm_factory(
            model=os.getenv("LOCAL_LLM_MODEL", "lm-studio"),
            client=client
        )
        judge_embeddings = RagasHuggingFaceEmbeddings(
            model="BAAI/bge-m3",
            device="cpu",
            normalize_embeddings=True
        )

    # Inizializza le metriche usando l'API moderna
    faithfulness_metric = Faithfulness(llm=judge_llm)
    answer_relevancy_metric = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
    context_precision_metric = ContextPrecision(llm=judge_llm)
    context_recall_metric = ContextRecall(llm=judge_llm, embeddings=judge_embeddings)

    async def evaluate_dataset(
        questions, answers, contexts_list, ground_truths,
        faithfulness_metric, answer_relevancy_metric, context_precision_metric, context_recall_metric
    ):
        semaphore = asyncio.Semaphore(10)  # Limita la concorrenza a 10 richieste parallele
        
        async def evaluate_row(q, a, ctx, gt):
            async with semaphore:
                try:
                    f_res = await faithfulness_metric.ascore(
                        user_input=q,
                        response=a,
                        retrieved_contexts=ctx
                    )
                    f_val = f_res.value if f_res is not None else 0.0
                except Exception as e:
                    print(f"   ⚠️ Errore Faithfulness per '{q[:30]}...': {e}")
                    f_val = None
                    
                try:
                    ar_res = await answer_relevancy_metric.ascore(
                        user_input=q,
                        response=a
                    )
                    ar_val = ar_res.value if ar_res is not None else 0.0
                except Exception as e:
                    print(f"   ⚠️ Errore AnswerRelevancy per '{q[:30]}...': {e}")
                    ar_val = None

                try:
                    cp_res = await context_precision_metric.ascore(
                        user_input=q,
                        reference=gt,
                        retrieved_contexts=ctx
                    )
                    cp_val = cp_res.value if cp_res is not None else 0.0
                except Exception as e:
                    print(f"   ⚠️ Errore ContextPrecision per '{q[:30]}...': {e}")
                    cp_val = None

                try:
                    cr_res = await context_recall_metric.ascore(
                        user_input=q,
                        retrieved_contexts=ctx,
                        reference=gt
                    )
                    cr_val = cr_res.value if cr_res is not None else 0.0
                except Exception as e:
                    print(f"   ⚠️ Errore ContextRecall per '{q[:30]}...': {e}")
                    cr_val = None

                return {
                    "question": q,
                    "answer": a,
                    "contexts": ctx,
                    "ground_truth": gt,
                    "faithfulness": f_val,
                    "answer_relevancy": ar_val,
                    "context_precision": cp_val,
                    "context_recall": cr_val
                }

        tasks = [
            evaluate_row(q, a, ctx, gt) 
            for q, a, ctx, gt in zip(questions, answers, contexts_list, ground_truths)
        ]
        return await asyncio.gather(*tasks)

    print("⏳ Avvio calcolo asincrono delle metriche Ragas (Faithfulness, Relevancy, Precision, Recall)...")
    try:
        results_list = asyncio.run(evaluate_dataset(
            questions, answers, contexts_list, ground_truths,
            faithfulness_metric, answer_relevancy_metric, context_precision_metric, context_recall_metric
        ))
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione di Ragas: {e}")
        return

    model_suffix = model.replace('/', '_').replace(':', '_') if model else ("default_local" if env == "locale" else "default_cloud")
    # Salva e visualizza i risultati
    out_dir = os.path.join(TESTS_DIR, "results_llm", "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"ragas_eval_{env}_{lang}_{chunk_size}_{rag_type}_{metodo}_{model_suffix}.csv")

    df_result = pd.DataFrame(results_list)
    df_result.to_csv(out_file, index=False, encoding='utf-8')

    print(f"\n✅ Valutazione completata con successo! Risultati salvati in:\n{out_file}")
    
    # Calcola le medie delle metriche per il report rapido
    faithfulness_mean = df_result['faithfulness'].mean()
    answer_relevancy_mean = df_result['answer_relevancy'].mean()
    context_precision_mean = df_result['context_precision'].mean()
    context_recall_mean = df_result['context_recall'].mean()

    # Salva il report in quick_eval
    quick_eval_dir = os.path.join(TESTS_DIR, "results_llm", "quick_eval")
    os.makedirs(quick_eval_dir, exist_ok=True)
    quick_eval_file = os.path.join(quick_eval_dir, f"ragas_eval_summary_{env}_{lang}_{chunk_size}_{rag_type}_{metodo}_{model_suffix}.md")
    
    summary_md = f"""# Report RAGAS Summary ({env}, {lang}, chunk: {chunk_size}, metodo: {metodo})

| Metric | Value |
| :--- | :--- |
| **RAG Algorithm** | {rag_type.capitalize()} |
| **LLM Model** | {model if model else 'Default'} |
| **Faithfulness (Fedeltà)** | {faithfulness_mean:.4f} |
| **Answer Relevancy (Pertinenza)** | {answer_relevancy_mean:.4f} |
| **Context Precision (Prec. Contesto)** | {context_precision_mean:.4f} |
| **Context Recall (Richiamo Contesto)** | {context_recall_mean:.4f} |
"""
    with open(quick_eval_file, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"⚡ Report di sintesi salvato in:\n{quick_eval_file}")
    
    print("\n📊 --- REPORT FINALE RAGAS ---")
    print(f"  RAG Algorithm                    : {rag_type.capitalize()}")
    print(f"  LLM Model                        : {model if model else 'Default'}")
    print(f"  Faithfulness (Fedeltà)           : {faithfulness_mean:.4f}")
    print(f"  Answer Relevancy (Pertinenza)     : {answer_relevancy_mean:.4f}")
    print(f"  Context Precision (Prec. Contesto): {context_precision_mean:.4f}")
    print(f"  Context Recall (Richiamo Contesto): {context_recall_mean:.4f}")
    print("------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valuta la pipeline RAG usando il framework Ragas")
    parser.add_argument("--env", type=str, choices=["locale", "cloud"], default="cloud", help="Ambiente LLM/Embedding")
    parser.add_argument("--lang", type=str, choices=["it", "en"], default="it", help="Lingua del test set")
    parser.add_argument("--chunk_size", type=int, default=700, help="Dimensione dei chunk usata per caricare il DB")
    parser.add_argument("--limit", type=int, default=None, help="Limita il numero di domande da valutare")
    parser.add_argument("--k", type=int, default=3, help="Numero di chunk da recuperare")
    parser.add_argument("--model", type=str, default=None,
                        help="Modello LLM da usare per la generazione (es. openai/gpt-4o-mini, google/gemini-2.5-flash, google/gemini-3.5-flash, meta-llama/llama-3.3-70b-instruct, qwen/qwen-2.5-72b-instruct, deepseek/deepseek-chat)")
    parser.add_argument("--rag_type", type=str, choices=["puro", "ibrido", "rerank", "graph"], default="ibrido",
                        help="Algoritmo RAG da usare: 'puro' (solo Vector Search), 'ibrido' (BM25 + Vector Search), 'rerank' (BM25 + Vector + Re-ranking) o 'graph' (GraphRAG leggero)")
    parser.add_argument(
        "--metodo", "-m", 
        type=str, 
        choices=["euristico", "pdf4llm", "docling", "llamaparse", "qwen"],
        default="pdf4llm",
        help="Metodo di estrazione dei PDF da testare."
    )
    args = parser.parse_args()

    run_ragas_evaluation(args.env, args.lang, args.chunk_size, args.limit, args.k, args.model, args.rag_type, args.metodo)
