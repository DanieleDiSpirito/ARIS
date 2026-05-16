"""
retrieval_metrics.py
====================
Modulo condiviso per il calcolo delle metriche di retrieval RAG.
Ispirato alla Fase 13 — Validazione del sistema (INFO.md).

Metriche implementate:
  - Hit Rate@k  : % domande per cui il chunk corretto è tra i top-k
  - Precision@k : media delle precisioni per ogni query (chunk corretti / k)
  - Recall@k    : media dei recall per ogni query (chunk corretti / totali attesi)
  - MRR         : Mean Reciprocal Rank
  - Tempo medio retrieval (secondi)
  - Breakdown per categoria e per difficoltà
"""

import time
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional


def _pagine_match(found_p: str, exp_p: str, tolleranza: int = 0) -> bool:
    found_p = str(found_p).strip()
    exp_p   = str(exp_p).strip()

    if found_p == exp_p:
        return True

    if tolleranza == 0:
        return False

    try:
        f_match = re.search(r'\d+', found_p)
        e_match = re.search(r'\d+', exp_p)
        if f_match and e_match:
            f_prefix = found_p[:f_match.start()]
            e_prefix = exp_p[:e_match.start()]
            if f_prefix == e_prefix:
                if abs(int(f_match.group()) - int(e_match.group())) <= tolleranza:
                    return True
    except Exception:
        pass
    return False


def _chunk_corretto(meta: Dict, expected_file: str, expected_page: str,
                    tolleranza: int = 0) -> bool:
    if not meta:
        return False
    if meta.get('file_name') != expected_file:
        return False
    return _pagine_match(str(meta.get('page', '')), expected_page, tolleranza)


def calcola_metriche_query(
    retrieved_metas: List[List[Dict]],
    expected_files: List[str],
    expected_pages: List[str],
    categories: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    k: int = 3,
    tolleranza: int = 0,
    tempi_retrieval: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Calcola Hit Rate@k, Precision@k, Recall@k e MRR su un batch di query.

    Parametri
    ---------
    retrieved_metas  : lista (per query) di liste di metadati ordinati per ranking
    expected_files   : file atteso per ogni query
    expected_pages   : pagina attesa per ogni query
    categories       : categoria di ogni domanda (opzionale, per breakdown)
    difficulties     : difficoltà di ogni domanda (opzionale, per breakdown)
    k                : numero di chunk considerati (default 3)
    tolleranza       : tolleranza di pagina ±N (0 = esatta)
    tempi_retrieval  : secondi impiegati per ogni retrieval (opzionale)

    Ritorna
    -------
    dict con tutte le metriche calcolate
    """
    n = len(expected_files)
    if n == 0:
        return {}

    hits      = 0      # per Hit Rate@k
    prec_sum  = 0.0    # per Precision@k
    rr_sum    = 0.0    # per MRR

    # Per breakdown
    cat_stats  = defaultdict(lambda: {"hits": 0, "total": 0})
    diff_stats = defaultdict(lambda: {"hits": 0, "total": 0})

    for i, metas in enumerate(retrieved_metas):
        ef = expected_files[i]
        ep = str(expected_pages[i])
        cat  = (categories[i]   if categories   else None)
        diff = (difficulties[i] if difficulties else None)

        top_k = metas[:k]

        # Hit Rate: il chunk corretto è in top-k?
        first_hit_rank = None
        corretti_in_k  = 0

        for rank, meta in enumerate(top_k, start=1):
            if _chunk_corretto(meta, ef, ep, tolleranza):
                corretti_in_k += 1
                if first_hit_rank is None:
                    first_hit_rank = rank

        trovato = (first_hit_rank is not None)
        if trovato:
            hits += 1

        # Precision@k = chunk corretti nei top-k / k
        # (per RAG mono-risposta attesa: al massimo 1 corretto → Precision@k = 1/k se trovato)
        prec_sum += corretti_in_k / k

        # MRR: 1/rank del primo risultato corretto (0 se non trovato)
        rr_sum += (1.0 / first_hit_rank) if first_hit_rank else 0.0

        # Recall@k = chunk corretti nei top-k / totale attesi
        # Per dataset con 1 chunk atteso per domanda → Recall@k = 1 se hit else 0
        # (coincide con Hit Rate in questo caso)

        # Breakdown
        if cat:
            cat_stats[cat]["total"]  += 1
            cat_stats[cat]["hits"]   += int(trovato)
        if diff:
            diff_stats[diff]["total"] += 1
            diff_stats[diff]["hits"]  += int(trovato)

    hit_rate   = hits / n * 100
    precision  = prec_sum / n * 100
    mrr        = rr_sum / n
    recall     = hit_rate            # con 1 chunk atteso, Recall@k = Hit Rate@k

    tempo_medio = None
    if tempi_retrieval and len(tempi_retrieval) == n:
        tempo_medio = sum(tempi_retrieval) / n

    return {
        "n_domande":   n,
        "k":           k,
        "tolleranza":  tolleranza,
        "hit_rate_k":  hit_rate,
        "precision_k": precision,
        "recall_k":    recall,
        "mrr":         mrr,
        "per_categoria":  dict(cat_stats),
        "per_difficolta": dict(diff_stats),
        "tempo_medio_s":  tempo_medio,
    }


def stampa_report(db_name: str, metriche: Dict[str, Any]) -> None:
    """Stampa un report leggibile delle metriche di retrieval."""
    k   = metriche.get("k", 3)
    tol = metriche.get("tolleranza", 0)
    n   = metriche.get("n_domande", 0)
    sep = "─" * 55

    print(f"\n{'═' * 55}")
    print(f"📊  REPORT RETRIEVAL — {db_name}")
    print(f"{'═' * 55}")
    print(f"Domande valutate: {n}   |   Top-k={k}   |   Tolleranza=±{tol}")
    print(sep)
    print(f"Hit Rate@{k}       : {metriche['hit_rate_k']:6.2f}%")
    print(f"Precision@{k}      : {metriche['precision_k']:6.2f}%")
    print(f"Recall@{k}         : {metriche['recall_k']:6.2f}%")
    print(f"MRR              :  {metriche['mrr']:6.4f}")

    if metriche.get("tempo_medio_s") is not None:
        print(f"Tempo medio      :  {metriche['tempo_medio_s']:.3f} s/query")

    per_cat = metriche.get("per_categoria", {})
    if per_cat:
        print(sep)
        print("Breakdown per CATEGORIA:\n")
        for cat, s in sorted(per_cat.items()):
            hr = s["hits"] / s["total"] * 100 if s["total"] else 0
            print(f"  {cat:<30}  {s['hits']:>3}/{s['total']:<3}  ({hr:.1f}%)")

    per_diff = metriche.get("per_difficolta", {})
    if per_diff:
        print(sep)
        print("Breakdown per DIFFICOLTÀ:\n")
        ordine = ["bassa", "media", "alta", "low", "medium", "high"]
        chiavi = sorted(per_diff.keys(), key=lambda x: ordine.index(x) if x in ordine else 99)
        for diff, s in [(d, per_diff[d]) for d in chiavi]:
            hr = s["hits"] / s["total"] * 100 if s["total"] else 0
            print(f"  {diff:<10}  {s['hits']:>3}/{s['total']:<3}  ({hr:.1f}%)")

    print(f"{'═' * 55}\n")
