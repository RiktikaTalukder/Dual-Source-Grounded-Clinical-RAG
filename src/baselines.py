"""
baselines.py
Week 10 — Farhana (M2)

Implements all 4 baseline methods using the same LLM (flan-t5-base)
and same evaluation setup as generator.py (dual_source_rag).

Baseline 1: Literature-only RAG
Baseline 2: Patient-only RAG
Baseline 3: No-retrieval LLM (no context, just the question)
Baseline 4: Fixed-chunk Literature-only RAG

Usage:
    from baselines import run_all_baselines
    results = run_all_baselines(query)
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pmc_retriever import retrieve_literature
from patient_retriever import load_resources, retrieve
from confidence_scorer import compute_confidence

# ── Load shared LLM once (same model as generator.py) ─────────────────────
print("[baselines] Loading flan-t5-base...")
_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
_llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
_llm.eval()
print("[baselines] LLM ready.")

# ── Load patient retriever resources ──────────────────────────────────────
print("[baselines] Loading patient retriever resources...")
_pat_model, _pat_meta, _pat_index = load_resources()
print("[baselines] Patient resources ready.")


def _generate(prompt: str) -> str:
    """Run flan-t5-base on a prompt and return the answer string.
    Identical setup to generator.py so results are comparable."""
    inputs = _tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = _llm.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True
        )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ── BASELINE 1: Literature-only RAG ───────────────────────────────────────
def baseline_literature_only(query: str, k: int = 3) -> dict:
    """
    Retrieves top-k PMC literature passages only.
    No patient case evidence used.
    Same LLM and confidence scorer as dual_source_rag.
    """
    lit_results = retrieve_literature(query, k=k)
    if lit_results and isinstance(lit_results[0], dict):
        passages = [r.get("passage", str(r)) for r in lit_results]
    else:
        passages = [str(r) for r in lit_results]

    # Build prompt with literature only
    lit_block = "\n".join(
        f"[LIT {i+1}] {p.strip()}" for i, p in enumerate(passages)
    )
    prompt = (
        "[SYSTEM: You are a clinical AI assistant. "
        "Answer the question using the evidence below. "
        "Be concise and medically accurate.]\n\n"
        f"[LITERATURE EVIDENCE:\n{lit_block}]\n\n"
        "[PATIENT CASE EVIDENCE:\nNot available for this baseline.]\n\n"
        f"[QUESTION: {query}]\n\n[ANSWER:]"
    )

    answer = _generate(prompt)

    # Score confidence — patient passages set to a placeholder
    placeholder_patient = ["No patient evidence used in this baseline."]
    scores = compute_confidence(answer, passages, placeholder_patient)

    return {
        "baseline":             "literature_only",
        "query":                query,
        "answer":               answer,
        "confidence":           round(scores["confidence"], 4),
        "s_al":                 round(scores["s_al"], 4),
        "s_ap":                 round(scores["s_ap"], 4),
        "a_lp":                 round(scores["a_lp"], 4),
        "penalty":              scores["penalty"],
        "literature_passages":  passages,
        "patient_summaries":    []
    }


# ── BASELINE 2: Patient-only RAG ──────────────────────────────────────────
def baseline_patient_only(query: str, k: int = 3) -> dict:
    """
    Retrieves top-k MIMIC patient cases only.
    No literature evidence used.
    Same LLM and confidence scorer as dual_source_rag.
    """
    pat_results = retrieve(
        query_text=query,
        query_icd=set(),
        model=_pat_model,
        meta=_pat_meta,
        index=_pat_index,
        top_k=k
    )
    if pat_results and isinstance(pat_results[0], dict):
        summaries = [
            r.get("summary", r.get("text", r.get("note_text", str(r))))
            for r in pat_results
        ]
    else:
        summaries = [str(r) for r in pat_results]

    # Build prompt with patient cases only
    pat_block = "\n".join(
        f"[PATIENT {i+1}] {s.strip()}" for i, s in enumerate(summaries)
    )
    prompt = (
        "[SYSTEM: You are a clinical AI assistant. "
        "Answer the question using the evidence below. "
        "Be concise and medically accurate.]\n\n"
        "[LITERATURE EVIDENCE:\nNot available for this baseline.]\n\n"
        f"[PATIENT CASE EVIDENCE:\n{pat_block}]\n\n"
        f"[QUESTION: {query}]\n\n[ANSWER:]"
    )

    answer = _generate(prompt)

    # Score confidence — literature passages set to a placeholder
    placeholder_lit = ["No literature evidence used in this baseline."]
    scores = compute_confidence(answer, placeholder_lit, summaries)

    return {
        "baseline":             "patient_only",
        "query":                query,
        "answer":               answer,
        "confidence":           round(scores["confidence"], 4),
        "s_al":                 round(scores["s_al"], 4),
        "s_ap":                 round(scores["s_ap"], 4),
        "a_lp":                 round(scores["a_lp"], 4),
        "penalty":              scores["penalty"],
        "literature_passages":  [],
        "patient_summaries":    summaries
    }


# ── BASELINE 3: No-retrieval LLM ──────────────────────────────────────────
def baseline_no_retrieval(query: str) -> dict:
    """
    Sends the question directly to the LLM with no retrieved evidence.
    Tests what the LLM knows from its own training alone.
    Same LLM as dual_source_rag.
    """
    prompt = (
        "[SYSTEM: You are a clinical AI assistant. "
        "Answer the following clinical question concisely and accurately.]\n\n"
        f"[QUESTION: {query}]\n\n[ANSWER:]"
    )

    answer = _generate(prompt)

    # No retrieved evidence — use placeholders for scoring
    placeholder = ["No evidence retrieved for this baseline."]
    scores = compute_confidence(answer, placeholder, placeholder)

    return {
        "baseline":             "no_retrieval",
        "query":                query,
        "answer":               answer,
        "confidence":           round(scores["confidence"], 4),
        "s_al":                 round(scores["s_al"], 4),
        "s_ap":                 round(scores["s_ap"], 4),
        "a_lp":                 round(scores["a_lp"], 4),
        "penalty":              scores["penalty"],
        "literature_passages":  [],
        "patient_summaries":    []
    }


# ── BASELINE 4: Fixed-chunk Literature-only RAG ───────────────────────────
def baseline_fixed_chunk_literature(query: str, k: int = 3) -> dict:
    """
    Retrieves literature using fixed-size chunking (512 tokens, 10% overlap)
    instead of the dynamic retrieval-guided chunking used in dual_source_rag.
    This isolates the effect of chunking strategy on answer quality.

    Note: pmc_retriever already uses sentence embeddings over full passages.
    Here we simulate fixed chunking by truncating each passage to 512 characters
    (roughly 100 words) before passing to the LLM — matching Baseline A from
    chunking_baselines.py in spirit.
    """
    lit_results = retrieve_literature(query, k=k)
    if lit_results and isinstance(lit_results[0], dict):
        raw_passages = [r.get("passage", str(r)) for r in lit_results]
    else:
        raw_passages = [str(r) for r in lit_results]

    # Simulate fixed-size chunking: truncate each passage to ~512 chars
    FIXED_CHUNK_CHARS = 512
    passages = [p[:FIXED_CHUNK_CHARS] for p in raw_passages]

    lit_block = "\n".join(
        f"[LIT {i+1}] {p.strip()}" for i, p in enumerate(passages)
    )
    prompt = (
        "[SYSTEM: You are a clinical AI assistant. "
        "Answer the question using the evidence below. "
        "Be concise and medically accurate.]\n\n"
        f"[LITERATURE EVIDENCE:\n{lit_block}]\n\n"
        "[PATIENT CASE EVIDENCE:\nNot available for this baseline.]\n\n"
        f"[QUESTION: {query}]\n\n[ANSWER:]"
    )

    answer = _generate(prompt)

    placeholder_patient = ["No patient evidence used in this baseline."]
    scores = compute_confidence(answer, passages, placeholder_patient)

    return {
        "baseline":             "fixed_chunk_literature",
        "query":                query,
        "answer":               answer,
        "confidence":           round(scores["confidence"], 4),
        "s_al":                 round(scores["s_al"], 4),
        "s_ap":                 round(scores["s_ap"], 4),
        "a_lp":                 round(scores["a_lp"], 4),
        "penalty":              scores["penalty"],
        "literature_passages":  passages,
        "patient_summaries":    []
    }


# ── Run all 4 baselines on a single query ─────────────────────────────────
def run_all_baselines(query: str) -> dict:
    """
    Runs all 4 baselines on one query.
    Returns a dict with keys: literature_only, patient_only,
                               no_retrieval, fixed_chunk_literature
    """
    return {
        "literature_only":        baseline_literature_only(query),
        "patient_only":           baseline_patient_only(query),
        "no_retrieval":           baseline_no_retrieval(query),
        "fixed_chunk_literature": baseline_fixed_chunk_literature(query)
    }


# ── Quick test when run directly ──────────────────────────────────────────
if __name__ == "__main__":
    import json
    test_query = "Does aspirin reduce the risk of cardiovascular events?"
    print(f"\nTesting all 4 baselines on: {test_query}\n")
    results = run_all_baselines(test_query)
    for name, r in results.items():
        print(f"\n{'='*60}")
        print(f"Baseline : {name}")
        print(f"Answer   : {r['answer'][:120]}...")
        print(f"Confidence: {r['confidence']}  "
              f"S_AL={r['s_al']}  S_AP={r['s_ap']}  A_LP={r['a_lp']}  "
              f"Penalty={'YES' if r['penalty'] else 'no'}")
    print("\n[baselines.py] All 4 baselines working correctly!")
