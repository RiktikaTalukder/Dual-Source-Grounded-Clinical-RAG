"""
baselines.py
Week 18 — Riktika (M1)

Implements all 4 baseline methods using the same LLM (flan-t5-base)
and same evaluation setup as generator.py (dual_source_rag).

Changes from Week 10 version:
- Constrained yes/no/maybe prompt (same as evidence_aligner.py Week 17)
- max_new_tokens=10 (matching generator.py Week 17)
- _extract_answer() added (matching generator.py Week 17)
- answer_raw, answer_extracted, extraction_method in all return dicts
- Patient-only baseline uses real BHC chunks (not metadata summary)
- lit_available / pat_available flags passed to compute_confidence()
- All debug prints removed

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
import re
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pmc_retriever import retrieve_literature
from patient_retriever import load_resources, retrieve
from confidence_scorer import compute_confidence
from config import extract_icd_hints, MODEL_REVISIONS, GENERATOR_MODEL

# ── Lazy-loaded resources (only loaded when first needed) ─────────────────
_tokenizer  = None
_llm        = None
_pat_model  = None
_pat_meta   = None
_pat_index  = None

def _ensure_resources():
    """Load all models the first time any baseline function is called."""
    global _tokenizer, _llm, _pat_model, _pat_meta, _pat_index
    if _llm is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            GENERATOR_MODEL,
            revision=MODEL_REVISIONS[GENERATOR_MODEL]
        )
        _llm = AutoModelForSeq2SeqLM.from_pretrained(
            GENERATOR_MODEL,
            revision=MODEL_REVISIONS[GENERATOR_MODEL]
        )
        _llm.eval()
    if _pat_model is None:
        _pat_model, _pat_meta, _pat_index = load_resources()


def _generate(prompt: str) -> str:
    """Run flan-t5-base on a prompt and return the raw answer string.
    max_new_tokens=10 matches generator.py — enough for yes/no/maybe."""
    _ensure_resources()
    inputs = _tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _llm.generate(
            **inputs,
            max_new_tokens=10,
            num_beams=4,
            early_stopping=True
        )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def _extract_answer(raw: str) -> tuple:
    """
    Three-step answer extraction. Matches generator.py exactly.
    Returns (answer_extracted, extraction_method).
    """
    # Step 1: direct — check first token
    first_token = raw.strip().split()[0] if raw.strip() else ""
    cleaned = re.sub(r'[^a-z]', '', first_token.lower())
    if cleaned in ("yes", "no", "maybe"):
        return cleaned, "direct"
    # Step 2: fallback regex — search first 20 tokens
    for token in raw.strip().split()[:20]:
        cleaned = re.sub(r'[^a-z]', '', token.lower())
        if cleaned in ("yes", "no", "maybe"):
            return cleaned, "fallback_regex"
    # Step 3: abstain
    return "abstain", "abstain"


# ── BASELINE 1: Literature-only RAG ───────────────────────────────────────
def baseline_literature_only(query: str, k: int = 3) -> dict:
    """
    Retrieves top-k PMC literature passages only.
    No patient case evidence used.
    """
    lit_results = retrieve_literature(query, k=k)
    if lit_results and isinstance(lit_results[0], dict):
        passages = [r.get("passage", str(r)) for r in lit_results]
    else:
        passages = [str(r) for r in lit_results]

    lit_block = "\n".join(
        f"[LIT {i+1}] {p.strip()}" for i, p in enumerate(passages)
    )
    prompt = (
        "[SYSTEM: You are a clinical AI assistant. "
        "Answer the question using the evidence below. "
        "Be concise and medically accurate. "
        "Your answer must be exactly one of three words: yes, no, or maybe. "
        "Do not include any explanation, qualification, or additional words.]\n\n"
        f"[LITERATURE EVIDENCE:\n{lit_block}]\n\n"
        "[PATIENT CASE EVIDENCE:\nNot available for this baseline.]\n\n"
        f"[QUESTION: {query}]\n\n"
        "[ANSWER (yes, no, or maybe):]"
    )

    answer_raw = _generate(prompt)
    answer_extracted, extraction_method = _extract_answer(answer_raw)

    scores = compute_confidence(
        answer_extracted, passages, [],
        lit_available=True, pat_available=False
    )

    return {
        "baseline":             "literature_only",
        "query":                query,
        "answer":               answer_extracted,
        "answer_raw":           answer_raw,
        "answer_extracted":     answer_extracted,
        "extraction_method":    extraction_method,
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
    Uses real BHC note chunks from the bhc_chunks field.
    No literature evidence used.
    """
    _ensure_resources()
    pat_results = retrieve(
        query_text=query,
        query_icd=extract_icd_hints(query),
        model=_pat_model,
        meta=_pat_meta,
        index=_pat_index,
        top_k=k
    )

    # Use real BHC chunks — fall back to metadata summary only if bhc absent
    bhc_chunks = []
    if pat_results and isinstance(pat_results[0], dict):
        for r in pat_results:
            chunks = r.get("bhc_chunks", [])
            if chunks:
                bhc_chunks.append(chunks[0])
            else:
                bhc_chunks.append(
                    f"Patient: age {r.get('age','?')}, "
                    f"gender {r.get('gender','?')}, "
                    f"ICD codes {r.get('icd_codes','?')}."
                )
    else:
        bhc_chunks = [str(r) for r in pat_results]

    pat_block = "\n".join(
        f"[PATIENT {i+1}] {s.strip()}" for i, s in enumerate(bhc_chunks)
    )
    prompt = (
        "[SYSTEM: You are a clinical AI assistant. "
        "Answer the question using the evidence below. "
        "Be concise and medically accurate. "
        "Your answer must be exactly one of three words: yes, no, or maybe. "
        "Do not include any explanation, qualification, or additional words.]\n\n"
        "[LITERATURE EVIDENCE:\nNot available for this baseline.]\n\n"
        f"[PATIENT CASE EVIDENCE:\n{pat_block}]\n\n"
        f"[QUESTION: {query}]\n\n"
        "[ANSWER (yes, no, or maybe):]"
    )

    answer_raw = _generate(prompt)
    answer_extracted, extraction_method = _extract_answer(answer_raw)

    scores = compute_confidence(
        answer_extracted, [], bhc_chunks,
        lit_available=False, pat_available=True
    )

    return {
        "baseline":             "patient_only",
        "query":                query,
        "answer":               answer_extracted,
        "answer_raw":           answer_raw,
        "answer_extracted":     answer_extracted,
        "extraction_method":    extraction_method,
        "confidence":           round(scores["confidence"], 4),
        "s_al":                 round(scores["s_al"], 4),
        "s_ap":                 round(scores["s_ap"], 4),
        "a_lp":                 round(scores["a_lp"], 4),
        "penalty":              scores["penalty"],
        "literature_passages":  [],
        "patient_summaries":    bhc_chunks
    }


# ── BASELINE 3: No-retrieval LLM ──────────────────────────────────────────
def baseline_no_retrieval(query: str) -> dict:
    """
    Sends the question directly to the LLM with no retrieved evidence.
    lit_available=False and pat_available=False passed to compute_confidence
    so no embedding calls are made (pre-embedding guard).
    """
    prompt = (
        "[SYSTEM: You are a clinical AI assistant. "
        "Answer the following clinical question. "
        "Your answer must be exactly one of three words: yes, no, or maybe. "
        "Do not include any explanation, qualification, or additional words.]\n\n"
        f"[QUESTION: {query}]\n\n"
        "[ANSWER (yes, no, or maybe):]"
    )

    answer_raw = _generate(prompt)
    answer_extracted, extraction_method = _extract_answer(answer_raw)

    # Both flags False — compute_confidence returns 0.5 immediately, no encoder called
    scores = compute_confidence(
        answer_extracted, [], [],
        lit_available=False, pat_available=False
    )

    return {
        "baseline":             "no_retrieval",
        "query":                query,
        "answer":               answer_extracted,
        "answer_raw":           answer_raw,
        "answer_extracted":     answer_extracted,
        "extraction_method":    extraction_method,
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
    Retrieves literature using fixed-size chunking (512 words, 10% overlap).
    Uses real chunk_fixed() from chunking_baselines.py.
    """
    from chunking_baselines import chunk_fixed

    lit_results = retrieve_literature(query, k=k)
    if lit_results and isinstance(lit_results[0], dict):
        raw_passages = [r.get("passage", str(r)) for r in lit_results]
    else:
        raw_passages = [str(r) for r in lit_results]

    passages = []
    for p in raw_passages:
        chunks = chunk_fixed(p, size=512, overlap=0.10)
        passages.append(chunks[0] if chunks else p)

    lit_block = "\n".join(
        f"[LIT {i+1}] {p.strip()}" for i, p in enumerate(passages)
    )
    prompt = (
        "[SYSTEM: You are a clinical AI assistant. "
        "Answer the question using the evidence below. "
        "Be concise and medically accurate. "
        "Your answer must be exactly one of three words: yes, no, or maybe. "
        "Do not include any explanation, qualification, or additional words.]\n\n"
        f"[LITERATURE EVIDENCE:\n{lit_block}]\n\n"
        "[PATIENT CASE EVIDENCE:\nNot available for this baseline.]\n\n"
        f"[QUESTION: {query}]\n\n"
        "[ANSWER (yes, no, or maybe):]"
    )

    answer_raw = _generate(prompt)
    answer_extracted, extraction_method = _extract_answer(answer_raw)

    scores = compute_confidence(
        answer_extracted, passages, [],
        lit_available=True, pat_available=False
    )

    return {
        "baseline":             "fixed_chunk_literature",
        "query":                query,
        "answer":               answer_extracted,
        "answer_raw":           answer_raw,
        "answer_extracted":     answer_extracted,
        "extraction_method":    extraction_method,
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
    test_query = "Does aspirin reduce the risk of cardiovascular events?"
    print(f"\nTesting all 4 baselines on: {test_query}\n")
    results = run_all_baselines(test_query)
    for name, r in results.items():
        print(f"\n{'='*60}")
        print(f"Baseline          : {name}")
        print(f"Answer raw        : {r['answer_raw']}")
        print(f"Answer extracted  : {r['answer_extracted']}")
        print(f"Extraction method : {r['extraction_method']}")
        print(f"Confidence        : {r['confidence']}  "
              f"S_AL={r['s_al']}  S_AP={r['s_ap']}  A_LP={r['a_lp']}  "
              f"Penalty={'YES' if r['penalty'] else 'no'}")
    print("\n[baselines.py] All 4 baselines working correctly!")
