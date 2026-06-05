"""
generator.py
Week 9 - Riktika (M1)

The main dual-source RAG pipeline.
dual_source_rag(query) -> {answer, confidence, sources}

Calls: pmc_retriever -> patient_retriever -> evidence_aligner -> LLM -> confidence_scorer
LLM: flan-t5-base via AutoModelForSeq2SeqLM (compatible with transformers 5.x)
"""

import sys
import os
import json
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pmc_retriever import retrieve_literature
from patient_retriever import load_resources, retrieve
from evidence_aligner import align_evidence
from confidence_scorer import compute_confidence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Load patient retriever resources once at startup ───────────────────────
print("[generator] Loading patient retriever resources...")
_pat_model, _pat_meta, _pat_index = load_resources()
print("[generator] Patient resources ready.")

# ── Load flan-t5-base once at startup ─────────────────────────────────────
print("[generator] Loading flan-t5-base LLM (first run downloads ~1GB)...")
_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
_llm_model  = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
_llm_model.eval()
print("[generator] LLM ready.")


def _generate_answer(prompt: str, max_new_tokens: int = 10) -> str:
    """Run flan-t5-base on the prompt and return the answer string."""
    inputs = _tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = _llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True
        )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def _extract_answer(raw: str) -> tuple:
    """
    Three-step answer extraction.
    Returns (answer_extracted, extraction_method)
    """
    import re
    # Step 1: direct — first token
    first_token = raw.strip().split()[0] if raw.strip() else ""
    cleaned = re.sub(r'[^a-z]', '', first_token.lower())
    if cleaned in ("yes", "no", "maybe"):
        return cleaned, "direct"
    # Step 2: fallback regex — search first 20 tokens
    tokens = raw.strip().split()[:20]
    for token in tokens:
        cleaned = re.sub(r'[^a-z]', '', token.lower())
        if cleaned in ("yes", "no", "maybe"):
            return cleaned, "fallback_regex"
    # Step 3: abstain
    return "abstain", "abstain"
def dual_source_rag(query: str, top_k_lit: int = 3, top_k_pat: int = 3) -> dict:
    """
    Full dual-source RAG pipeline.

    Args:
        query      : Clinical question string
        top_k_lit  : Number of literature passages to retrieve
        top_k_pat  : Number of patient cases to retrieve

    Returns dict with keys:
        query, answer, confidence, s_al, s_ap, a_lp, penalty,
        literature_passages, patient_summaries, runtime_seconds
    """
    t_start = time.time()

    # ── Step 1: Retrieve literature passages from PMC ──────────────────────
    print(f"  [1/5] Retrieving literature...")
    lit_results = retrieve_literature(query, k=top_k_lit)
    if lit_results and isinstance(lit_results[0], dict):
        literature_passages = [r.get("text", r.get("chunk", str(r))) for r in lit_results]
    else:
        literature_passages = [str(r) for r in lit_results]

    # ── Step 2: Retrieve similar patient cases from MIMIC ──────────────────
    print(f"  [2/5] Retrieving patient cases...")
    pat_results = retrieve(
        query_text=query,
        query_icd=set(),
        model=_pat_model,
        meta=_pat_meta,
        index=_pat_index,
        top_k=top_k_pat
    )
    if pat_results and isinstance(pat_results[0], dict):
        patient_summaries = [
            r.get("summary", r.get("text", r.get("note_text", str(r))))
            for r in pat_results
        ]
    else:
        patient_summaries = [str(r) for r in pat_results]

    # ── Step 3: Align evidence into structured prompt ──────────────────────
    print(f"  [3/5] Aligning evidence into prompt...")
    prompt = align_evidence(query, literature_passages, patient_summaries)

    # ── Step 4: Generate answer with flan-t5-base ──────────────────────────
    print(f"  [4/5] Generating answer...")
    answer_raw = _generate_answer(prompt)
    answer_extracted, extraction_method = _extract_answer(answer_raw)

    # ── Step 5: Score confidence ───────────────────────────────────────────
    print(f"  [5/5] Scoring confidence...")
    scores = compute_confidence(answer_raw, literature_passages, patient_summaries)

    runtime = round(time.time() - t_start, 2)

    result = {
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
        "literature_passages":  literature_passages,
        "patient_summaries":    patient_summaries,
        "runtime_seconds":      runtime
    }

    print(f"  ✓ Done in {runtime}s | confidence={result['confidence']}")
    return result


# ── Quick single-query test ────────────────────────────────────────────────
if __name__ == "__main__":
    test_query = "What is the first-line treatment for community-acquired pneumonia?"
    print("\n=== DUAL SOURCE RAG — SINGLE QUERY TEST ===")
    print(f"Query: {test_query}\n")
    output = dual_source_rag(test_query)
    print("\n=== RESULT ===")
    print(f"Answer     : {output['answer']}")
    print(f"Confidence : {output['confidence']}")
    print(f"S(A,L)     : {output['s_al']}")
    print(f"S(A,P)     : {output['s_ap']}")
    print(f"A(L,P)     : {output['a_lp']}")
    print(f"Penalty    : {output['penalty']}")
    print(f"Runtime    : {output['runtime_seconds']}s")
    print("\n=== generator.py works correctly! ===")
