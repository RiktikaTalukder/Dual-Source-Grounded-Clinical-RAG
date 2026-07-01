"""
pipeline.py
Week 11 — Riktika (M1)

Wraps the full dual-source RAG system into a Pipeline class.
Also supports running with custom confidence weights for grid search.

Usage:
    from pipeline import Pipeline
    result = Pipeline().run("Does aspirin reduce cardiovascular risk?")
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pmc_retriever import retrieve_literature
from patient_retriever import load_resources, retrieve
from confidence_scorer import compute_confidence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evidence_aligner import align_evidence
from config import CONFIDENCE_WEIGHTS, PENALTY_THRESHOLD, PENALTY_MULTIPLIER, extract_icd_hints, MODEL_REVISIONS, GENERATOR_MODEL, DEVICE
import torch
import pandas as pd
import glob
from chunking_baselines import chunk_dynamic
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default equal weights (same as confidence_scorer.py Week 8)
DEFAULT_WEIGHTS = CONFIDENCE_WEIGHTS   # (0.5, 0.25, 0.25) -- best weights from Week 20 100-combo grid search

import re as _re_module

def _extract_answer(raw: str) -> tuple:
    """Three-step answer extraction. Matches baselines.py exactly."""
    first_token = raw.strip().split()[0] if raw.strip() else ""
    cleaned = _re_module.sub(r'[^a-z]', '', first_token.lower())
    if cleaned in ("yes", "no", "maybe"):
        return cleaned, "direct"
    for token in raw.strip().split()[:20]:
        cleaned = _re_module.sub(r'[^a-z]', '', token.lower())
        if cleaned in ("yes", "no", "maybe"):
            return cleaned, "fallback_regex"
    return "abstain", "abstain"
class Pipeline:
    """
    Full dual-source RAG pipeline.
    Loads all models once, then you can call .run() many times.
    """

    def __init__(self, weights=None):
        """
        weights: tuple of (alpha, beta, gamma) for (S_AL, S_AP, A_LP)
                 If None, uses equal weights (1/3, 1/3, 1/3)
        """
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS
        # Load patient metadata for discharge_location lookup
        meta_path = os.path.join(BASE, "data", "mimic", "processed", "patient_metadata_stratified.csv")
        if os.path.exists(meta_path):
            self.pat_meta_full = pd.read_csv(meta_path)
        else:
            self.pat_meta_full = None
        self.notes_dir = os.path.join(BASE, "data", "mimic", "mimic_sample")

        print(f"[Pipeline] Loading LLM ({GENERATOR_MODEL})...")
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL, revision=MODEL_REVISIONS[GENERATOR_MODEL])
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL, revision=MODEL_REVISIONS[GENERATOR_MODEL]).to(DEVICE)
        self.llm.eval()
        print(f"[Pipeline] LLM ready on {DEVICE}.")

        print("[Pipeline] Loading patient retriever...")
        self.pat_model, self.pat_meta, self.pat_index = load_resources()
        print("[Pipeline] Patient retriever ready.")

    def _generate(self, prompt: str) -> str:
        """Run LLM and return answer string."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=800
        ).to(DEVICE)
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=10,
                num_beams=4,
                early_stopping=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def _compute_weighted_confidence(self, answer, lit_passages, pat_summaries):
        """
        Compute confidence using custom weights instead of always 1/3 each.
        This is the key change that lets grid search tune the weights.
        """
        from confidence_scorer import (
            score_answer_literature,
            score_answer_patient,
            score_alignment
        )

        s_al = score_answer_literature(answer, lit_passages)
        s_ap = score_answer_patient(answer, pat_summaries)
        a_lp = score_alignment(lit_passages, pat_summaries)

        alpha, beta, gamma = self.weights
        raw = alpha * s_al + beta * s_ap + gamma * a_lp

        penalty = a_lp < PENALTY_THRESHOLD
        if penalty:
            raw *= PENALTY_MULTIPLIER

        return {
            "s_al":       round(s_al, 4),
            "s_ap":       round(s_ap, 4),
            "a_lp":       round(a_lp, 4),
            "confidence": round(raw, 4),
            "penalty":    penalty
        }


    def _get_patient_chunks(self, subject_id, query):
        """Read patient note from disk, return top-2 chunks via chunk_dynamic."""
        pattern = os.path.join(self.notes_dir, f"note_{subject_id}-*.txt")
        matches = glob.glob(pattern)
        if matches:
            with open(matches[0], "r", encoding="utf-8") as f:
                note_text = f.read()
            chunks = chunk_dynamic(note_text, query, top_n=2)
            return [c for c in chunks if c.strip()]
        # Fallback to bhc field
        if self.pat_meta_full is not None:
            row = self.pat_meta_full[
                self.pat_meta_full["subject_id"] == int(subject_id)
            ]
            if not row.empty:
                bhc = str(row.iloc[0].get("bhc", ""))
                if bhc.strip():
                    return [bhc[:800]]
        return []

    def run(self, query: str, k: int = 3) -> dict:
        """
        Run the full dual-source RAG pipeline on one query.

        Parameters
        ----------
        query : str  — the clinical question
        k     : int  — how many passages/cases to retrieve

        Returns
        -------
        dict with answer, confidence scores, and retrieved evidence
        """
        # Step 1: Retrieve literature
        lit_results = retrieve_literature(query, k=k)
        if lit_results and isinstance(lit_results[0], dict):
            lit_passages = [r.get("passage", str(r)) for r in lit_results]
        else:
            lit_passages = [str(r) for r in lit_results]

        # Step 2: Retrieve patient cases
        pat_results = retrieve(
            query_text=query,
            query_icd=extract_icd_hints(query),
            model=self.pat_model,
            meta=self.pat_meta,
            index=self.pat_index,
            top_k=k
        )
        if pat_results and isinstance(pat_results[0], dict):
            pat_summaries = []
            for r in pat_results:
                subject_id = r.get("subject_id", "")
                chunks = self._get_patient_chunks(subject_id, query) if subject_id else []
                age        = r.get("age", "?")
                gender     = r.get("gender", "?")
                admission  = r.get("admission_type", "?")
                icd        = r.get("icd_codes", "?")
                outcome    = r.get("discharge_location", "?")
                if chunks:
                    chunk_text = " ... ".join(c.strip() for c in chunks if c.strip())
                else:
                    # fallback to BHC field if note file not found
                    chunk_text = r.get("bhc_preview", "No clinical notes available.")
                    if not chunk_text or str(chunk_text).strip() in ("", "nan"):
                        chunk_text = "No clinical notes available."
                pat_summaries.append(
                    f"Patient (age {age}, {gender}, {admission}, "
                    f"ICD: {icd}, outcome: {outcome}): {str(chunk_text)[:600]}"
                )
        else:
            pat_summaries = [str(r) for r in pat_results]

        # Step 3: Build prompt using evidence_aligner (consistent with generator.py)
        prompt = align_evidence(query, lit_passages, pat_summaries, tokenizer=self.tokenizer)

        # Step 4: Generate answer and extract yes/no/maybe
        answer_raw = self._generate(prompt)
        answer_extracted, extraction_method = _extract_answer(answer_raw)

        # Step 5: Score with custom weights
        pat_texts = []
        for p in pat_summaries:
            if isinstance(p, dict):
                chunks = p.get('chunks', [])
                if chunks:
                    pat_texts.append(' '.join(str(c) for c in chunks))
                else:
                    pat_texts.append(str(p.get('icd_codes_top5', '')))
            else:
                pat_texts.append(str(p))
        scores = self._compute_weighted_confidence(answer_raw, lit_passages, pat_texts)

        return {
            "query":               query,
            "answer":              answer_extracted,
            "answer_raw":          answer_raw,
            "answer_extracted":    answer_extracted,
            "extraction_method":   extraction_method,
            "confidence":          scores["confidence"],
            "s_al":                scores["s_al"],
            "s_ap":                scores["s_ap"],
            "a_lp":                scores["a_lp"],
            "penalty":             scores["penalty"],
            "weights":             self.weights,
            "literature_passages": lit_passages,
            "patient_summaries":   pat_summaries,
        }


# Quick test
if __name__ == "__main__":
    pipe = Pipeline()
    result = pipe.run("Does aspirin reduce the risk of cardiovascular events?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print(f"S_AL={result['s_al']}  S_AP={result['s_ap']}  A_LP={result['a_lp']}")
    print(f"Penalty: {result['penalty']}")
