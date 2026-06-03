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
from config import CONFIDENCE_WEIGHTS
import torch

# Default equal weights (same as confidence_scorer.py Week 8)
DEFAULT_WEIGHTS = CONFIDENCE_WEIGHTS   # (0.3, 0.3, 0.4) — best weights from Week 11 grid search

def _extract_icd_hints(query_text: str) -> set:
    """
    Extract rough ICD code hints from free-text query using keyword matching.
    This gives the patient retriever's Jaccard component something to work with
    instead of always receiving an empty set.
    """
    q = query_text.lower()
    hints = set()
    # Common clinical concept → representative ICD code mappings
    keyword_map = {
        "diabetes":        {"E11", "25001"},
        "heart failure":   {"I50", "42831"},
        "pneumonia":       {"J18", "4861"},
        "sepsis":          {"A41", "99591"},
        "hypertension":    {"I10", "4019"},
        "copd":            {"J44", "49121"},
        "stroke":          {"I63", "43491"},
        "myocardial":      {"I21", "41001"},
        "asthma":          {"J45", "49300"},
        "kidney":          {"N18", "5859"},
        "renal":           {"N18", "5859"},
        "cancer":          {"C80", "1999"},
        "obesity":         {"E66", "2780"},
        "depression":      {"F32", "29620"},
        "appendicitis":    {"K37", "5409"},
        "atrial":          {"I48", "42731"},
        "anticoagulation": {"Z79", "V5861"},
        "cholesterol":     {"E78", "2720"},
        "vitamin d":       {"E55", "2689"},
        "fracture":        {"M84", "8290"},
    }
    for keyword, codes in keyword_map.items():
        if keyword in q:
            hints.update(codes)
    return hints

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

        print("[Pipeline] Loading LLM (flan-t5-base)...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        self.llm.eval()
        print("[Pipeline] LLM ready.")

        print("[Pipeline] Loading patient retriever...")
        self.pat_model, self.pat_meta, self.pat_index = load_resources()
        print("[Pipeline] Patient retriever ready.")

    def _generate(self, prompt: str) -> str:
        """Run LLM and return answer string."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=200,
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

        penalty = a_lp < 0.3
        if penalty:
            raw *= 0.7

        return {
            "s_al":       round(s_al, 4),
            "s_ap":       round(s_ap, 4),
            "a_lp":       round(a_lp, 4),
            "confidence": round(raw, 4),
            "penalty":    penalty
        }

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
            query_icd=_extract_icd_hints(query),
            model=self.pat_model,
            meta=self.pat_meta,
            index=self.pat_index,
            top_k=k
        )
        if pat_results and isinstance(pat_results[0], dict):
            pat_summaries = [
                (f"Patient: age {r.get('age','?')}, gender {r.get('gender','?')}, "
                 f"admission type {r.get('admission_type','?')}, "
                 f"ICD codes {r.get('icd_codes','?')}. "
                 f"Similarity score: {round(r.get('rank_score', 0), 3)}.")
                for r in pat_results
            ]
        else:
            pat_summaries = [str(r) for r in pat_results]

        # Step 3: Build prompt using evidence_aligner (consistent with generator.py)
        prompt = align_evidence(query, lit_passages, pat_summaries)

        # Step 4: Generate answer
        answer = self._generate(prompt)

        # Step 5: Score with custom weights
        scores = self._compute_weighted_confidence(answer, lit_passages, pat_summaries)

        return {
            "query":               query,
            "answer":              answer,
            "confidence":          scores["confidence"],
            "s_al":                scores["s_al"],
            "s_ap":                scores["s_ap"],
            "a_lp":                scores["a_lp"],
            "penalty":             scores["penalty"],
            "weights":             self.weights,
            "literature_passages": lit_passages,
            "patient_summaries":   pat_summaries
        }


# Quick test
if __name__ == "__main__":
    pipe = Pipeline()
    result = pipe.run("Does aspirin reduce the risk of cardiovascular events?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print(f"S_AL={result['s_al']}  S_AP={result['s_ap']}  A_LP={result['a_lp']}")
    print(f"Penalty: {result['penalty']}")
