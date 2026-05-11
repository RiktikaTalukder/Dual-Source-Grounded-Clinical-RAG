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
import torch

# Default equal weights (same as confidence_scorer.py Week 8)
DEFAULT_WEIGHTS = (1/3, 1/3, 1/3)


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
            query_icd=set(),
            model=self.pat_model,
            meta=self.pat_meta,
            index=self.pat_index,
            top_k=k
        )
        if pat_results and isinstance(pat_results[0], dict):
            pat_summaries = [
                r.get("summary", r.get("text", r.get("note_text", str(r))))
                for r in pat_results
            ]
        else:
            pat_summaries = [str(r) for r in pat_results]

        # Step 3: Build prompt
        lit_block = "\n".join(
            f"[LIT {i+1}] {p.strip()}" for i, p in enumerate(lit_passages)
        )
        pat_block = "\n".join(
            f"[PATIENT {i+1}] {s.strip()}" for i, s in enumerate(pat_summaries)
        )
        prompt = (
            "[SYSTEM: You are a clinical AI assistant. "
            "Answer the question using the evidence below. "
            "Be concise and medically accurate.]\n\n"
            f"[LITERATURE EVIDENCE:\n{lit_block}]\n\n"
            f"[PATIENT CASE EVIDENCE:\n{pat_block}]\n\n"
            f"[QUESTION: {query}]\n\n[ANSWER:]"
        )

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
