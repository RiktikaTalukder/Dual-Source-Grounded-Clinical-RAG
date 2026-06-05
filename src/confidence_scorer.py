"""
confidence_scorer.py
Week 15 — Farhana (M2)

Computes a confidence score for a generated answer given
literature passages and patient chunk strings.

Formula:
    S(AL)      = cosine similarity: answer vs literature passages
    S(AP)      = cosine similarity: answer vs patient chunk strings
    A(L,P)     = NLI entailment score: top-1 literature vs top-1 patient chunk
                 IMPORTANT: A(L,P) takes NO answer argument.
                 It measures agreement between the two sources only.
                 This makes it structurally independent of S(AL) and S(AP).
    Confidence = alpha*S(AL) + beta*S(AP) + gamma*A(L,P)
    where (alpha, beta, gamma) = CONFIDENCE_WEIGHTS from config.py

Availability flags (GAP 3 fix):
    lit_available  — False if no literature was retrieved
    pat_available  — False if no patient evidence was retrieved
    Flags are checked FIRST, before any encoder call.
    Missing source → its score and any joint score set to NEUTRAL_SCORE (0.5).
    Both missing   → return all 0.5 immediately, no encoder called.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
from config import (
    CONFIDENCE_WEIGHTS,
    PENALTY_THRESHOLD,
    PENALTY_MULTIPLIER,
    NEUTRAL_SCORE,
    MODEL_REVISIONS,
)

# ── Load models once at module level ─────────────────────────────────────────
print("Loading SentenceTransformer (S-PubMedBert-MS-MARCO)...")
_embedder = SentenceTransformer(
    "pritamdeka/S-PubMedBert-MS-MARCO",
    revision=MODEL_REVISIONS["pritamdeka/S-PubMedBert-MS-MARCO"],
)

print("Loading NLI model (bart-large-mnli)...")
_nli = hf_pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    revision=MODEL_REVISIONS["facebook/bart-large-mnli"],
    device=-1,   # -1 = CPU. Change to 0 if you have a GPU.
)
print("Models loaded.")


# ── Helper: embed a single text ───────────────────────────────────────────────
def _embed(text: str) -> np.ndarray:
    """Return a normalised embedding vector for a single text string."""
    vec = _embedder.encode([text], normalize_embeddings=True)
    return vec[0]


# ── Helper: mean embedding of a list of texts ─────────────────────────────────
def _mean_embed(texts: list) -> np.ndarray:
    """Embed each passage and return their average vector."""
    vecs = _embedder.encode(texts, normalize_embeddings=True)
    return np.mean(vecs, axis=0)


# ── S(AL): answer vs literature similarity ────────────────────────────────────
def score_answer_literature(answer: str, literature_passages: list) -> float:
    """
    Cosine similarity between the answer embedding and the
    mean embedding of all literature passages.
    Range: 0.0 to 1.0
    """
    answer_vec = _embed(answer)
    lit_vec    = _mean_embed(literature_passages)
    return round(float(np.dot(answer_vec, lit_vec)), 4)


# ── S(AP): answer vs patient chunks similarity ────────────────────────────────
def score_answer_patient(answer: str, patient_chunks: list) -> float:
    """
    Cosine similarity between the answer embedding and the
    mean embedding of all patient chunk strings.
    Range: 0.0 to 1.0
    """
    answer_vec  = _embed(answer)
    patient_vec = _mean_embed(patient_chunks)
    return round(float(np.dot(answer_vec, patient_vec)), 4)


# ── A(L,P): literature vs patient agreement — NLI, NO answer involved ─────────
def score_alignment(literature_passages: list, patient_chunks: list) -> float:
    """
    Computes NLI entailment between the top-1 literature passage (premise)
    and the top-1 patient chunk string (hypothesis).

    CRITICAL DESIGN NOTE:
    This function takes NO answer argument. The only inputs are the two
    retrieved source lists. This makes A(L,P) structurally independent of
    S(AL) and S(AP) — it measures source agreement, not answer grounding.

    Premise   : top-1 literature passage, truncated to 500 characters
    Hypothesis: top-1 patient chunk string, truncated to 350 characters

    Range: 0.0 (contradiction/neutral) to 1.0 (full entailment)
    """
    premise    = literature_passages[0][:500]
    hypothesis = patient_chunks[0][:350]

    result = _nli(
        premise,
        text_pair=hypothesis,
        top_k=None,
    )

    # bart-large-mnli label names: ENTAILMENT / CONTRADICTION / NEUTRAL
    label_map = {item["label"].upper(): item["score"] for item in result}
    entailment_score = label_map.get("ENTAILMENT", 0.0)
    return round(float(entailment_score), 4)


# ── Main function: compute full confidence score ──────────────────────────────
def compute_confidence(
    answer: str,
    literature_passages: list,
    patient_chunks: list,
    lit_available: bool = True,
    pat_available: bool = True,
) -> dict:
    """
    Compute the full confidence score for a generated answer.

    Parameters
    ----------
    answer              : str   — the answer the RAG system generated
    literature_passages : list  — retrieved PMC passage strings
    patient_chunks      : list  — retrieved MIMIC patient chunk strings
    lit_available       : bool  — False if no literature was retrieved
    pat_available       : bool  — False if no patient evidence was retrieved

    IMPORTANT: availability flags are checked FIRST, before any encoder call.
    If a source is unavailable, its score and any joint score are set to
    NEUTRAL_SCORE (0.5) without calling the embedding model or NLI model.

    Returns
    -------
    dict with keys: s_al, s_ap, a_lp, confidence, penalty
    """

    # ── GAP 3 FIX: check flags FIRST, before any encoder call ────────────────
    if not lit_available and not pat_available:
        # No retrieval at all — return all neutral, no encoder called
        return {
            "s_al":       NEUTRAL_SCORE,
            "s_ap":       NEUTRAL_SCORE,
            "a_lp":       NEUTRAL_SCORE,
            "confidence": NEUTRAL_SCORE,
            "penalty":    False,
        }

    if not lit_available:
        # Only patient evidence available
        s_al = NEUTRAL_SCORE
        a_lp = NEUTRAL_SCORE
        s_ap = score_answer_patient(answer, patient_chunks)
        alpha, beta, gamma = CONFIDENCE_WEIGHTS
        raw_confidence = alpha * s_al + beta * s_ap + gamma * a_lp
        penalty_applied = False   # cannot compute disagreement without both sources
        return {
            "s_al":       s_al,
            "s_ap":       round(s_ap, 4),
            "a_lp":       a_lp,
            "confidence": round(raw_confidence, 4),
            "penalty":    penalty_applied,
        }

    if not pat_available:
        # Only literature available
        s_ap = NEUTRAL_SCORE
        a_lp = NEUTRAL_SCORE
        s_al = score_answer_literature(answer, literature_passages)
        alpha, beta, gamma = CONFIDENCE_WEIGHTS
        raw_confidence = alpha * s_al + beta * s_ap + gamma * a_lp
        penalty_applied = False   # cannot compute disagreement without both sources
        return {
            "s_al":       round(s_al, 4),
            "s_ap":       s_ap,
            "a_lp":       a_lp,
            "confidence": round(raw_confidence, 4),
            "penalty":    penalty_applied,
        }

    # ── Both sources available: compute all three components ──────────────────
    s_al = score_answer_literature(answer, literature_passages)
    s_ap = score_answer_patient(answer, patient_chunks)
    a_lp = score_alignment(literature_passages, patient_chunks)

    alpha, beta, gamma = CONFIDENCE_WEIGHTS
    raw_confidence = alpha * s_al + beta * s_ap + gamma * a_lp

    penalty_applied = a_lp < PENALTY_THRESHOLD
    if penalty_applied:
        raw_confidence *= PENALTY_MULTIPLIER

    return {
        "s_al":       s_al,
        "s_ap":       s_ap,
        "a_lp":       a_lp,
        "confidence": round(raw_confidence, 4),
        "penalty":    penalty_applied,
    }


# ── Test: 4 scenarios required by workplan ────────────────────────────────────
if __name__ == "__main__":

    lit = [
        "Aspirin therapy reduces mortality in acute myocardial infarction.",
        "Antiplatelet agents are standard treatment for MI."
    ]
    pat = [
        "Patient given aspirin 325mg on admission for chest pain.",
        "History of MI, currently on aspirin therapy."
    ]
    answer = "Aspirin reduces cardiovascular mortality."

    print("\n" + "="*60)
    print("SCENARIO 1: Both sources available")
    print("="*60)
    r = compute_confidence(answer, lit, pat, lit_available=True, pat_available=True)
    print(f"  S(AL)={r['s_al']}  S(AP)={r['s_ap']}  A(L,P)={r['a_lp']}")
    print(f"  Confidence={r['confidence']}  Penalty={r['penalty']}")
    assert 0.0 <= r["confidence"] <= 1.0, "Score out of range"
    print("  ✅ PASS")

    print("\n" + "="*60)
    print("SCENARIO 2: Literature only (pat_available=False)")
    print("="*60)
    r = compute_confidence(answer, lit, [], lit_available=True, pat_available=False)
    print(f"  S(AL)={r['s_al']}  S(AP)={r['s_ap']}  A(L,P)={r['a_lp']}")
    print(f"  Confidence={r['confidence']}  Penalty={r['penalty']}")
    assert r["s_ap"] == 0.5, "S(AP) should be neutral 0.5"
    assert r["a_lp"] == 0.5, "A(L,P) should be neutral 0.5"
    print("  ✅ PASS — no encoder called for patient, no crash")

    print("\n" + "="*60)
    print("SCENARIO 3: Patient only (lit_available=False)")
    print("="*60)
    r = compute_confidence(answer, [], pat, lit_available=False, pat_available=True)
    print(f"  S(AL)={r['s_al']}  S(AP)={r['s_ap']}  A(L,P)={r['a_lp']}")
    print(f"  Confidence={r['confidence']}  Penalty={r['penalty']}")
    assert r["s_al"] == 0.5, "S(AL) should be neutral 0.5"
    assert r["a_lp"] == 0.5, "A(L,P) should be neutral 0.5"
    print("  ✅ PASS — no encoder called for literature, no crash")

    print("\n" + "="*60)
    print("SCENARIO 4: No retrieval (both False)")
    print("="*60)
    r = compute_confidence(answer, [], [], lit_available=False, pat_available=False)
    print(f"  S(AL)={r['s_al']}  S(AP)={r['s_ap']}  A(L,P)={r['a_lp']}")
    print(f"  Confidence={r['confidence']}  Penalty={r['penalty']}")
    assert r["s_al"] == 0.5
    assert r["s_ap"] == 0.5
    assert r["a_lp"] == 0.5
    assert r["confidence"] == 0.5
    print("  ✅ PASS — no encoder called at all, no crash")

    print("\n" + "="*60)
    print("ALL 4 SCENARIOS PASSED")
    print("="*60)