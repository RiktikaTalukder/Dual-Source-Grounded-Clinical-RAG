"""
confidence_scorer.py
Week 8 — Farhana (M2)
Computes a confidence score for a generated answer given
literature passages and patient case passages.

Formula:
    S(AL)      = cosine similarity: answer vs literature passages
    S(AP)      = cosine similarity: answer vs patient case passages
    A(L,P)     = NLI entailment score: literature vs patient evidence
    Confidence = (1/3)*S(AL) + (1/3)*S(AP) + (1/3)*A(L,P)
    If A(L,P) < 0.3: apply disagreement penalty → multiply by 0.7
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ── Load models once at module level ──────────────────────────────────────
print("Loading SentenceTransformer (ClinicalBERT)...")
_embedder = SentenceTransformer("medicalai/ClinicalBERT")

print("Loading NLI model (bart-large-mnli)...")
_nli = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1        # -1 = CPU. Change to 0 if you have a GPU.
)
print("Models loaded!")

# ── Helper: embed a piece of text ─────────────────────────────────────────
def _embed(text: str) -> np.ndarray:
    """Return a normalized embedding vector for a single text string."""
    vec = _embedder.encode([text], normalize_embeddings=True)
    return vec[0]   # shape: (embedding_dim,)

# ── Helper: mean embedding of a list of texts ─────────────────────────────
def _mean_embed(texts: list) -> np.ndarray:
    """
    Embed each passage in the list, then return their average vector.
    This represents the 'overall meaning' of a set of passages.
    """
    vecs = _embedder.encode(texts, normalize_embeddings=True)
    return np.mean(vecs, axis=0)   # shape: (embedding_dim,)

# ── S(AL): answer vs literature similarity ────────────────────────────────
def score_answer_literature(answer: str, literature_passages: list) -> float:
    """
    Cosine similarity between the answer embedding and the
    mean embedding of all literature passages.
    Range: 0.0 (no match) to 1.0 (perfect match)
    """
    answer_vec = _embed(answer)
    lit_vec    = _mean_embed(literature_passages)
    score      = float(np.dot(answer_vec, lit_vec))
    return round(score, 4)

# ── S(AP): answer vs patient cases similarity ─────────────────────────────
def score_answer_patient(answer: str, patient_passages: list) -> float:
    """
    Cosine similarity between the answer embedding and the
    mean embedding of all patient case passages.
    Range: 0.0 (no match) to 1.0 (perfect match)
    """
    answer_vec     = _embed(answer)
    patient_vec    = _mean_embed(patient_passages)
    score          = float(np.dot(answer_vec, patient_vec))
    return round(score, 4)

# ── A(L,P): literature vs patient agreement (NLI) ────────────────────────
def score_alignment(literature_passages: list, patient_passages: list) -> float:
    """
    Uses NLI to check whether literature ENTAILS (agrees with) patient evidence.
    
    We format the comparison as a single text and classify it as entailment,
    contradiction, or neutral using zero-shot classification.
    
    Range: 0.0 (completely disagree) to 1.0 (fully agree)
    """
    premise    = " ".join(literature_passages)[:800]
    hypothesis = " ".join(patient_passages)[:400]

    # Format as a single NLI-style statement for zero-shot classification
    # "Given the literature, the patient evidence is consistent" 
    nli_input = f"Premise: {premise} Hypothesis: {hypothesis}"

    result = _nli(
        nli_input,
        candidate_labels=["entailment", "contradiction", "neutral"],
    )

    labels = result["labels"]
    scores = result["scores"]
    entailment_score = scores[labels.index("entailment")]
    return round(float(entailment_score), 4)

# ── Main function: compute full confidence score ───────────────────────────
def compute_confidence(
    answer: str,
    literature_passages: list,
    patient_passages: list
) -> dict:
    """
    Compute the full confidence score for a generated answer.

    Parameters
    ----------
    answer               : str   — the answer your RAG system generated
    literature_passages  : list  — list of retrieved PMC passage strings
    patient_passages     : list  — list of retrieved MIMIC passage strings

    Returns
    -------
    dict with keys:
        s_al        — answer vs literature score
        s_ap        — answer vs patient score
        a_lp        — literature vs patient agreement score
        confidence  — final weighted confidence score
        penalty     — True if disagreement penalty was applied
    """
    s_al = score_answer_literature(answer, literature_passages)
    s_ap = score_answer_patient(answer, patient_passages)
    a_lp = score_alignment(literature_passages, patient_passages)

    # Equal-weight combination
    raw_confidence = (1/3) * s_al + (1/3) * s_ap + (1/3) * a_lp

    # Disagreement penalty
    penalty_applied = a_lp < 0.3
    if penalty_applied:
        raw_confidence *= 0.7

    return {
        "s_al":       s_al,
        "s_ap":       s_ap,
        "a_lp":       a_lp,
        "confidence": round(raw_confidence, 4),
        "penalty":    penalty_applied
    }

# ── Test on 10 dummy triples ───────────────────────────────────────────────
if __name__ == "__main__":

    test_cases = [
        {
            "answer": "Aspirin is recommended for patients with acute MI to reduce mortality.",
            "literature": [
                "Aspirin therapy reduces mortality in acute myocardial infarction patients.",
                "Antiplatelet agents including aspirin are standard treatment for MI."
            ],
            "patient": [
                "Patient was given aspirin 325mg on admission for chest pain.",
                "History of MI, currently on aspirin therapy."
            ]
        },
        {
            "answer": "Insulin is the first-line treatment for type 2 diabetes.",
            "literature": [
                "Metformin is the recommended first-line drug for type 2 diabetes.",
                "Lifestyle modification and metformin are initial treatments for T2DM."
            ],
            "patient": [
                "Patient diagnosed with type 2 diabetes, started on metformin.",
                "Blood glucose controlled with oral hypoglycaemics, not insulin."
            ]
        },
        {
            "answer": "Hypertension increases the risk of stroke and heart disease.",
            "literature": [
                "Elevated blood pressure is a major risk factor for cardiovascular events.",
                "Hypertension is strongly associated with stroke incidence."
            ],
            "patient": [
                "Patient has long-standing hypertension and presented with TIA.",
                "BP 160/95 on admission, history of hypertension."
            ]
        },
        {
            "answer": "Sepsis is treated with broad-spectrum antibiotics and fluid resuscitation.",
            "literature": [
                "Early antibiotic therapy and IV fluids are cornerstones of sepsis management.",
                "Broad-spectrum antibiotics should be given within one hour of sepsis diagnosis."
            ],
            "patient": [
                "Patient admitted with sepsis, started on piperacillin-tazobactam and IV fluids.",
                "Blood cultures drawn, empirical antibiotics commenced."
            ]
        },
        {
            "answer": "Beta-blockers are contraindicated in asthma patients.",
            "literature": [
                "Non-selective beta-blockers can cause bronchospasm in asthmatic patients.",
                "Beta-blockers should be used with caution or avoided in asthma."
            ],
            "patient": [
                "Patient with asthma — beta-blocker avoided, switched to calcium channel blocker.",
                "Known asthmatic, no beta-blocker prescribed."
            ]
        },
        {
            "answer": "Warfarin is used to prevent blood clots in atrial fibrillation.",
            "literature": [
                "Anticoagulation with warfarin reduces stroke risk in atrial fibrillation.",
                "Warfarin is effective for thromboembolic prevention in AF patients."
            ],
            "patient": [
                "Patient with AF, prescribed warfarin 5mg daily.",
                "INR monitored weekly, warfarin dose adjusted."
            ]
        },
        {
            "answer": "Kidney failure patients should avoid NSAIDs.",
            "literature": [
                "NSAIDs can worsen renal function and are contraindicated in CKD.",
                "Renal impairment is a known adverse effect of NSAID use."
            ],
            "patient": [
                "Patient with CKD stage 3, NSAIDs discontinued on admission.",
                "Renal function monitoring — ibuprofen stopped."
            ]
        },
        {
            "answer": "Oxygen therapy is given to all patients with pneumonia.",
            "literature": [
                "Supplemental oxygen is indicated when SpO2 drops below 94% in pneumonia.",
                "Not all pneumonia patients require oxygen; it depends on saturation levels."
            ],
            "patient": [
                "Patient SpO2 91%, started on 2L nasal cannula oxygen.",
                "Oxygen saturation improved to 96% after supplementation."
            ]
        },
        {
            "answer": "Statins reduce cholesterol and lower cardiovascular risk.",
            "literature": [
                "Statin therapy significantly reduces LDL cholesterol and cardiovascular events.",
                "Long-term statin use is associated with reduced MI and stroke risk."
            ],
            "patient": [
                "Patient prescribed atorvastatin 40mg for hypercholesterolaemia.",
                "LDL improved from 4.2 to 2.1 mmol/L after statin initiation."
            ]
        },
        {
            "answer": "Surgery is always required for appendicitis.",
            "literature": [
                "Uncomplicated appendicitis may be managed with antibiotics alone in select patients.",
                "Appendectomy remains standard but non-operative management is emerging."
            ],
            "patient": [
                "Patient with mild appendicitis treated successfully with IV antibiotics.",
                "No surgery performed — conservative management with antibiotics."
            ]
        },
    ]

    print("\n" + "="*60)
    print("CONFIDENCE SCORER — 10 DUMMY TEST CASES")
    print("="*60)

    for i, case in enumerate(test_cases, 1):
        result = compute_confidence(
            answer              = case["answer"],
            literature_passages = case["literature"],
            patient_passages    = case["patient"]
        )
        print(f"\nCase {i}: {case['answer'][:60]}...")
        print(f"  S(AL) = {result['s_al']}  |  "
              f"S(AP) = {result['s_ap']}  |  "
              f"A(L,P) = {result['a_lp']}")
        print(f"  Confidence = {result['confidence']}"
              f"  {'⚠ Penalty applied' if result['penalty'] else '✓ No penalty'}")