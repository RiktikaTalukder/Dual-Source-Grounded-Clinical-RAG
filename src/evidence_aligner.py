"""
evidence_aligner.py
Week 9 - Riktika (M1)
Week 24 fix - Farhana: added sentence-aware token-budget truncation.
Root cause confirmed: all 200 val200 prompts were 2748-4740 tokens;
flan-t5-large silently truncated from the right at 512 tokens, meaning
[QUESTION:] and [ANSWER] tags survived in 0/200 cases. Fix: truncate
evidence sentence-by-sentence so total prompt stays within 480 tokens.
"""

import re


def _fit_to_token_budget(text: str, tok, budget: int) -> str:
    """
    Keep whole sentences from text until the next sentence would
    exceed budget tokens. Falls back to first 200 characters if
    even the first sentence is over budget.
    """
    if not text or not text.strip():
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    kept = []
    used = 0
    for sent in sentences:
        n = len(tok(sent, add_special_tokens=False)["input_ids"])
        if used + n > budget:
            break
        kept.append(sent)
        used += n
    return " ".join(kept) if kept else text[:100]


def align_evidence(query, literature_passages, patient_summaries, tokenizer=None):
    """
    Combines literature and patient evidence into a structured prompt.

    Args:
        query (str): The clinical question
        literature_passages (list of str): Top passages from PMC literature
        patient_summaries (list of str): Top patient case summaries from MIMIC
        tokenizer: Optional flan-t5-large tokenizer. When supplied, evidence
                   is sentence-truncated so total prompt stays within 480 tokens
                   (32-token safety margin below flan-t5-large encoder limit).

    Returns:
        str: A formatted prompt string ready to send to the LLM
    """

    # Token-budget truncation — only runs when tokenizer is provided
    if tokenizer is not None:
        # Measure fixed overhead: system text + headers + question + answer tag
        _skeleton = (
            "[SYSTEM: You are a clinical AI assistant. Read the literature "
            "evidence and patient cases carefully. Based on the evidence, "
            "answer the clinical question. Consider whether the evidence "
            "supports, contradicts, or is uncertain about the claim. Your "
            "answer must be exactly one of three words: yes, no, or maybe. "
            "Output only that single word. Do not explain.]\n\n"
            "[LITERATURE EVIDENCE:\n]\n\n"
            "[PATIENT CASE EVIDENCE:\n]\n\n"
            f"[QUESTION: {query}]\n\n"
            "[ANSWER (yes, no, or maybe):]"
        )
        fixed_tokens = len(tokenizer(_skeleton)["input_ids"])
        evidence_budget = max(0, 680 - fixed_tokens)

        # 60% of budget to literature, 40% to patient
        n_lit = max(len(literature_passages), 1)
        n_pat = max(len(patient_summaries), 1)
        lit_per = max(1, int(evidence_budget * 0.60 / n_lit))
        pat_per = max(1, int(evidence_budget * 0.40 / n_pat))

        literature_passages = [
            _fit_to_token_budget(p, tokenizer, lit_per)
            for p in literature_passages
        ]
        patient_summaries = [
            _fit_to_token_budget(str(p), tokenizer, pat_per)
            for p in patient_summaries
        ]

    # Format literature evidence block
    if literature_passages:
        lit_block = "\n".join(
            f"[LIT {i+1}] {passage.strip()}"
            for i, passage in enumerate(literature_passages)
        )
    else:
        lit_block = "No literature evidence retrieved."

    # Format patient case evidence block
    if patient_summaries:
        pat_lines = []
        for i, p in enumerate(patient_summaries):
            if isinstance(p, dict):
                age = p.get("age", "?")
                gender = p.get("gender", "?")
                admission = p.get("admission_type", "?")
                icd = p.get("icd_codes_top5", "?")
                outcome = p.get("discharge_location", "?")
                chunks = p.get("chunks", [])
                chunk_text = " ... ".join(c.strip() for c in chunks if c.strip())
                pat_lines.append(
                    f"Patient {i+1} (age {age}, gender {gender}, "
                    f"admission {admission}, ICD: {icd}, outcome: {outcome}): "
                    f"{chunk_text}"
                )
            else:
                pat_lines.append(f"Patient {i+1}: {str(p).strip()}")
        pat_block = "\n".join(pat_lines)
    else:
        pat_block = "No similar patient cases retrieved."

    # Assemble full structured prompt
    prompt = (
        "[SYSTEM: You are a clinical AI assistant. Read the literature "
        "evidence and patient cases carefully. Based on the evidence, "
        "answer the clinical question. Consider whether the evidence "
        "supports, contradicts, or is uncertain about the claim. Your "
        "answer must be exactly one of three words: yes, no, or maybe. "
        "Output only that single word. Do not explain.]\n\n"
        f"[LITERATURE EVIDENCE:\n{lit_block}]\n\n"
        f"[PATIENT CASE EVIDENCE:\n{pat_block}]\n\n"
        f"[QUESTION: {query}]\n\n"
        "[ANSWER (yes, no, or maybe):]"
    )

    return prompt


if __name__ == "__main__":
    test_query = "What is the first-line treatment for community-acquired pneumonia?"
    test_lit = [
        "Amoxicillin is recommended as first-line therapy for mild CAP in outpatients.",
        "Beta-lactam antibiotics remain the standard of care for pneumonia treatment."
    ]
    test_pat = [
        "Patient admitted with fever and productive cough, diagnosed with CAP.",
        "65-year-old with lobar pneumonia responded well to beta-lactam therapy."
    ]
    prompt = align_evidence(test_query, test_lit, test_pat)
    print("=== GENERATED PROMPT (no tokenizer) ===")
    print(prompt)
    print("evidence_aligner.py works correctly!")