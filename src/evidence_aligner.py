"""
evidence_aligner.py
Week 9 - Riktika (M1)

Takes retrieved literature passages and patient summaries,
formats them into a structured prompt string for the LLM.
"""

def align_evidence(query, literature_passages, patient_summaries):
    """
    Combines literature and patient evidence into a structured prompt.

    Args:
        query (str): The clinical question
        literature_passages (list of str): Top passages from PMC literature
        patient_summaries (list of str): Top patient case summaries from MIMIC

    Returns:
        str: A formatted prompt string ready to send to the LLM
    """

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

    # Assemble the full structured prompt
    prompt = f"""[SYSTEM: You are a clinical AI assistant. Read the literature evidence and patient cases carefully. Based on the evidence, answer the clinical question. Consider whether the evidence supports, contradicts, or is uncertain about the claim. Your answer must be exactly one of three words: yes, no, or maybe. Output only that single word. Do not explain.]

[LITERATURE EVIDENCE:
{lit_block}]

[PATIENT CASE EVIDENCE:
{pat_block}]

[QUESTION: {query}]

[ANSWER (yes, no, or maybe):]"""

    return prompt


# Quick test when running this file directly
if __name__ == "__main__":
    test_query = "What is the first-line treatment for community-acquired pneumonia?"
    test_lit = [
        "Amoxicillin is recommended as first-line therapy for mild CAP in outpatients.",
        "Beta-lactam antibiotics remain the standard of care for pneumonia treatment."
    ]
    test_pat = [
        "Patient admitted with fever and productive cough, diagnosed with CAP, treated with amoxicillin.",
        "65-year-old with lobar pneumonia responded well to beta-lactam therapy."
    ]

    prompt = align_evidence(test_query, test_lit, test_pat)
    print("=== GENERATED PROMPT ===")
    print(prompt)
    print("========================")
    print("evidence_aligner.py works correctly!")
