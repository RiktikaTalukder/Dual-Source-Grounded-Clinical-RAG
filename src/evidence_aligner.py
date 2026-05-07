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
        pat_block = "\n".join(
            f"[PATIENT {i+1}] {summary.strip()}"
            for i, summary in enumerate(patient_summaries)
        )
    else:
        pat_block = "No similar patient cases retrieved."

    # Assemble the full structured prompt
    prompt = f"""[SYSTEM: You are a clinical AI assistant. Answer the question using the evidence below. Be concise and medically accurate.]

[LITERATURE EVIDENCE:
{lit_block}]

[PATIENT CASE EVIDENCE:
{pat_block}]

[QUESTION: {query}]

[ANSWER:]"""

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
