"""
template_isolation_experiment.py
Diagnostic: tests whether the patient_only vs literature_only template
SHAPE alone (independent of retrieved content) explains the answer-
distribution divergence (lit: yes-skewed, patient: no-skewed).

Holds content IDENTICAL (neutral filler, no negation words) across
three prompt shapes:
  - lit_shaped:  matches baseline_literature_only's real template
  - pat_shaped:  matches baseline_patient_only's real template
  - dual_shaped: matches evidence_aligner.align_evidence (dual_source)

If lit_shaped and pat_shaped diverge in yes/no split despite identical
filler content, that's evidence of template-driven bias.
If they come back similar to each other, the bias is content-driven
(real negation-heavy clinical text), not structural.
"""

import sys
import os
import json
import re
import torch
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import MODEL_REVISIONS, GENERATOR_MODEL, DEVICE
from evidence_aligner import align_evidence

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_QUESTIONS = 30   # pilot size — bump to 200 later if signal looks real

# ── Neutral filler: no negation words, no yes/no signal, ~80-100 words,
#    deliberately uninformative about any clinical question ──────────────
NEUTRAL_FILLER = (
    "The case was reviewed in accordance with standard institutional "
    "documentation procedures. Relevant background information was "
    "recorded as part of the routine clinical workflow. The care team "
    "followed established protocols during the encounter. Information "
    "was logged for continuity of care and administrative purposes. "
    "Standard follow-up procedures were noted as part of the documented "
    "course of care."
)

# ── Load model directly (bypass baselines.py's _ensure_resources, which
#    would also try to load the patient FAISS index we don't need here) ──
print(f"Loading {GENERATOR_MODEL}...")
_tokenizer = AutoTokenizer.from_pretrained(
    GENERATOR_MODEL, revision=MODEL_REVISIONS[GENERATOR_MODEL]
)
_llm = AutoModelForSeq2SeqLM.from_pretrained(
    GENERATOR_MODEL, revision=MODEL_REVISIONS[GENERATOR_MODEL]
).to(DEVICE)
_llm.eval()
print(f"Loaded on {DEVICE}.\n")


def _generate(prompt: str) -> str:
    inputs = _tokenizer(prompt, return_tensors="pt",
                         truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = _llm.generate(
            **inputs, max_new_tokens=10, num_beams=4, early_stopping=True
        )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def _extract_answer(raw: str) -> str:
    first_token = raw.strip().split()[0] if raw.strip() else ""
    cleaned = re.sub(r'[^a-z]', '', first_token.lower())
    if cleaned in ("yes", "no", "maybe"):
        return cleaned
    for token in raw.strip().split()[:20]:
        cleaned = re.sub(r'[^a-z]', '', token.lower())
        if cleaned in ("yes", "no", "maybe"):
            return cleaned
    return "abstain"


# ── The three real template shapes, with filler substituted in ───────────
def build_lit_shaped(query):
    return (
        "[SYSTEM: You are a clinical AI assistant. "
        "Answer the question using the evidence below. "
        "Be concise and medically accurate. "
        "Your answer must be exactly one of three words: yes, no, or maybe. "
        "Do not include any explanation, qualification, or additional words.]\n\n"
        f"[LITERATURE EVIDENCE:\n[LIT 1] {NEUTRAL_FILLER}]\n\n"
        "[PATIENT CASE EVIDENCE:\nNot available for this baseline.]\n\n"
        f"[QUESTION: {query}]\n\n"
        "[ANSWER (yes, no, or maybe):]"
    )

def build_pat_shaped(query):
    return (
        "[SYSTEM: You are a clinical AI assistant. "
        "Answer the question using the evidence below. "
        "Be concise and medically accurate. "
        "Your answer must be exactly one of three words: yes, no, or maybe. "
        "Do not include any explanation, qualification, or additional words.]\n\n"
        "[LITERATURE EVIDENCE:\nNot available for this baseline.]\n\n"
        f"[PATIENT CASE EVIDENCE:\n[PATIENT 1] {NEUTRAL_FILLER}]\n\n"
        f"[QUESTION: {query}]\n\n"
        "[ANSWER (yes, no, or maybe):]"
    )

def build_dual_shaped(query):
    # Uses the REAL evidence_aligner.align_evidence — different SYSTEM
    # text and richer patient wrapper, exactly as dual_source uses it.
    return align_evidence(query, [NEUTRAL_FILLER], [NEUTRAL_FILLER])


def load_questions(n):
    val_ids_path = os.path.join(BASE, "data", "pubmedqa", "processed", "val_ids.json")
    raw_path     = os.path.join(BASE, "data", "pubmedqa", "raw", "ori_pqal.json")
    with open(val_ids_path) as f:
        val_ids = json.load(f)
    with open(raw_path) as f:
        raw = json.load(f)
    out = []
    for pmid in val_ids[:n]:
        out.append(raw[pmid]["QUESTION"])
    return out


def main():
    questions = load_questions(N_QUESTIONS)
    results = {"lit_shaped": [], "pat_shaped": [], "dual_shaped": []}

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q[:70]}...")
        for cond, builder in [
            ("lit_shaped", build_lit_shaped),
            ("pat_shaped", build_pat_shaped),
            ("dual_shaped", build_dual_shaped),
        ]:
            raw = _generate(builder(q))
            ans = _extract_answer(raw)
            results[cond].append(ans)
            print(f"    {cond:12s} -> {ans:8s} (raw: {raw[:30]!r})")

    print("\n" + "=" * 60)
    print("RESULTS — same neutral filler, three template shapes")
    print("=" * 60)
    for cond, answers in results.items():
        print(f"{cond}: {dict(Counter(answers))}")

    out_path = os.path.join(BASE, "results", "template_isolation_pilot.json")
    with open(out_path, "w") as f:
        json.dump({"questions": questions, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
