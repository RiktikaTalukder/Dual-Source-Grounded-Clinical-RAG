# Week 22 — Farhana — Session 1 Handoff (Updated)

## Summary
Fixed 14+ bugs blocking a trustworthy val200 rerun, migrated the pipeline
from CPU-only to GPU, produced a genuine fresh 200-question run, fixed
BERTScore's measurement validity, ran McNemar/Cohen's h against all 4
baselines — then discovered a serious, unresolved answer-distribution
anomaly while sanity-checking the result against the project's base paper
(Wong & Wong, MECR-RAG). This anomaly is NOT YET RESOLVED and must be
addressed before the val200 accuracy numbers are treated as final.

## Environment (unchanged from last log)
torch 2.6.0+cu124, GPU confirmed (GTX 1660 Ti). ~24GB disk free.
DEVICE/DEVICE_INDEX wired into config.py and all 7 model-loading files.

## Bugs fixed this session (see prior log for items 1-14; new items below)

15. **BERTScore measurement validity.** The hand-rolled mean-pooled-cosine
    workaround in evaluate.py was producing near-uniform ~0.98 scores
    across every method, every run — a measurement artifact, not a real
    result (anisotropy in non-fine-tuned mean-pooled embeddings).
    Diagnostic: direct test of the real `bert_score` library against
    roberta-large returned 0.83 on a clean smoke test, well inside the
    expected 0.5-0.9 range. The original mean-pooling workaround existed
    to avoid a deberta-xlarge-mnli-specific tokenizer bug that does NOT
    apply to roberta-large — confirmed directly. Replaced compute_bertscore()
    with a call to the real library. Verified at full scale on val200:
    dual_source bertscore_mean = 0.757 (status: ok, no warning). Also
    incidentally fixed baseline_no_retrieval's raw exception string —
    the new code path returns a clean "N/A — no evidence retrieved"
    status automatically when no candidate-reference pairs exist.

16. **fixed_chunk == literature_only investigated, not a bug.** Word-count
    diagnostic on val200's literature_passages: mean=630 words, 454/600
    passages exceed the 512-word chunk_fixed() size — so passages ARE
    being truncated, disproving the original "chunks never truncate"
    hypothesis. Real explanation: baseline_fixed_chunk_literature always
    takes chunks[0] (the first 512 words), and PMC abstracts front-load
    conclusion-relevant content, so front-truncation rarely removes
    decision-relevant text for this retrieval setup. This is a genuine
    finding about the project's literature-chunking ablation, not a bug —
    documented for the thesis Discussion. Separately noted: the project's
    actual dynamic-chunking contribution is only ever applied to patient
    notes (pipeline.py's _get_patient_chunks), never to literature passages
    — this baseline does not exercise that contribution at all.

## McNemar / Cohen's h — dual_source vs each baseline (val200, n=200)

| Comparison | dual_source acc | baseline acc | McNemar p | Cohen's h | Significant? |
|---|---|---|---|---|---|
| vs literature_only | 53.0% | 42.5% | 0.0111 | 0.2106 | Yes |
| vs patient_only | 53.0% | 38.5% | 0.0127 | 0.2921 | Yes |
| vs no_retrieval | 53.0% | 44.5% | 0.1002 | 0.1703 | No |
| vs fixed_chunk | 53.0% | 42.5% | 0.0111 | 0.2106 | Yes |

dual_source is significantly better than 3 of 4 baselines. The exception
(no_retrieval, p=0.10) was already the strongest baseline by accuracy
before this anomaly investigation began — see below for why this may not
mean what it appears to mean.

## CRITICAL OPEN ISSUE — answer-distribution anomaly, unresolved

While sanity-checking val200 results against the project's base paper
(Wong & Wong, "Multi-Evidence Clinical Reasoning RAG," JMIR Med Inform
2026), found that the base paper's ablation shows a clean MONOTONIC
pattern: no-retrieval is the WORST configuration, and each added source
improves over it (baseline 0.542 -> guideline-only 0.587 -> case-only
0.775 -> full MECR-RAG 0.802).

OUR pattern does NOT match: no_retrieval (44.5%) outperforms BOTH
literature_only (42.5%) AND patient_only (38.5%) — only dual_source (53.0%)
exceeds no_retrieval. This is inconsistent with the base paper's expected
shape and needs explanation before it goes in the thesis.

Diagnostic checks run so far:
- Leakage check: 0/200 val questions have verbatim-matching contexts in
  the PMC corpus. Leakage in the "exact copy" sense is ruled out. NOTE:
  this check only catches exact substring matches — paraphrased overlap
  is NOT ruled out.
- Per-class accuracy + raw prediction counts reveal SEVERE answer
  collapse, especially in patient_only:
    literature_only  preds: yes=128 no=70   (golds: yes=113 no=68 maybe=19)
    patient_only     preds: yes=23  no=177  (golds: yes=113 no=68 maybe=19)
    dual_source      preds: yes=157 no=40   (golds: yes=113 no=68 maybe=19)
  patient_only predicts "no" on 88.5% of questions when only 34% of gold
  labels are "no" — this looks like near-constant-output collapse, not
  evidence-driven reasoning. dual_source and literature_only both skew
  yes-heavy, in the SAME direction as the gold skew (56.5% yes) — raising
  the possibility that part of the accuracy advantage is "prompt template
  biases toward the majority class" rather than "better reasoning."

REMAINING CHECKS, NOT YET DONE (priority order):
1. Get no_retrieval's prediction distribution — missing from the check
   above, did not print, MUST be obtained before drawing conclusions.
2. Inspect actual patient_only prompts/answer_raw text for negation-pattern
   bias (discharge notes are negation-heavy by genre; could be inducing
   reflexive "no" outputs independent of question content).
3. Controlled template-isolation experiment: feed identical neutral filler
   text into all 3 evidence-bearing prompt templates, see if patient_only's
   template alone induces "no" regardless of content. Not yet designed in
   detail — needs careful construction so only template shape varies.
4. Check whether PubMedQA is in the FLAN Collection's pretraining task
   mixture (Chung et al. 2022 appendix / FLAN GitHub task list) — if so,
   no_retrieval's "no evidence" result may partly reflect memorized
   pretraining association rather than pure zero-shot reasoning, which
   reframes the entire baseline comparison.
5. Cross-reference with the already-planned Week 22 retrieval-quality
   spot check once the above narrows down template-bias vs content-bias
   as the dominant explanation.

DO NOT report val200 accuracy comparisons as final, and DO NOT run the
800-question experiment, until this is resolved or explicitly documented
as an open limitation with a clear paragraph explaining the deviation from
the base paper's expected pattern.

## Corrected val200 results (still stands, but accuracy interpretation is
## now flagged as needing the above investigation before being final)

| Method | Acc | F1 | ECE | R@5 | R@10 | Abstain |
|---|---|---|---|---|---|---|
| dual_source | 53.0% | 0.3136 | 0.0190 | 0.585 | 0.585 | 1.5% |
| literature_only | 42.5% | 0.2744 | 0.2471 | 0.585 | 0.585 | 1.0% |
| patient_only | 38.5% | 0.2444 | 0.2030 | 0.0 | 0.0 | 0.0% |
| no_retrieval | 44.5% | 0.3004 | 0.0550 | 0.0 | 0.0 | 0.0% |
| fixed_chunk | 42.5% | 0.2744 | 0.2471 | 0.585 | 0.585 | 1.0% |

The ECE finding (dual_source 0.019 vs all single-source baselines >0.2)
is NOT directly implicated by the answer-collapse issue in the same way
accuracy is — ECE measures confidence-correctness alignment, which is a
separate question from which label gets predicted most often. This finding
likely survives the investigation above, but should be re-stated alongside
whatever the accuracy investigation concludes, not in isolation.

## Still not started
- Random-patient ablation on val200
- 800-question main experiment (BLOCKED pending the anomaly investigation
  above — do not spend GPU hours on this until resolved or documented)
- McNemar/Cohen's h on 800-question data
- Week 22 ablations
- summary_table.csv / stats_results.json

## Minor unconfirmed item
- patient_retriever.py may still have a leftover duplicate
  "from config import MODEL_REVISIONS" import line — never re-checked
  since being flagged