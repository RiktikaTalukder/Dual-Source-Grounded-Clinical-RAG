
## Official evaluate.py results (added after initial draft)

Ran `python src/evaluate.py results/generation_samples/baseline_patient_only_val200.json`:
- Accuracy: 37.5% (75/200) — matches hand-computed value exactly
- Per-class accuracy: yes=0.124, no=0.897, maybe=0.0
- Macro-F1: 0.2343 (yes=0.209, no=0.494, maybe=0.0)
- ECE: 0.213
- BERTScore: 0.7449, status "ok" (REFRESHED — replaces stale ~0.98 value
  in val200_metrics_summary.json, using the real bert_score library fix
  from Week 22)
- Recall@5/10: 0.0/0.0 (expected — patient_only has no literature
  evidence, so literature-recall is structurally undefined for this
  method, not a retrieval failure)
- evaluate.py's own built-in check fired: "WARNING: accuracy below 0.40
  threshold" — independent confirmation from the tool itself, not just
  our own read of the number
- Also ran "maybe" check on patient_only's raw output: 0/200 contain
  the word "maybe" — consistent with no_retrieval's same result,
  strengthening the "structural to flan-t5-large beam search" conclusion
  across 2 of 5 methods now.

Note: val200_metrics_summary.json now has the refreshed BERTScore for
patient_only specifically. literature_only, dual_source, and fixed_chunk
still need the same evaluate.py rerun (Phase 1 task from week22, still
not done for those 3 — only patient_only and dual_source have been
re-verified with the real bert_score library so far).
