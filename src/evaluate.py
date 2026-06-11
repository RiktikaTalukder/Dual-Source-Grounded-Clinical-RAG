"""
src/evaluate.py
Single source of truth for all evaluation metrics.
Usage:
    python src/evaluate.py results/generation_samples/dual_source_20.json
    python src/evaluate.py results/generation_samples/baseline_literature_only_20.json
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from scipy.stats import pearsonr

# ── Constants ────────────────────────────────────────────────────────────────
LABELS          = ["yes", "no", "maybe"]
MAJORITY_ACC    = 0.55          # ~55% yes in PubMedQA expert subset
RECALL_THRESHOLD = 0.50         # cosine similarity hit threshold for Recall@K
ECE_BINS        = 10
ABSTAIN_WARN    = 0.10          # hard warning threshold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PQAL_PATH = os.path.join(ROOT, "data", "pubmedqa", "raw", "ori_pqal.json")
FIGURES_DIR = os.path.join(ROOT, "results", "figures")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_outputs(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    return data


def normalise_label(label):
    if label is None:
        return "abstain"
    label = str(label).strip().lower()
    if label in LABELS:
        return label
    return "abstain"


# ── 1. Accuracy ──────────────────────────────────────────────────────────────

def compute_accuracy(golds, preds):
    """
    Three-class accuracy overall and per class.
    Abstains count as incorrect (GAP 9).
    """
    assert len(golds) == len(preds)
    n = len(golds)
    overall_correct = sum(g == p for g, p in zip(golds, preds))
    overall_acc = overall_correct / n

    per_class = {}
    for label in LABELS:
        indices = [i for i, g in enumerate(golds) if g == label]
        if indices:
            correct = sum(preds[i] == label for i in indices)
            per_class[label] = round(correct / len(indices), 4)
        else:
            per_class[label] = None   # no gold examples for this class

    return {
        "accuracy_overall": round(overall_acc, 4),
        "accuracy_yes":     per_class["yes"],
        "accuracy_no":      per_class["no"],
        "accuracy_maybe":   per_class["maybe"],
        "majority_baseline": MAJORITY_ACC,
        "n_total":          n,
        "n_correct":        overall_correct,
    }


# ── 2. Macro-F1 ──────────────────────────────────────────────────────────────

def compute_macro_f1(golds, preds):
    """
    Macro-average F1 across yes/no/maybe.
    Abstain predictions will appear as their own class and hurt F1 correctly.
    """
    f1 = f1_score(golds, preds, labels=LABELS, average="macro", zero_division=0)
    per_class_f1 = f1_score(golds, preds, labels=LABELS, average=None,
                            zero_division=0)
    return {
        "macro_f1": round(float(f1), 4),
        "f1_yes":   round(float(per_class_f1[0]), 4),
        "f1_no":    round(float(per_class_f1[1]), 4),
        "f1_maybe": round(float(per_class_f1[2]), 4),
    }


# ── 3. BERTScore ─────────────────────────────────────────────────────────────

def compute_bertscore(data):
    """
    BERTScore F1 between answer_raw and each retrieved literature passage.
    Judge model: microsoft/deberta-xlarge-mnli.
    Computed manually via mean-pooled embeddings + cosine similarity to avoid
    bert_score 0.3.x tokenizer overflow bug with deberta on recent tokenizers.
    Returns mean BERTScore F1 across all queries.
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        return {"bertscore_mean": None, "bertscore_error": "transformers not installed"}

    MODEL_NAME = "roberta-large"
    MAX_LENGTH = 512

    print(f"  [BERTScore] Loading {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model     = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
    except Exception as e:
        return {"bertscore_mean": None, "bertscore_error": f"model load failed: {e}"}

    def encode_texts(texts):
        """Mean-pool last hidden state over non-padding tokens."""
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**enc, output_hidden_states=False)
        hidden = out.last_hidden_state          # (B, T, H)
        mask   = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)   # (B, H)
        # L2-normalise
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled

    def cosine_f1(cand_emb, ref_emb):
        """Token-level BERTScore F1 approximation using sentence embeddings."""
        sim = float(torch.dot(cand_emb, ref_emb).clamp(-1, 1))
        # P = R = F1 = sim when using sentence-level embeddings
        return sim

    pairs        = []   # (query_idx, candidate_text, reference_text)
    query_indices = []

    for i, entry in enumerate(data):
        answer   = entry.get("answer_raw", "") or ""
        passages = entry.get("literature_passages", [])
        if not passages or not answer.strip():
            continue
        for p in passages:
            text = p.get("text", "") if isinstance(p, dict) else str(p)
            if text.strip():
                pairs.append((answer, text))
                query_indices.append(i)

    if not pairs:
        return {"bertscore_mean": None, "bertscore_error": "no valid candidate-reference pairs"}

    print(f"  [BERTScore] Scoring {len(pairs)} pairs (deberta-xlarge-mnli)...")

    try:
        BATCH = 8
        all_f1 = []
        for start in range(0, len(pairs), BATCH):
            batch      = pairs[start:start + BATCH]
            cands      = [p[0] for p in batch]
            refs       = [p[1] for p in batch]
            cand_embs  = encode_texts(cands)
            ref_embs   = encode_texts(refs)
            for c_emb, r_emb in zip(cand_embs, ref_embs):
                all_f1.append(cosine_f1(c_emb, r_emb))

        from collections import defaultdict
        query_scores = defaultdict(list)
        for idx, f1 in zip(query_indices, all_f1):
            query_scores[idx].append(f1)
        per_query_means = [np.mean(v) for v in query_scores.values()]
        mean_bs = float(np.mean(per_query_means))

        status = "ok"
        if not (0.5 <= mean_bs <= 0.9):
            status = f"WARNING: mean BERTScore {mean_bs:.4f} outside expected 0.5-0.9 range"

        return {
            "bertscore_mean":    round(mean_bs, 4),
            "bertscore_n_pairs": len(pairs),
            "bertscore_status":  status,
        }
    except Exception as e:
        return {"bertscore_mean": None, "bertscore_error": str(e)}


# ── 4. Recall@K ──────────────────────────────────────────────────────────────

def compute_recall_at_k(data, k_values=(5, 10)):
    """
    Recall@5 and Recall@10 for literature retrieval.
    Judge model: sentence-transformers/all-mpnet-base-v2 (NOT S-PubMedBert).
    A retrieved passage is a hit if cosine similarity to any gold context >= 0.5.
    Gold contexts come from ori_pqal.json via gold_contexts field in output.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        return {"recall@5": None, "recall@10": None,
                "recall_error": "sentence_transformers not installed"}

    print("  [Recall@K] Loading all-mpnet-base-v2 judge model...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    recall_scores = {k: [] for k in k_values}

    for entry in data:
        gold_contexts = entry.get("gold_contexts", [])
        passages = entry.get("literature_passages", [])

        if not gold_contexts or not passages:
            for k in k_values:
                recall_scores[k].append(0.0)
            continue

        # Encode gold contexts
        gold_texts = [str(g) for g in gold_contexts if str(g).strip()]
        if not gold_texts:
            for k in k_values:
                recall_scores[k].append(0.0)
            continue

        gold_embs = model.encode(gold_texts, normalize_embeddings=True,
                                 show_progress_bar=False)

        for k in k_values:
            top_k = passages[:k]
            hit = False
            for p in top_k:
                text = p.get("text", "") if isinstance(p, dict) else str(p)
                if not text.strip():
                    continue
                p_emb = model.encode([text], normalize_embeddings=True,
                                     show_progress_bar=False)
                sims = np.dot(gold_embs, p_emb.T).flatten()
                if float(np.max(sims)) >= RECALL_THRESHOLD:
                    hit = True
                    break
            recall_scores[k].append(1.0 if hit else 0.0)

    result = {}
    for k in k_values:
        scores = recall_scores[k]
        result[f"recall@{k}"] = round(float(np.mean(scores)), 4) if scores else None

    return result


# ── 5. ECE ───────────────────────────────────────────────────────────────────

def compute_ece(golds, preds, confidences, n_bins=ECE_BINS):
    """
    Three-class ECE using correctness-based signal.
    correctness = 1.0 if answer_extracted == gold_label, else 0.0.
    No 0.5 for maybe — gold maybe vs predicted maybe = correct (1.0),
    gold maybe vs predicted yes/no = incorrect (0.0). (B5 Option 1)
    Abstains are treated as incorrect (GAP 9).
    """
    correctness = np.array([1.0 if g == p else 0.0
                            for g, p in zip(golds, preds)], dtype=float)
    confs = np.array(confidences, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confs >= lo) & (confs < hi) if i < n_bins - 1 \
               else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            bin_data.append({
                "bin_lower": round(lo, 2), "bin_upper": round(hi, 2),
                "count": 0, "avg_conf": None, "avg_correct": None
            })
            continue
        avg_conf    = float(np.mean(confs[mask]))
        avg_correct = float(np.mean(correctness[mask]))
        ece += (mask.sum() / len(confs)) * abs(avg_conf - avg_correct)
        bin_data.append({
            "bin_lower":   round(lo, 2),
            "bin_upper":   round(hi, 2),
            "count":       int(mask.sum()),
            "avg_conf":    round(avg_conf, 4),
            "avg_correct": round(avg_correct, 4),
        })

    status = "ok"
    if ece >= 0.3:
        status = f"WARNING: ECE {ece:.4f} >= 0.3 threshold"

    return {
        "ece":        round(float(ece), 4),
        "ece_status": status,
        "bin_data":   bin_data,
    }


# ── 6. Reliability diagram ───────────────────────────────────────────────────

def plot_reliability_diagram(ece_result, method_name, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [Reliability] matplotlib not installed — skipping diagram.")
        return None

    os.makedirs(out_dir, exist_ok=True)

    bin_data = [b for b in ece_result["bin_data"] if b["count"] > 0]
    if not bin_data:
        print("  [Reliability] No non-empty bins — skipping diagram.")
        return None

    avg_confs   = [b["avg_conf"]    for b in bin_data]
    avg_correct = [b["avg_correct"] for b in bin_data]

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.bar(avg_confs, avg_correct, width=0.07,
           color="#4C72B0", alpha=0.8, label="Actual accuracy")

    for ac, co in zip(avg_confs, avg_correct):
        lo = min(ac, co)
        hi = max(ac, co)
        ax.bar(ac, hi - lo, bottom=lo, width=0.07,
               color="red", alpha=0.3)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    legend_elements = [
        mpatches.Patch(color="#4C72B0", alpha=0.8,  label="Actual accuracy"),
        mpatches.Patch(color="red",     alpha=0.3,  label="Miscalibration gap"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="Perfect calibration"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    safe_name = method_name.replace("/", "_").replace(" ", "_")
    ax.set_title(f"Reliability Diagram — {method_name}\nECE = {ece_result['ece']:.4f}",
                 fontsize=11)
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Actual Accuracy", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"reliability_{safe_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Reliability] Diagram saved → {out_path}")
    return out_path


# ── 7. Abstain rate ──────────────────────────────────────────────────────────

def compute_abstain_rate(preds):
    n = len(preds)
    n_abstain = sum(1 for p in preds if p == "abstain")
    rate = n_abstain / n if n > 0 else 0.0
    result = {"abstain_rate": round(rate, 4), "n_abstain": n_abstain, "n_total": n}
    if rate > ABSTAIN_WARN:
        result["abstain_warning"] = (
            "WARNING: abstain rate exceeds 10%. Pipeline generation is failing "
            "to produce valid labels for more than 1 in 10 queries. "
            "Do not proceed to main experiment until this is resolved."
        )
    return result


# ── 8. McNemar + Cohen's h ───────────────────────────────────────────────────

def compute_mcnemar_cohens_h(golds_a, preds_a, golds_b, preds_b,
                              name_a="method_a", name_b="method_b"):
    """
    McNemar test and Cohen's h for two paired methods on the same questions.
    Both lists must be aligned (same questions in same order).
    """
    from statsmodels.stats.contingency_tables import mcnemar

    if len(preds_a) != len(preds_b):
        return {"error": "prediction lists must be same length for McNemar test"}

    correct_a = np.array([1 if g == p else 0
                          for g, p in zip(golds_a, preds_a)])
    correct_b = np.array([1 if g == p else 0
                          for g, p in zip(golds_b, preds_b)])

    # McNemar contingency table:
    # [both correct, a correct b wrong]
    # [a wrong b correct, both wrong]
    n_both_correct   = int(np.sum((correct_a == 1) & (correct_b == 1)))
    n_a_only         = int(np.sum((correct_a == 1) & (correct_b == 0)))
    n_b_only         = int(np.sum((correct_a == 0) & (correct_b == 1)))
    n_both_wrong     = int(np.sum((correct_a == 0) & (correct_b == 0)))

    table = [[n_both_correct, n_a_only],
             [n_b_only,       n_both_wrong]]

    try:
        result_mc = mcnemar(table, exact=True)
        p_value = float(result_mc.pvalue)
    except Exception as e:
        p_value = None

    # Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    n = len(preds_a)
    p1 = float(np.mean(correct_a))
    p2 = float(np.mean(correct_b))
    cohens_h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

    return {
        "comparison":       f"{name_a} vs {name_b}",
        "accuracy_a":       round(p1, 4),
        "accuracy_b":       round(p2, 4),
        "mcnemar_p":        round(p_value, 4) if p_value is not None else None,
        "cohens_h":         round(float(cohens_h), 4),
        "contingency_table": table,
        "n":                n,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def evaluate(path, paired_path=None, save_json=True):
    print(f"\n{'='*65}")
    print(f"  evaluate.py — {os.path.basename(path)}")
    print(f"{'='*65}")

    data = load_outputs(path)
    print(f"  Loaded {len(data)} entries.")

    golds  = [normalise_label(d.get("gold_label")) for d in data]
    preds  = [normalise_label(d.get("answer_extracted")) for d in data]
    confs  = [float(d.get("confidence", 0.0)) for d in data]

    # Derive method name from filename
    method_name = os.path.splitext(os.path.basename(path))[0]

    results = {"method": method_name, "n": len(data)}

    # 1. Abstain rate (first — hard stop warning)
    print("\n[1] Abstain rate")
    abstain = compute_abstain_rate(preds)
    results["abstain"] = abstain
    print(f"    abstain_rate : {abstain['abstain_rate']*100:.1f}%  "
          f"({abstain['n_abstain']}/{abstain['n_total']})")
    if "abstain_warning" in abstain:
        print(f"\n  *** {abstain['abstain_warning']} ***\n")

    # 2. Accuracy
    print("\n[2] Accuracy")
    acc = compute_accuracy(golds, preds)
    results["accuracy"] = acc
    print(f"    overall      : {acc['accuracy_overall']*100:.1f}%  "
          f"({acc['n_correct']}/{acc['n_total']})")
    print(f"    majority_baseline: {acc['majority_baseline']*100:.0f}%")
    print(f"    per-class    : yes={acc['accuracy_yes']}  "
          f"no={acc['accuracy_no']}  maybe={acc['accuracy_maybe']}")
    if acc["accuracy_overall"] < 0.40:
        print("    WARNING: accuracy below 0.40 threshold")

    # 3. Macro-F1
    print("\n[3] Macro-F1")
    f1 = compute_macro_f1(golds, preds)
    results["f1"] = f1
    print(f"    macro_f1     : {f1['macro_f1']}")
    print(f"    per-class    : yes={f1['f1_yes']}  no={f1['f1_no']}  "
          f"maybe={f1['f1_maybe']}")

    # 4. ECE
    print("\n[4] ECE (three-class, correctness-based)")
    ece_result = compute_ece(golds, preds, confs)
    results["ece"] = ece_result
    print(f"    ECE          : {ece_result['ece']}")
    if ece_result["ece_status"] != "ok":
        print(f"    {ece_result['ece_status']}")

    # 5. Reliability diagram
    print("\n[5] Reliability diagram")
    diagram_path = plot_reliability_diagram(ece_result, method_name, FIGURES_DIR)
    results["reliability_diagram"] = diagram_path

    # 6. BERTScore
    print("\n[6] BERTScore (deberta-xlarge-mnli judge)")
    bs = compute_bertscore(data)
    results["bertscore"] = bs
    if bs.get("bertscore_mean") is not None:
        print(f"    bertscore_mean: {bs['bertscore_mean']}")
        print(f"    status        : {bs['bertscore_status']}")
    else:
        print(f"    ERROR: {bs.get('bertscore_error')}")

    # 7. Recall@K
    print("\n[7] Recall@5 and Recall@10 (all-mpnet-base-v2 judge)")
    recall = compute_recall_at_k(data)
    results["recall"] = recall
    print(f"    Recall@5     : {recall.get('recall@5')}")
    print(f"    Recall@10    : {recall.get('recall@10')}")

    # 8. McNemar + Cohen's h (only if paired file provided)
    if paired_path:
        print(f"\n[8] McNemar + Cohen's h vs {os.path.basename(paired_path)}")
        paired_data  = load_outputs(paired_path)
        paired_golds = [normalise_label(d.get("gold_label")) for d in paired_data]
        paired_preds = [normalise_label(d.get("answer_extracted")) for d in paired_data]
        paired_name  = os.path.splitext(os.path.basename(paired_path))[0]
        stats = compute_mcnemar_cohens_h(
            golds, preds, paired_golds, paired_preds,
            name_a=method_name, name_b=paired_name
        )
        results["mcnemar_cohens_h"] = stats
        print(f"    {stats['comparison']}")
        print(f"    McNemar p    : {stats['mcnemar_p']}")
        print(f"    Cohen's h    : {stats['cohens_h']}")
        print(f"    accuracy_a   : {stats['accuracy_a']}  "
              f"accuracy_b: {stats['accuracy_b']}")
    else:
        print("\n[8] McNemar + Cohen's h — skipped (no paired file provided)")
        print("    Pass a second file path as argument to enable.")

    # Save JSON
    if save_json:
        stem    = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(ROOT, "results", "main_experiment")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"metrics_{stem}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Metrics saved → {out_path}")

    print(f"\n{'='*65}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pipeline outputs.")
    parser.add_argument("path",          help="Path to output JSON file to evaluate")
    parser.add_argument("paired_path",   nargs="?", default=None,
                        help="Optional second output file for McNemar/Cohen's h comparison")
    args = parser.parse_args()
    evaluate(args.path, paired_path=args.paired_path)
