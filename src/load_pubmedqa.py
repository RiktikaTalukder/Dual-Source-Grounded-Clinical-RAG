import json
from pathlib import Path

RAW_DIR = Path("data/pubmedqa/raw")

DATA_FILES = {
    "pqaa": RAW_DIR / "ori_pqaa.json",
    "pqal": RAW_DIR / "ori_pqal.json",
    "pqau": RAW_DIR / "ori_pqau.json",
}


def load_pubmedqa_file(file_path: Path, split_name: str):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for key, value in data.items():
        samples.append({
            "id": key,
            "split": split_name,
            "question": value.get("QUESTION"),
            "contexts": value.get("CONTEXTS", []),
            "label": value.get("final_decision"),
            "long_answer": value.get("LONG_ANSWER"),
        })

    return samples


def load_pubmedqa_all():
    all_samples = []
    for split_name, file_path in DATA_FILES.items():
        samples = load_pubmedqa_file(file_path, split_name)
        all_samples.extend(samples)
    return all_samples


if __name__ == "__main__":
    print("Checking PubMedQA files...\n")

    total_count = 0
    for split_name, file_path in DATA_FILES.items():
        samples = load_pubmedqa_file(file_path, split_name)
        print(f"{split_name}: {len(samples)} samples from {file_path.name}")
        total_count += len(samples)

    print(f"\nTotal samples: {total_count}")

    dataset = load_pubmedqa_all()
    print("\nExample item:")
    print(dataset[0])