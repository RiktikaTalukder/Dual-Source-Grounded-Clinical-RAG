import json
from pathlib import Path

DATA_FILE = Path("data/pubmedqa/raw/ori_pqal.json")

def load_pubmedqa():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for key, value in data.items():
        samples.append({
            "id": key,
            "question": value.get("QUESTION"),
            "contexts": value.get("CONTEXTS"),
            "label": value.get("final_decision")
        })

    return samples


if __name__ == "__main__":
    dataset = load_pubmedqa()
    print("Dataset size:", len(dataset))
    print("\nExample item:")
    print(dataset[0])