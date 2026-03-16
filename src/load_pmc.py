from pathlib import Path
import json

BASE_DIR = Path("data/pmc_literature")
RAW_DIR = BASE_DIR / "raw"
XML_DIR = BASE_DIR / "xml"
PARSED_DIR = BASE_DIR / "parsed"
META_DIR = BASE_DIR / "metadata"

SAMPLE_PMC_IDS = [
    "PMC1000001",
    "PMC1000002",
    "PMC1000003",
]


def ensure_directories():
    """Create the local PMC folder structure if it does not already exist."""
    for folder in [RAW_DIR, XML_DIR, PARSED_DIR, META_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def write_sample_manifest():
    """
    Write a small sample manifest for future PMC download experiments.
    This is not the actual corpus download.
    """
    manifest_path = META_DIR / "sample_pmc_ids.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "Starter manifest for future PMC OA download/testing",
                "pmc_ids": SAMPLE_PMC_IDS,
            },
            f,
            indent=2,
        )
    return manifest_path


def verify_structure():
    """Print the status of the PMC local folder structure."""
    print("PMC local structure status:\n")
    for folder in [RAW_DIR, XML_DIR, PARSED_DIR, META_DIR]:
        print(f"{folder} -> {'exists' if folder.exists() else 'missing'}")


if __name__ == "__main__":
    print("Preparing PMC local folders...\n")
    ensure_directories()
    manifest_path = write_sample_manifest()
    verify_structure()
    print(f"\nSample manifest written to: {manifest_path}")
    print("\nPMC helper setup complete.")