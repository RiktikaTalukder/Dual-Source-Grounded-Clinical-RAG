"""
run_val200_generate.py
Orchestrator: launches each of the 5 methods as a SEPARATE subprocess,
so no single process ever accumulates models across methods.
"""

import subprocess
import sys
import os

METHODS = [
    "dual_source",
    "baseline_literature_only",
    "baseline_patient_only",
    "baseline_no_retrieval",
    "baseline_fixed_chunk",
]

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_one_method.py")

def main():
    for method in METHODS:
        print(f"\n{'#'*70}\n# LAUNCHING SUBPROCESS FOR: {method}\n{'#'*70}")
        result = subprocess.run([sys.executable, SCRIPT, method])
        if result.returncode != 0:
            print(f"\n*** {method} exited with code {result.returncode} "
                  f"(non-zero / possible crash) — check above for details. ***")
        else:
            print(f"\n--- {method} finished cleanly ---")
    print("\nAll methods attempted. Now run: python run_val200_evaluate.py (fresh terminal)")

if __name__ == "__main__":
    main()