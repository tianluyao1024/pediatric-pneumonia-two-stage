from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / "release_v1_1_sha256.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mismatches = []
    for row in rows:
        path = repo / row["relative_path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        if actual != row["sha256"] or (path.is_file() and path.stat().st_size != int(row["size_bytes"])):
            mismatches.append(row["relative_path"])

    main_folds = pd.read_csv(repo / "folds" / "outer_fold_assignments_main_v1.csv")
    conservative_folds = pd.read_csv(
        repo / "folds" / "outer_fold_assignments_conservative_raw_id_v1.csv"
    )
    runtime = pd.read_csv(repo / "results" / "revision_v1_1" / "runtime_accounting_by_fold.csv")
    report = {
        "hashed_files": len(rows),
        "hash_mismatches": mismatches,
        "main_fold_rows": len(main_folds),
        "conservative_fold_rows": len(conservative_folds),
        "runtime_rows": len(runtime),
        "runtime_complete_rows": int((runtime["status"] == "COMPLETE").sum()),
        "fixed_exploratory_test_reads": int(runtime["test_reads"].sum()),
        "runtime_nan_cells": int(runtime.isna().sum().sum()),
    }
    print(json.dumps(report, indent=2))
    if mismatches or len(main_folds) != 5281 or len(conservative_folds) != 5281:
        raise SystemExit(1)
    if len(runtime) != 30 or report["runtime_complete_rows"] != 30:
        raise SystemExit(1)
    if report["fixed_exploratory_test_reads"] != 0 or report["runtime_nan_cells"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
