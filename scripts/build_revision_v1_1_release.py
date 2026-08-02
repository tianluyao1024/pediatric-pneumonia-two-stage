from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("PNEUMONIA_PROJECT_ROOT", REPO.parent))
ART = PROJECT / "artifacts" / "revision_credible_audit_v1"
OUT = REPO / "results" / "revision_v1_1"


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_named(source_dir: Path, destination_dir: Path, names: list[str]) -> None:
    for name in names:
        copy_file(source_dir / name, destination_dir / name)


def write_public_environment() -> None:
    source = json.loads((ART / "phase0" / "environment.json").read_text(encoding="utf-8"))
    allowed = [
        "python", "torch", "torchvision", "cuda_runtime", "cuda_available", "gpu",
        "cudnn", "numpy", "pandas", "scikit_learn", "pillow",
        "timestamp_utc_updated", "manifest_sha256",
    ]
    public = {key: source.get(key) for key in allowed}
    public["note"] = "Local executable and filesystem paths were intentionally removed."
    destination = REPO / "environment" / "public_environment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_runtime_accounting() -> None:
    rows: list[dict[str, object]] = []
    root = ART / "phase4b_v2"
    for status_file in sorted(root.glob("*/*/fold_*/status.json")):
        payload = json.loads(status_file.read_text(encoding="utf-8"))
        inner_steps = int(payload.get("inner", {}).get("optimizer_steps", 0) or 0)
        outer_steps = int(payload.get("outer_final", {}).get("optimizer_steps", 0) or 0)
        elapsed = float(payload.get("elapsed_seconds", 0.0) or 0.0)
        peak = int(payload.get("peak_cuda_memory_bytes", 0) or 0)
        budget = payload.get("budget_plan", {}) or {}
        rows.append({
            "analysis": payload.get("analysis"),
            "model": payload.get("kind"),
            "outer_fold": payload.get("fold"),
            "status": payload.get("status"),
            "inner_optimizer_steps": inner_steps,
            "outer_optimizer_steps": outer_steps,
            "total_optimizer_steps": inner_steps + outer_steps,
            "elapsed_seconds": f"{elapsed:.6f}",
            "gpu_hours": f"{elapsed / 3600.0:.6f}",
            "peak_cuda_memory_bytes": peak,
            "peak_cuda_memory_gib": f"{peak / (1024 ** 3):.6f}",
            "test_reads": payload.get("test_reads"),
            "planned_training_flops": budget.get(
                "single_stage_planned_training_flops"
                if payload.get("kind") == "single"
                else "cascade_planned_training_flops"
            ),
            "budget_within_5_percent": budget.get("within_5_percent_tolerance"),
        })
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 Phase-4B status files, found {len(rows)}")
    destination = OUT / "runtime_accounting_by_fold.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_hash_manifest() -> None:
    manifest = REPO / "release_v1_1_sha256.csv"
    files = [
        path for path in REPO.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path != manifest
        and not (
            path.parent == REPO / "figures" / "final_submission"
            and path.suffix.lower() in {".png", ".tiff"}
        )
    ]
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        for path in sorted(files):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            writer.writerow([path.relative_to(REPO).as_posix(), path.stat().st_size, digest])


def main() -> None:
    protocol = PROJECT / "protocol"
    for source in sorted(protocol.glob("*_FINAL.yaml")):
        copy_file(source, REPO / "protocol" / source.name)
    for source in sorted(protocol.glob("protocol_amendment_v*.md")):
        copy_file(source, REPO / "protocol" / source.name)
    copy_file(ART / "phase3" / "analysis_plan_v1.0.yaml", REPO / "protocol" / "analysis_plan_v1.0.yaml")

    copy_named(
        ART / "phase3",
        REPO / "folds",
        [
            "outer_fold_assignments_main_v1.csv",
            "outer_fold_assignments_conservative_raw_id_v1.csv",
            "outer_fold_audit.csv",
            "phase3_validation_report.md",
        ],
    )

    copy_named(
        ART / "phase2",
        REPO / "audit" / "duplicate_review",
        [
            "near_duplicate_method.md",
            "near_duplicate_review_adjudicated.csv",
            "protocol_amendment_v1_phase2_near_duplicate_FINAL.md",
            "phase2_2_manual_review_validation_report.md",
        ],
    )

    for analysis in ["main", "conservative_raw_identifier"]:
        stats = ART / "phase4b_v2" / analysis / "statistics"
        destination = OUT / "phase4b" / analysis
        copy_named(
            stats,
            destination,
            [
                "all_metrics_group_bootstrap.csv",
                "paired_cascade_vs_single_group_bootstrap.csv",
                "cascade_error_pathways.csv",
                "statistics_status.json",
            ],
        )

    copy_named(
        ART / "phase5_source_technical_audit",
        OUT / "phase5_source_directory_audit",
        [
            "source_classifier_oof_group_bootstrap.csv",
            "source_classifier_label_permutation_negative_control.csv",
            "source_audit_status.json",
        ],
    )

    copy_named(
        ART / "phase6_external_stage1_audit",
        OUT / "phase6_external_stage1",
        [
            "external_label_mapping_v1.1_source_level_pa.yaml",
            "external_inference_approval_v1.1.yaml",
            "external_stage1_one_time_results.json",
            "external_dicom_audit_summary.json",
            "PHASE6_EXTERNAL_INFERENCE_COMPLETE.md",
        ],
    )

    source_data = ART / "phase7_publication" / "source_data"
    if source_data.is_dir():
        for source in sorted(source_data.iterdir()):
            if source.is_file():
                copy_file(source, OUT / "figure_source_data" / source.name)

    figures = ART / "phase7_publication" / "figures"
    for source in sorted(figures.iterdir()):
        if source.suffix.lower() in {".svg", ".pdf"}:
            copy_file(source, REPO / "figures" / "final_submission" / source.name)

    write_public_environment()
    write_runtime_accounting()
    write_hash_manifest()
    print(f"Prepared reproducibility release at {REPO}")


if __name__ == "__main__":
    main()
