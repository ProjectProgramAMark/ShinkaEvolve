"""Build the frozen R15/R20 comparison tables, figures, and hash manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = TASK_DIR / "results"
PROJECT_ROOT = TASK_DIR.parents[2]
MICROCOSMOS_ROOT = PROJECT_ROOT / "microcosmos"
OPERATORS = (
    "clone",
    "parametric_conservative",
    "parametric_standard",
    "parametric_exploratory",
    "structural",
    "mixed",
)


EVALUATIONS = (
    (
        "R20 gen16",
        "hard training",
        RESULTS_ROOT
        / "evo2-exploratory-r20-compact-stress-20260716"
        / "punctuated/gen_16/results/metrics.json",
    ),
    (
        "R20 gen16",
        "R18 development",
        RESULTS_ROOT
        / "evo2-exploratory-r20-gen16-r18-development-20260716"
        / "evaluation_shinka_gen_16/metrics.json",
    ),
    (
        "fixed standard",
        "R18 development",
        RESULTS_ROOT
        / "evo2-exploratory-r20-gen16-r18-development-20260716"
        / "baselines/fixed_standard/metrics.json",
    ),
    (
        "fixed conservative",
        "R18 development",
        RESULTS_ROOT
        / "evo2-exploratory-r20-gen16-r18-development-20260716"
        / "baselines/fixed_conservative/metrics.json",
    ),
    (
        "R15 gen18",
        "R18 development",
        RESULTS_ROOT
        / "evo2-exploratory-r20-gen16-r18-development-20260716"
        / "baselines/r15_gen18/metrics.json",
    ),
    (
        "matched sparse",
        "R18 development",
        RESULTS_ROOT
        / "evo2-exploratory-r15-gen18-r18-analysis-20260716"
        / "development_matched_sparse/metrics.json",
    ),
    (
        "no-crisis ablation",
        "R18 development",
        RESULTS_ROOT
        / "evo2-exploratory-r15-gen18-r18-analysis-20260716"
        / "development_no_crisis_signal/metrics.json",
    ),
    (
        "R15 gen18",
        "R18 sealed",
        RESULTS_ROOT
        / "evo2-exploratory-r15-gen18-r18-sealed-20260716"
        / "evaluation_r15_gen18/metrics.json",
    ),
)


SOURCE_ARTIFACTS = (
    RESULTS_ROOT
    / "evo2-exploratory-r15-development-gen18-20260715"
    / "programs/shinka_gen_18.py",
    RESULTS_ROOT
    / "evo2-exploratory-r20-compact-stress-20260716"
    / "punctuated/gen_16/main.py",
    RESULTS_ROOT
    / "evo2-exploratory-r15-gen18-r18-analysis-20260716"
    / "programs/matched_sparse_constant.py",
    RESULTS_ROOT
    / "evo2-exploratory-r15-gen18-r18-analysis-20260716"
    / "programs/ablation_no_crisis_signal.py",
    MICROCOSMOS_ROOT
    / "experiments/evo2_ecosystem/r18_artifacts/manifests/development_early_injury.json",
    MICROCOSMOS_ROOT
    / "experiments/evo2_ecosystem/r18_artifacts/manifests/sealed_early_injury.json",
    MICROCOSMOS_ROOT
    / "experiments/evo2_ecosystem/r18_artifacts/founders/index.json",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(label: str, partition: str, path: Path) -> dict:
    metrics = _load(path)
    public = metrics["public"]
    private = metrics["private"]
    fractions = public.get("operator_fraction") or {}
    repeats = private.get("repeat_scores") or []
    return {
        "candidate": label,
        "partition": partition,
        "robust_score": float(metrics["combined_score"]),
        "repeat_scores": [float(value) for value in repeats],
        "sham_effect": float(public["sham_auc_delta"]),
        "injury_effect": float(public["shock_auc_delta"]),
        "survival_rate": float(public["survival_rate"]),
        "ancestor_survival_rate": float(public["ancestor_survival_rate"]),
        "operator_fraction": {
            name: float(fractions.get(name, 0.0)) for name in OPERATORS
        },
        "metrics_path": str(path.resolve()),
        "metrics_sha256": _sha256(path),
    }


def _write_tables(records: list[dict], output_dir: Path) -> None:
    columns = (
        "candidate",
        "partition",
        "robust_score",
        "repeat_1",
        "repeat_2",
        "repeat_3",
        "sham_effect",
        "injury_effect",
        "survival_rate",
        *OPERATORS,
    )
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {
                "candidate": record["candidate"],
                "partition": record["partition"],
                "robust_score": record["robust_score"],
                "sham_effect": record["sham_effect"],
                "injury_effect": record["injury_effect"],
                "survival_rate": record["survival_rate"],
                **record["operator_fraction"],
            }
            for index, value in enumerate(record["repeat_scores"], start=1):
                row[f"repeat_{index}"] = value
            writer.writerow(row)

    lines = [
        "| Candidate | Partition | Robust score | Repeat scores | Sham | Injury |",
        "|---|---|---:|---|---:|---:|",
    ]
    for record in records:
        repeats = ", ".join(f"{value:+.6f}" for value in record["repeat_scores"])
        lines.append(
            f"| {record['candidate']} | {record['partition']} | "
            f"{record['robust_score']:+.6f} | {repeats} | "
            f"{record['sham_effect']:+.6f} | {record['injury_effect']:+.6f} |"
        )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_repeat_scores(records: list[dict], output_dir: Path) -> None:
    labels = [f"{item['candidate']}\n{item['partition']}" for item in records]
    figure, axis = plt.subplots(figsize=(13.5, 5.6))
    for index, record in enumerate(records):
        repeats = np.asarray(record["repeat_scores"], dtype=np.float64)
        axis.scatter(np.full(repeats.shape, index), repeats, s=42, alpha=0.8)
        axis.scatter(index, record["robust_score"], marker="D", s=55, color="black")
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    axis.set_ylabel("paired robust score versus exact clone")
    axis.set_title("Replicated candidate effects (diamond = coherent median)")
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"repeat_scores.{suffix}", dpi=220)
    plt.close(figure)


def _plot_treatment_effects(records: list[dict], output_dir: Path) -> None:
    labels = [f"{item['candidate']}\n{item['partition']}" for item in records]
    x = np.arange(len(records))
    width = 0.38
    figure, axis = plt.subplots(figsize=(13.5, 5.6))
    axis.bar(x - width / 2, [item["sham_effect"] for item in records], width, label="sham")
    axis.bar(x + width / 2, [item["injury_effect"] for item in records], width, label="injury")
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("AUC delta versus exact clone")
    axis.set_title("Sham and injury treatment effects")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"treatment_effects.{suffix}", dpi=220)
    plt.close(figure)


def _plot_operator_fractions(records: list[dict], output_dir: Path) -> None:
    selected = [record for record in records if record["partition"] != "hard training"]
    labels = [f"{item['candidate']}\n{item['partition']}" for item in selected]
    x = np.arange(len(selected))
    bottom = np.zeros(len(selected), dtype=np.float64)
    figure, axis = plt.subplots(figsize=(13.0, 5.6))
    for operator in OPERATORS:
        values = np.asarray(
            [item["operator_fraction"][operator] for item in selected],
            dtype=np.float64,
        )
        axis.bar(x, values, bottom=bottom, label=operator.replace("parametric_", ""))
        bottom += values
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("realized birth fraction")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Realized heredity-operator allocation")
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"operator_fractions.{suffix}", dpi=220)
    plt.close(figure)


def _lineage(search_dir: Path) -> list[dict]:
    records = []
    for generation_dir in sorted(
        search_dir.glob("gen_*"), key=lambda path: int(path.name.split("_")[1])
    ):
        metrics_path = generation_dir / "results/metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = _load(metrics_path)
        records.append(
            {
                "generation": int(generation_dir.name.split("_")[1]),
                "score": float(metrics["combined_score"]),
                "valid": bool(metrics["private"].get("integrity_valid", False)),
            }
        )
    return records


def _plot_lineages(output_dir: Path) -> dict[str, list[dict]]:
    searches = {
        "R15": RESULTS_ROOT
        / "evo2-exploratory-r15-sparse-head-injury-20260715/punctuated",
        "R20": RESULTS_ROOT
        / "evo2-exploratory-r20-compact-stress-20260716/punctuated",
    }
    lineages = {name: _lineage(path) for name, path in searches.items()}
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=False)
    for axis, (name, records) in zip(axes, lineages.items(), strict=True):
        valid = [record for record in records if record["valid"]]
        invalid = [record for record in records if not record["valid"]]
        axis.plot(
            [record["generation"] for record in valid],
            [record["score"] for record in valid],
            marker="o",
            linewidth=1.5,
            label="valid",
        )
        if invalid:
            valid_scores = np.asarray(
                [record["score"] for record in valid], dtype=np.float64
            )
            span = max(float(np.ptp(valid_scores)), 0.01)
            invalid_y = float(valid_scores.min() - 0.10 * span)
            axis.scatter(
                [record["generation"] for record in invalid],
                [invalid_y] * len(invalid),
                marker="x",
                label="invalid (not scored)",
            )
        axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
        axis.set_title(f"{name} Shinka lineage")
        axis.set_xlabel("generation")
        axis.set_ylabel("training score vs clone")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"program_lineages.{suffix}", dpi=220)
    plt.close(figure)
    return lineages


def _write_hashes(output_dir: Path, records: list[dict]) -> None:
    paths = [Path(record["metrics_path"]) for record in records]
    paths.extend(SOURCE_ARTIFACTS)
    paths.extend(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "sha256_manifest.json"
    )
    payload = {
        str(path.resolve()): _sha256(path)
        for path in sorted(set(paths), key=lambda item: str(item.resolve()))
    }
    (output_dir / "sha256_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    records = [_record(*evaluation) for evaluation in EVALUATIONS]
    (arguments.output_dir / "comparison.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_tables(records, arguments.output_dir)
    _plot_repeat_scores(records, arguments.output_dir)
    _plot_treatment_effects(records, arguments.output_dir)
    _plot_operator_fractions(records, arguments.output_dir)
    lineages = _plot_lineages(arguments.output_dir)
    (arguments.output_dir / "program_lineages.json").write_text(
        json.dumps(lineages, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_hashes(arguments.output_dir, records)


if __name__ == "__main__":
    main()
