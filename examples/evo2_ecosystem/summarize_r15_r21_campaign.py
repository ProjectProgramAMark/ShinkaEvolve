"""Build the final frozen R15-R21 tables, figures, lineage, and hash manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import summarize_r15_r20_campaign as previous


TASK_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = TASK_DIR / "results"
PROJECT_ROOT = TASK_DIR.parents[2]
MICROCOSMOS_ROOT = PROJECT_ROOT / "microcosmos"
R21_RUN = RESULTS_ROOT / "evo2-final-r21-cross-founder-20260716"
R21_SEARCH = R21_RUN / "punctuated"
R21_DEVELOPMENT = RESULTS_ROOT / "evo2-final-r21-gen4-development-20260716"
R21_ARTIFACTS = MICROCOSMOS_ROOT / "experiments/evo2_ecosystem/r21_artifacts"

EVALUATIONS = previous.EVALUATIONS + (
    (
        "R21 gen4",
        "R21 training",
        R21_SEARCH / "gen_4/results/metrics.json",
    ),
    (
        "R21 gen4",
        "R21 development",
        R21_DEVELOPMENT / "evaluation_gen4/metrics.json",
    ),
)

SEARCHES = {
    "R15": RESULTS_ROOT / "evo2-exploratory-r15-sparse-head-injury-20260715/punctuated",
    "R20": RESULTS_ROOT / "evo2-exploratory-r20-compact-stress-20260716/punctuated",
    "R21": R21_SEARCH,
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generation_records(search_dir: Path) -> list[dict]:
    records: list[dict] = []
    for generation_dir in sorted(
        search_dir.glob("gen_*"), key=lambda path: int(path.name.split("_")[1])
    ):
        generation = int(generation_dir.name.split("_")[1])
        metrics_path = generation_dir / "results/metrics.json"
        source_path = generation_dir / "main.py"
        if metrics_path.is_file():
            metrics = _load(metrics_path)
            private = metrics.get("private", {})
            public = metrics.get("public", {})
            records.append(
                {
                    "generation": generation,
                    "valid": bool(private.get("integrity_valid", False)),
                    "score": float(metrics["combined_score"]),
                    "repeat_scores": [
                        float(value) for value in private.get("repeat_scores", [])
                    ],
                    "sham_effect": float(public.get("sham_auc_delta", 0.0)),
                    "injury_effect": float(public.get("shock_auc_delta", 0.0)),
                    "operator_fraction": public.get("operator_fraction", {}),
                    "source_sha256": _sha256(source_path),
                    "metrics_sha256": _sha256(metrics_path),
                }
            )
        else:
            records.append(
                {
                    "generation": generation,
                    "valid": False,
                    "score": None,
                    "repeat_scores": [],
                    "sham_effect": None,
                    "injury_effect": None,
                    "operator_fraction": {},
                    "source_sha256": _sha256(source_path),
                    "metrics_sha256": None,
                }
            )
    return records


def _database_lineage() -> list[dict]:
    database_path = R21_SEARCH / "programs.sqlite"
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, generation, parent_id, archive_inspiration_ids,
                   top_k_inspiration_ids, correct, combined_score, metadata,
                   timestamp, island_idx
            FROM programs
            ORDER BY generation, timestamp
            """
        ).fetchall()
    finally:
        connection.close()
    lineage = []
    for row in rows:
        metadata = json.loads(row["metadata"] or "{}")
        lineage.append(
            {
                "id": row["id"],
                "generation": int(row["generation"]),
                "parent_id": row["parent_id"],
                "archive_inspiration_ids": json.loads(
                    row["archive_inspiration_ids"] or "[]"
                ),
                "top_k_inspiration_ids": json.loads(
                    row["top_k_inspiration_ids"] or "[]"
                ),
                "valid": bool(row["correct"]),
                "score": (
                    float(row["combined_score"])
                    if row["combined_score"] is not None
                    else None
                ),
                "patch_name": metadata.get("patch_name"),
                "patch_description": metadata.get("patch_description"),
                "api_cost": float(metadata.get("api_costs", 0.0)),
                "sampling_seconds": metadata.get("sampling_seconds"),
                "evaluation_seconds": metadata.get("evaluation_seconds"),
                "island": row["island_idx"],
                "timestamp": float(row["timestamp"]),
            }
        )
    return lineage


def _runtime_summary() -> dict:
    log_text = (R21_SEARCH / "evolution_run.log").read_text(encoding="utf-8")
    development_metrics = _load(R21_DEVELOPMENT / "evaluation_gen4/metrics.json")
    cost_match = re.search(r"Total API cost: \$(\d+(?:\.\d+)?)", log_text)
    runtime_match = re.search(r"Total runtime: (\d+(?:\.\d+)?) seconds", log_text)
    proposal_match = re.search(r"Total proposals generated: (\d+)", log_text)
    return {
        "target_evaluations": 20,
        "stored_candidate_programs": 20,
        "valid_candidate_programs": 18,
        "invalid_candidate_programs": 2,
        "proposals_generated_after_initial_source": int(proposal_match.group(1)),
        "api_cost_usd": float(cost_match.group(1)),
        "wall_time_seconds": float(runtime_match.group(1)),
        "wall_time_hms": "8h 15m 37s",
        "reported_total_compute_hms": "8h 27m 19s",
        "training_qualifier_generation": 4,
        "training_qualifier_development_repeats": development_metrics["private"][
            "repeat_scores"
        ],
        "development_passed": False,
        "sealed_opened": False,
        "final_conclusion": "negative cross-founder generalization result",
    }


def _write_generation_table(records: list[dict], output_dir: Path) -> None:
    fields = (
        "generation",
        "valid",
        "score",
        "repeat_1",
        "repeat_2",
        "repeat_3",
        "sham_effect",
        "injury_effect",
        "clone_fraction",
        "conservative_fraction",
        "standard_fraction",
        "source_sha256",
        "metrics_sha256",
    )
    with (output_dir / "r21_generations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            fractions = record["operator_fraction"]
            row = {
                "generation": record["generation"],
                "valid": record["valid"],
                "score": record["score"],
                "sham_effect": record["sham_effect"],
                "injury_effect": record["injury_effect"],
                "clone_fraction": fractions.get("clone"),
                "conservative_fraction": fractions.get("parametric_conservative"),
                "standard_fraction": fractions.get("parametric_standard"),
                "source_sha256": record["source_sha256"],
                "metrics_sha256": record["metrics_sha256"],
            }
            for index, value in enumerate(record["repeat_scores"], start=1):
                row[f"repeat_{index}"] = value
            writer.writerow(row)

    lines = [
        "| Gen | Valid | Robust score | Repeat scores | Sham | Injury |",
        "|---:|:---:|---:|---|---:|---:|",
    ]
    for record in records:
        score = "invalid" if record["score"] is None else f"{record['score']:+.6f}"
        repeats = ", ".join(f"{value:+.6f}" for value in record["repeat_scores"])
        sham = "-" if record["sham_effect"] is None else f"{record['sham_effect']:+.6f}"
        injury = (
            "-" if record["injury_effect"] is None else f"{record['injury_effect']:+.6f}"
        )
        lines.append(
            f"| {record['generation']} | {'yes' if record['valid'] else 'no'} | "
            f"{score} | {repeats or '-'} | {sham} | {injury} |"
        )
    (output_dir / "r21_generations.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _plot_lineages(lineages: dict[str, list[dict]], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), sharey=False)
    for axis, (name, records) in zip(axes, lineages.items(), strict=True):
        valid = [record for record in records if record["valid"] and record["score"] is not None]
        invalid = [record for record in records if not record["valid"]]
        axis.plot(
            [record["generation"] for record in valid],
            [record["score"] for record in valid],
            marker="o",
            linewidth=1.4,
            label="valid",
        )
        if invalid and valid:
            scores = np.asarray([record["score"] for record in valid], dtype=np.float64)
            invalid_y = float(scores.min() - max(float(np.ptp(scores)), 0.01) * 0.10)
            axis.scatter(
                [record["generation"] for record in invalid],
                [invalid_y] * len(invalid),
                marker="x",
                s=54,
                label="invalid",
            )
        axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
        axis.set_title(f"{name} Shinka lineage")
        axis.set_xlabel("evaluation")
        axis.set_ylabel("training score vs exact clone")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"program_lineages.{suffix}", dpi=220)
    plt.close(figure)


def _write_hashes(output_dir: Path, comparison_records: list[dict]) -> None:
    paths = [Path(record["metrics_path"]) for record in comparison_records]
    paths.extend(previous.SOURCE_ARTIFACTS)
    paths.extend(
        (
            R21_RUN / "launch.json",
            R21_RUN / "complete.json",
            R21_SEARCH / "evolution_run.log",
            R21_SEARCH / "programs.sqlite",
            R21_SEARCH / "gen_4/main.py",
            R21_DEVELOPMENT / "launch.json",
            R21_DEVELOPMENT / "complete.json",
            R21_DEVELOPMENT / "evaluation_gen4/metrics.json",
            R21_ARTIFACTS / "freeze.json",
            R21_ARTIFACTS / "training_founders/index.json",
            R21_ARTIFACTS / "manifests/training_expanded_r18_early_injury.json",
            R21_ARTIFACTS / "manifests/development_early_injury.json",
            R21_ARTIFACTS / "manifests/sealed_early_injury.json",
            MICROCOSMOS_ROOT / "docs/evo2/evo2-r21-cross-founder-preregistration.md",
        )
    )
    for generation_dir in R21_SEARCH.glob("gen_*"):
        paths.append(generation_dir / "main.py")
        metrics_path = generation_dir / "results/metrics.json"
        if metrics_path.is_file():
            paths.append(metrics_path)
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    comparison_records = [previous._record(*evaluation) for evaluation in EVALUATIONS]
    (arguments.output_dir / "comparison.json").write_text(
        json.dumps(comparison_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    previous._write_tables(comparison_records, arguments.output_dir)
    previous._plot_repeat_scores(comparison_records, arguments.output_dir)
    previous._plot_treatment_effects(comparison_records, arguments.output_dir)
    previous._plot_operator_fractions(comparison_records, arguments.output_dir)

    lineages = {name: _generation_records(path) for name, path in SEARCHES.items()}
    (arguments.output_dir / "program_lineages.json").write_text(
        json.dumps(lineages, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_lineages(lineages, arguments.output_dir)

    r21_generations = lineages["R21"]
    (arguments.output_dir / "r21_generations.json").write_text(
        json.dumps(r21_generations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_generation_table(r21_generations, arguments.output_dir)
    (arguments.output_dir / "r21_program_lineage.json").write_text(
        json.dumps(_database_lineage(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (arguments.output_dir / "compute_and_cost.json").write_text(
        json.dumps(_runtime_summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_hashes(arguments.output_dir, comparison_records)


if __name__ == "__main__":
    main()
