"""Render frozen Evo² recovery, mechanism, and program-lineage figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OPERATORS = (
    "clone",
    "conservative",
    "standard",
    "exploratory",
    "structural",
    "mixed",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _curve_summary(episodes: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([episode[key] for episode in episodes], dtype=np.float64)
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, 1.96 * sem


def _plot_recovery(
    trajectory: dict,
    output_dir: Path,
    candidate_label: str,
) -> dict:
    groups: dict[str, list[dict]] = {}
    for episode in trajectory["episodes"]:
        groups.setdefault(episode["event_kind"], []).append(episode)
    figure, axes = plt.subplots(1, len(groups), figsize=(6.4 * len(groups), 4.4))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    data = {}
    for axis, (event_kind, episodes) in zip(axes, sorted(groups.items()), strict=True):
        candidate_mean, candidate_ci = _curve_summary(
            episodes, "candidate_productivity"
        )
        reference_mean, reference_ci = _curve_summary(
            episodes, "reference_productivity"
        )
        x = np.arange(candidate_mean.size)
        axis.plot(x, candidate_mean, label=candidate_label, linewidth=2.2)
        axis.fill_between(
            x,
            candidate_mean - candidate_ci,
            candidate_mean + candidate_ci,
            alpha=0.18,
        )
        axis.plot(x, reference_mean, label="fixed clone", linewidth=2.2)
        axis.fill_between(
            x,
            reference_mean - reference_ci,
            reference_mean + reference_ci,
            alpha=0.18,
        )
        axis.set_title("sham" if event_kind == "null" else "injury")
        axis.set_xlabel("post-event checkpoint")
        axis.set_ylabel("normalized resource productivity")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
        data[event_kind] = {
            "candidate_mean": candidate_mean.tolist(),
            "candidate_95ci": candidate_ci.tolist(),
            "reference_mean": reference_mean.tolist(),
            "reference_95ci": reference_ci.tolist(),
        }
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"recovery_curves.{suffix}", dpi=220)
    plt.close(figure)
    return data


def _realized_fraction(episodes: list[dict]) -> np.ndarray:
    counts = np.asarray(
        [episode["candidate_operator_counts"] for episode in episodes],
        dtype=np.float64,
    ).sum(axis=0)
    return counts / counts.sum()


def _plot_mechanism(
    metrics: dict,
    trajectory: dict,
    output_dir: Path,
    run_label: str,
) -> dict:
    public = metrics["public"]
    pre = np.asarray(public["pre_selection_probability"], dtype=np.float64)
    post = np.asarray(public["post_selection_probability"], dtype=np.float64)
    sham = _realized_fraction(
        [episode for episode in trajectory["episodes"] if episode["event_kind"] == "null"]
    )
    injury = _realized_fraction(
        [episode for episode in trajectory["episodes"] if episode["event_kind"] != "null"]
    )
    x = np.arange(len(OPERATORS))
    width = 0.20
    figure, axis = plt.subplots(figsize=(10.0, 4.6))
    axis.bar(x - 1.5 * width, pre, width, label="before event")
    axis.bar(x - 0.5 * width, post, width, label="after event")
    axis.bar(x + 0.5 * width, sham, width, label="sham births")
    axis.bar(x + 1.5 * width, injury, width, label="injury births")
    axis.set_xticks(x, OPERATORS, rotation=20, ha="right")
    axis.set_ylabel("probability / fraction")
    axis.set_title(f"{run_label} heredity allocation")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"operator_mechanism.{suffix}", dpi=220)
    plt.close(figure)
    return {
        "pre": pre.tolist(),
        "post": post.tolist(),
        "sham_realized": sham.tolist(),
        "injury_realized": injury.tolist(),
    }


def _plot_lineage(
    search_dir: Path,
    output_dir: Path,
    run_label: str,
) -> list[dict]:
    records = []
    for generation_dir in sorted(
        search_dir.glob("gen_*"), key=lambda path: int(path.name.split("_")[1])
    ):
        metrics_path = generation_dir / "results" / "metrics.json"
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
    figure, axis = plt.subplots(figsize=(9.0, 4.4))
    axis.plot(
        [record["generation"] for record in records],
        [record["score"] for record in records],
        marker="o",
        linewidth=1.6,
    )
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axis.set_xlabel("Shinka generation")
    axis.set_ylabel("paired robust score vs clone")
    axis.set_title(f"{run_label} program-evolution lineage")
    axis.grid(alpha=0.22)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"program_lineage.{suffix}", dpi=220)
    plt.close(figure)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-label", default="Shinka policy")
    parser.add_argument("--run-label", default="Evo²")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = _load(arguments.trajectory_json)
    figure_data = {
        "recovery": _plot_recovery(
            trajectory,
            arguments.output_dir,
            arguments.candidate_label,
        ),
        "mechanism": _plot_mechanism(
            _load(arguments.metrics_json),
            trajectory,
            arguments.output_dir,
            arguments.run_label,
        ),
        "lineage": _plot_lineage(
            arguments.search_dir,
            arguments.output_dir,
            arguments.run_label,
        ),
    }
    (arguments.output_dir / "figure_data.json").write_text(
        json.dumps(figure_data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
