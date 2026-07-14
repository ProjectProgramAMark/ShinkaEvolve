"""Direct exploratory evaluator for six-action Evo² heredity programs."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys

import jax
from shinka.core.wrap_eval import save_json_results


TASK_DIR = Path(__file__).resolve().parent
SHINKA_ROOT = TASK_DIR.parents[1]
if str(SHINKA_ROOT) not in sys.path:
    sys.path.insert(0, str(SHINKA_ROOT))

from examples.evo2_ecosystem import evaluate  # noqa: E402
from examples.evo2_ecosystem import run_spec  # noqa: E402


PROJECT_ROOT = TASK_DIR.parents[2]
MICROCOSMOS_ROOT = PROJECT_ROOT / "microcosmos"


def _microcosmos_imports():
    for path in (MICROCOSMOS_ROOT, MICROCOSMOS_ROOT / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from experiments.evo2_ecosystem.episode import (  # noqa: PLC0415
        SimulatorConfig,
        evaluate_manifest_paired_delta,
    )
    from experiments.evo2_ecosystem.protocol import (  # noqa: PLC0415
        canonical_manifest_bytes,
        manifest_from_json_bytes,
        manifest_sha256,
    )
    from microcosmos.heredity import make_r4_offspring_policy  # noqa: PLC0415

    return (
        SimulatorConfig,
        evaluate_manifest_paired_delta,
        canonical_manifest_bytes,
        manifest_from_json_bytes,
        manifest_sha256,
        make_r4_offspring_policy,
    )


def evaluate_program(
    program_path: str | Path,
    manifest_path: str | Path,
    founder_index_path: str | Path,
    *,
    ancestor_program_path: str | Path | None = None,
    regime: str,
    numerical_repeats: int,
) -> dict[str, object]:
    if regime not in evaluate.REGIMES:
        raise ValueError("regime must be stable or punctuated")
    if numerical_repeats < 1 or numerical_repeats % 2 == 0:
        raise ValueError("numerical_repeats must be a positive odd integer")

    initial = TASK_DIR / "initial_r4.py"
    candidate = evaluate._load_candidate(
        program_path,
        run_spec.R4_CONTRACT,
        initial,
    )
    ancestor_path = initial if ancestor_program_path is None else ancestor_program_path
    ancestor = evaluate._load_candidate(
        ancestor_path,
        run_spec.R4_CONTRACT,
        initial,
    )
    for module in (candidate, ancestor):
        evaluate._smoke_validate_candidate(
            module,
            6,
            contract_version=run_spec.R4_CONTRACT,
            runtime_budget_ms=100.0,
        )

    (
        SimulatorConfig,
        paired_evaluator,
        canonical_manifest_bytes,
        manifest_from_json_bytes,
        manifest_sha256,
        make_r4_offspring_policy,
    ) = _microcosmos_imports()
    raw = Path(manifest_path).read_bytes()
    manifest = manifest_from_json_bytes(raw)
    if raw != canonical_manifest_bytes(manifest) + b"\n":
        raise ValueError("exploratory manifest is not canonical")

    evaluation = paired_evaluator(
        manifest,
        SimulatorConfig(),
        make_r4_offspring_policy(candidate.make_offspring),
        make_r4_offspring_policy(ancestor.make_offspring),
        regime=regime,
        founder_index_path=founder_index_path,
        numerical_repeats=numerical_repeats,
        capture_finalists=False,
    )
    metrics = evaluate._episode_metrics_r4(evaluation, regime)
    violations = int(metrics["private"]["policy_violation_count"])
    physical_integrity = all(
        bool(getattr(episode, "integrity_valid", True))
        for episode in (*evaluation.episodes, *evaluation.ancestor_episodes)
    )
    raw_score = float(evaluation.candidate_score)
    valid = physical_integrity and violations == 0 and math.isfinite(raw_score)
    score = raw_score if valid else -2.0
    public = metrics["public"]
    public["score"] = score
    public["ancestor_survival_rate"] = sum(
        float(episode.survived) for episode in evaluation.ancestor_episodes
    ) / len(evaluation.ancestor_episodes)
    metrics["combined_score"] = score
    metrics["private"].update(
        {
            "integrity_valid": valid,
            "physical_integrity_valid": physical_integrity,
            "pair_deltas": [float(value) for value in evaluation.pair_deltas.tolist()],
            "repeat_scores": [
                float(value) for value in evaluation.repeat_scores.tolist()
            ],
            "selected_repeat_index": int(evaluation.selected_repeat_index),
            "survival_is_diagnostic": True,
            "adaptive_observations": list(evaluation.adaptive_observations),
        }
    )
    feedback = [
        f"paired ancestor delta={score:.4f}",
        f"sham delta={public['sham_auc_delta']:.4f}",
    ]
    if regime == "punctuated":
        feedback.append(f"shock delta={public['shock_auc_delta']:.4f}")
    feedback.extend(
        [
            f"survival={public['survival_rate']:.2f}",
            f"mean births={public['mean_births']:.1f}",
            "survival is scored, not a validity gate",
        ]
    )
    metrics["text_feedback"] = "; ".join(feedback) + "."
    metrics["private"].update(
        {
            "backend": jax.default_backend(),
            "ancestor_sha256": ancestor._evo2_source_sha256,
            "candidate_sha256": candidate._evo2_source_sha256,
            "exploratory": True,
            "founder_index_sha256": hashlib.sha256(
                Path(founder_index_path).read_bytes()
            ).hexdigest(),
            "manifest_sha256": manifest_sha256(manifest),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--founder_index_path", required=True)
    parser.add_argument("--ancestor_program_path")
    parser.add_argument("--regime", choices=evaluate.REGIMES, required=True)
    parser.add_argument("--numerical_repeats", type=int, default=1)
    arguments = parser.parse_args()

    evaluate._scrub_sensitive_environment()
    try:
        metrics = evaluate_program(
            arguments.program_path,
            arguments.manifest_path,
            arguments.founder_index_path,
            ancestor_program_path=arguments.ancestor_program_path,
            regime=arguments.regime,
            numerical_repeats=arguments.numerical_repeats,
        )
        correct = bool(metrics["private"]["integrity_valid"])
        error = None if correct else "ecosystem integrity check failed"
    except Exception as caught:
        metrics = {
            "combined_score": -2.0,
            "public": {"score": -2.0},
            "private": {
                "candidate_sha256": hashlib.sha256(
                    Path(arguments.program_path).read_bytes()
                ).hexdigest(),
                "error_type": type(caught).__name__,
                "exploratory": True,
                "full_evaluation_performed": False,
                "integrity_valid": False,
            },
            "text_feedback": f"Exploratory evaluation failed: {type(caught).__name__}",
        }
        correct = False
        error = f"exploratory evaluation failed ({type(caught).__name__})"
    save_json_results(arguments.results_dir, metrics, correct, error, verbose=False)


if __name__ == "__main__":
    main()
