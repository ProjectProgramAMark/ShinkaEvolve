"""Direct exploratory evaluator for six-action Evo² heredity programs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import jax
from shinka.core.wrap_eval import save_json_results

from examples.evo2_ecosystem import evaluate
from examples.evo2_ecosystem import run_spec


TASK_DIR = Path(__file__).resolve().parent
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
    ancestor = evaluate._load_candidate(
        initial,
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
    metrics["private"].update(
        {
            "backend": jax.default_backend(),
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
    parser.add_argument("--regime", choices=evaluate.REGIMES, required=True)
    parser.add_argument("--numerical_repeats", type=int, default=1)
    arguments = parser.parse_args()

    evaluate._scrub_sensitive_environment()
    try:
        metrics = evaluate_program(
            arguments.program_path,
            arguments.manifest_path,
            arguments.founder_index_path,
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
