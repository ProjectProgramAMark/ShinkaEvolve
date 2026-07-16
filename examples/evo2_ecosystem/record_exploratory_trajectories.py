"""Record paired Evo² productivity trajectories for a frozen analysis run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


SHINKA_ROOT = Path(__file__).resolve().parents[2]
if str(SHINKA_ROOT) not in sys.path:
    sys.path.insert(0, str(SHINKA_ROOT))

from examples.evo2_ecosystem import evaluate  # noqa: E402
from examples.evo2_ecosystem import evaluate_exploratory  # noqa: E402
from examples.evo2_ecosystem import run_spec  # noqa: E402


def _episode_record(world, candidate, reference) -> dict[str, object]:
    return {
        "scenario_id": world.scenario_id,
        "pair_id": world.pair_id,
        "founder_id": world.founder_id,
        "event_kind": world.event_kind.value,
        "candidate_productivity": [
            float(value) for value in np.asarray(candidate.post_event_productivity)
        ],
        "reference_productivity": [
            float(value) for value in np.asarray(reference.post_event_productivity)
        ],
        "candidate_operator_counts": [
            int(value) for value in np.asarray(candidate.operator_counts)
        ],
        "reference_operator_counts": [
            int(value) for value in np.asarray(reference.operator_counts)
        ],
        "candidate_primary_score": float(np.asarray(candidate.primary_score)),
        "reference_primary_score": float(np.asarray(reference.primary_score)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program-path", type=Path, required=True)
    parser.add_argument("--reference-program-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--founder-index-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    initial = evaluate_exploratory.TASK_DIR / "initial_r4.py"
    candidate = evaluate._load_candidate(
        arguments.program_path, run_spec.R4_CONTRACT, initial
    )
    reference = evaluate._load_candidate(
        arguments.reference_program_path, run_spec.R4_CONTRACT, initial
    )
    for module in (candidate, reference):
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
    ) = evaluate_exploratory._microcosmos_imports()
    raw = arguments.manifest_path.read_bytes()
    manifest = manifest_from_json_bytes(raw)
    if raw != canonical_manifest_bytes(manifest) + b"\n":
        raise ValueError("manifest is not canonical")
    paired = paired_evaluator(
        manifest,
        SimulatorConfig(),
        make_r4_offspring_policy(candidate.make_offspring),
        make_r4_offspring_policy(reference.make_offspring),
        regime="punctuated",
        founder_index_path=arguments.founder_index_path,
        numerical_repeats=1,
        capture_finalists=False,
    )
    record = {
        "schema_version": 1,
        "candidate_sha256": candidate._evo2_source_sha256,
        "reference_sha256": reference._evo2_source_sha256,
        "manifest_sha256": manifest_sha256(manifest),
        "founder_index_sha256": hashlib.sha256(
            arguments.founder_index_path.read_bytes()
        ).hexdigest(),
        "episodes": [
            _episode_record(world, episode, ancestor)
            for world, episode, ancestor in zip(
                manifest.worlds,
                paired.episodes,
                paired.ancestor_episodes,
                strict=True,
            )
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
