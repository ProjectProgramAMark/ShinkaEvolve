"""Re-evaluate completed Shinka candidates and freeze one development winner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any

import numpy as np

if __package__:
    from examples.evo2_ecosystem import (
        evaluate as task_evaluator,
        r4_selection,
        run_spec,
    )
else:  # direct ``python examples/.../freeze_finalist.py``
    import evaluate as task_evaluator
    import r4_selection
    import run_spec


TASK_DIR = Path(__file__).resolve().parent
MICROCOSMOS_ROOT = TASK_DIR.parents[2] / "microcosmos"
ANALYSIS_PATH = MICROCOSMOS_ROOT / "experiments" / "evo2_ecosystem" / "analysis.py"
BASELINE_PATH = MICROCOSMOS_ROOT / "experiments" / "evo2_ecosystem" / "run_baselines.py"
SEALED_PATH = MICROCOSMOS_ROOT / "experiments" / "evo2_sealed" / "final.json"
SEALED_HASH_PATH = SEALED_PATH.with_suffix(".sha256")
SEALED_WORKFLOW_PATH = SEALED_PATH.with_name("workflow.py")
HEREDITY_PATH = MICROCOSMOS_ROOT / "src" / "microcosmos" / "heredity.py"
R4_TOOL_ROOT = (
    MICROCOSMOS_ROOT / "experiments" / "evo2_ecosystem" / "heredity_adaptation_v4"
)
_GENERATION = re.compile(r"gen_(\d+)")


@dataclass(frozen=True)
class Candidate:
    """One complete, integrity-valid generation result."""

    generation: int
    source_path: Path
    source_sha256: str
    training_score: float
    training_metrics: dict[str, Any]


def _require_stored_run_spec(
    run_root: Path,
    spec_raw: bytes,
    spec_hash: str,
) -> Path:
    """Authenticate the canonical run specification stored with run artifacts."""
    stored_spec = run_root / "run_spec.json"
    stored_hash = run_root / "run_spec.sha256"
    if (
        not stored_spec.is_file()
        or stored_spec.read_bytes() != spec_raw
        or not stored_hash.is_file()
        or stored_hash.read_text(encoding="utf-8") != f"{spec_hash}  run_spec.json\n"
    ):
        raise RuntimeError("stored run specification does not match the canonical spec")
    return stored_spec


def _load_r5_baselines(spec: dict[str, Any]) -> ModuleType:
    """Load the source-bound Microcosmos r5 baseline module."""
    if run_spec.schema_version(spec) != 4:
        raise ValueError("r5 baselines require a schema-v4 run specification")
    expected_path = (run_spec.PROJECT_ROOT / run_spec.R5_BASELINE_SOURCE_PATH).resolve()
    if (
        not expected_path.is_file()
        or run_spec.sha256_file(expected_path) != spec["source_sha256"]["baseline"]
    ):
        raise RuntimeError("r5 baseline source does not match the run specification")
    root = str(MICROCOSMOS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("experiments.evo2_ecosystem.r5.baselines")
    module_path = Path(module.__file__).resolve()
    if (
        module_path != expected_path
        or run_spec.sha256_file(module_path) != spec["source_sha256"]["baseline"]
    ):
        raise RuntimeError("loaded r5 baseline module is not the bound source")
    for name in (
        "publish_structured_random_roster",
        "load_structured_random_roster",
    ):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"r5 baseline module lacks {name}")
    return module


def _load_bound_baselines(spec: dict[str, Any]) -> ModuleType:
    """Preserve R5 loading and route schema 5 to its exact bound source."""
    if run_spec.schema_version(spec) == 4:
        return _load_r5_baselines(spec)
    if run_spec.schema_version(spec) != 5:
        raise ValueError("bound baselines require schema 4 or 5")
    expected_path = run_spec.baseline_source_path(spec)
    root = str(MICROCOSMOS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("experiments.evo2_ecosystem.r5.baselines")
    module_path = Path(module.__file__).resolve()
    if (
        module_path != expected_path
        or run_spec.sha256_file(module_path) != spec["source_sha256"]["baseline"]
    ):
        raise RuntimeError("loaded schema-v5 baseline is not the bound source")
    return module


def _verify_structured_training_inputs(spec: dict[str, Any]) -> None:
    """Apply the same schema-v4 source, prerequisite, and holdout preflight."""
    if __package__:
        from examples.evo2_ecosystem import run_evo  # noqa: PLC0415
    else:
        import run_evo  # type: ignore[no-redef]  # noqa: PLC0415

    run_evo._require_holdouts_locked(spec)
    run_evo._verify_search_inputs(spec, "punctuated")


def discover_candidates(
    results_dir: Path,
    expected_private: dict[str, Any] | None = None,
    *,
    minimum_generation: int = 0,
) -> list[Candidate]:
    """Return unique correct candidates, highest training score first."""
    candidates: list[Candidate] = []
    for generation_dir in results_dir.glob("gen_*"):
        match = _GENERATION.fullmatch(generation_dir.name)
        source_path = generation_dir / "main.py"
        metrics_path = generation_dir / "results" / "metrics.json"
        correct_path = generation_dir / "results" / "correct.json"
        if match is None or not all(
            path.is_file() for path in (source_path, metrics_path, correct_path)
        ):
            continue
        try:
            correct = json.loads(correct_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if correct.get("correct") is not True:
            continue
        score = metrics.get("combined_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        score = float(score)
        if not math.isfinite(score):
            continue
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        recorded_hash = metrics.get("private", {}).get("candidate_sha256")
        if recorded_hash != source_hash:
            raise ValueError(f"candidate hash mismatch in {generation_dir.name}")
        if expected_private is not None and any(
            metrics.get("private", {}).get(key) != value
            for key, value in expected_private.items()
        ):
            raise ValueError(f"candidate protocol mismatch in {generation_dir.name}")
        generation = int(match.group(1))
        if generation < minimum_generation:
            continue
        candidates.append(
            Candidate(
                generation=generation,
                source_path=source_path,
                source_sha256=source_hash,
                training_score=score,
                training_metrics=metrics,
            )
        )

    candidates.sort(key=lambda item: item.generation)
    unique: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.source_sha256 not in seen:
            seen.add(candidate.source_sha256)
            unique.append(candidate)
    unique.sort(key=lambda item: (-item.training_score, item.generation))
    return unique


def _audit_arm(
    spec: dict[str, Any],
    regime: str,
    paths: run_spec.RunPaths,
) -> dict[str, Any]:
    """Require one completed, protocol-matched evaluation per budgeted generation."""
    expected_static = {
        "manifest_sha256": run_spec.manifest_hash(spec, f"training_{regime}"),
        "evaluator_source_sha256": spec["source_sha256"]["evaluator"],
        "simulator_source_sha256": spec["source_sha256"]["simulator"],
        "simulator_config_sha256": spec["simulator_config_sha256"],
    }
    if run_spec.schema_version(spec) >= 2:
        expected_static.update(
            {
                "run_spec_sha256": run_spec.sha256_bytes(
                    run_spec.canonical_json_bytes(spec)
                ),
                "run_spec_schema_version": run_spec.schema_version(spec),
                "candidate_output_width": run_spec.candidate_output_width(spec),
                "founder_index_sha256": spec["founder_index"]["sha256"],
            }
        )
        if run_spec.schema_version(spec) >= 3:
            expected_static["candidate_contract_version"] = (
                run_spec.candidate_contract_version(spec)
            )
            expected_static["initial_program_sha256"] = spec["initial_program"][
                "sha256"
            ]
    completed: list[int] = []
    correct_count = 0
    full_evaluation_count = 0
    invalid_proposal_count = 0
    for generation_dir in paths.arm_results.glob("gen_*"):
        match = _GENERATION.fullmatch(generation_dir.name)
        if match is None:
            continue
        source_path = generation_dir / "main.py"
        metrics_path = generation_dir / "results" / "metrics.json"
        correct_path = generation_dir / "results" / "correct.json"
        present = [path.is_file() for path in (source_path, metrics_path, correct_path)]
        if not any(present):
            continue
        if not all(present):
            raise RuntimeError(f"incomplete generation artifact: {generation_dir.name}")
        generation = int(match.group(1))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        correct = json.loads(correct_path.read_text(encoding="utf-8"))
        private = metrics.get("private", {})
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if private.get("candidate_sha256") != source_hash or any(
            private.get(key) != value for key, value in expected_static.items()
        ):
            raise RuntimeError(f"protocol mismatch in {generation_dir.name}")
        full_evaluation = private.get("full_evaluation_performed") is True
        if full_evaluation:
            repeat_scores = private.get("repeat_scores")
            selected_repeat = private.get("selected_repeat_index")
            if (
                private.get("numerical_repeats") != spec["numerical_repeats"]
                or not isinstance(repeat_scores, list)
                or len(repeat_scores) != spec["numerical_repeats"]
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in repeat_scores
                )
                or not isinstance(selected_repeat, int)
                or isinstance(selected_repeat, bool)
                or not 0 <= selected_repeat < len(repeat_scores)
                or float(metrics.get("combined_score", float("nan")))
                != float(repeat_scores[selected_repeat])
            ):
                raise RuntimeError(f"full evaluation mismatch in {generation_dir.name}")
            full_evaluation_count += 1
        else:
            if (
                private.get("full_evaluation_performed") is not False
                or private.get("numerical_repeats") != 0
                or private.get("repeat_scores") != []
                or private.get("selected_repeat_index") is not None
            ):
                raise RuntimeError(
                    f"invalid proposal audit mismatch in {generation_dir.name}"
                )
            invalid_proposal_count += 1
        if correct.get("correct") is True and not full_evaluation:
            raise RuntimeError(
                f"correct proposal lacks evaluation in {generation_dir.name}"
            )
        completed.append(generation)
        correct_count += correct.get("correct") is True

    expected_generations = list(range(spec["generations"]))
    if sorted(completed) != expected_generations:
        raise RuntimeError(
            f"{regime} arm has not completed the matched generation budget"
        )
    database = paths.arm_results / "programs.sqlite"
    if not database.is_file():
        raise RuntimeError(f"archive database is missing for {regime}")
    wal = Path(f"{database}-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError(f"archive database still has an active WAL for {regime}")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        program_count, database_correct_count = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(correct), 0) FROM programs"
        ).fetchone()
    finally:
        connection.close()
    command_records = [paths.arm_results / "launch.json"] + sorted(
        paths.arm_results.glob("resume_*.json")
    )
    if not command_records[0].is_file():
        raise RuntimeError(f"launch record is missing for {regime}")
    database_hash = run_spec.sha256_file(database)
    completion_path = paths.run_root / f"{regime}.complete.json"
    if not completion_path.is_file():
        raise RuntimeError(f"completion marker is missing for {regime}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected_completion = {
        "run_id": spec["run_id"],
        "regime": regime,
        "run_spec_sha256": run_spec.sha256_bytes(run_spec.canonical_json_bytes(spec)),
        "completed_evaluation_count": spec["generations"],
        "database_sha256": database_hash,
    }
    if any(completion.get(key) != value for key, value in expected_completion.items()):
        raise RuntimeError(f"completion marker does not match {regime} artifacts")
    if (
        run_spec.schema_version(spec) >= 3
        and completion.get("llm_generated_descendant_count") != spec["proposal_budget"]
    ):
        raise RuntimeError("completion marker has the wrong r4 proposal count")
    return {
        "generation_budget": spec["generations"],
        "llm_generated_descendant_count": (
            spec["proposal_budget"] if run_spec.schema_version(spec) >= 3 else None
        ),
        "completed_evaluation_count": len(completed),
        "correct_evaluation_count": correct_count,
        "full_evaluation_count": full_evaluation_count,
        "invalid_proposal_count": invalid_proposal_count,
        "integrity_failed_evaluation_count": full_evaluation_count - correct_count,
        "archive_program_count": int(program_count),
        "archive_correct_program_count": int(database_correct_count),
        "database_sha256": database_hash,
        "completion_marker_sha256": run_spec.sha256_file(completion_path),
        "command_record_sha256": {
            path.name: run_spec.sha256_file(path) for path in command_records
        },
    }


def audit_matched_run(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Audit both arms before either may inspect development worlds."""
    return {
        regime: _audit_arm(spec, regime, run_spec.paths_for(spec, regime))
        for regime in run_spec.REGIMES
    }


def _json_column(value: str | None, expected_type: type, field: str) -> Any:
    try:
        parsed = json.loads(value) if value else expected_type()
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {field} in archive database") from error
    if not isinstance(parsed, expected_type):
        raise ValueError(f"invalid {field} in archive database")
    return parsed


def export_archive_lineage(results_dir: Path, output: Path) -> dict[str, Any]:
    """Export the ignored SQLite archive as a compact, reviewable lineage."""
    database = results_dir / "programs.sqlite"
    if not database.is_file():
        raise ValueError(f"archive database is missing at {database}")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """SELECT id, code, parent_id, archive_inspiration_ids,
                      top_k_inspiration_ids, generation, code_diff,
                      combined_score, public_metrics, private_metrics,
                      complexity, correct, metadata, island_idx
               FROM programs ORDER BY generation, id"""
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError("archive database contains no programs")

    programs = []
    ids = {row["id"] for row in rows}
    for row in rows:
        if row["parent_id"] is not None and row["parent_id"] not in ids:
            raise ValueError("archive lineage references a missing parent")
        public = _json_column(row["public_metrics"], dict, "public_metrics")
        private = _json_column(row["private_metrics"], dict, "private_metrics")
        metadata = _json_column(row["metadata"], dict, "metadata")
        source_hash = hashlib.sha256(row["code"].encode("utf-8")).hexdigest()
        if bool(row["correct"]) and private.get("candidate_sha256") != source_hash:
            raise ValueError("archive source hash does not match evaluator record")
        programs.append(
            {
                "program_id": row["id"],
                "parent_id": row["parent_id"],
                "archive_inspiration_ids": _json_column(
                    row["archive_inspiration_ids"], list, "archive_inspiration_ids"
                ),
                "top_k_inspiration_ids": _json_column(
                    row["top_k_inspiration_ids"], list, "top_k_inspiration_ids"
                ),
                "generation": int(row["generation"]),
                "island": int(row["island_idx"]),
                "correct": bool(row["correct"]),
                "score": (
                    None
                    if row["combined_score"] is None
                    else float(row["combined_score"])
                ),
                "complexity": (
                    None if row["complexity"] is None else float(row["complexity"])
                ),
                "candidate_source_sha256": source_hash,
                "code_diff": row["code_diff"] or "",
                "public_metrics": public,
                "private_metrics": private,
                "patch": {
                    key: metadata.get(key)
                    for key in (
                        "patch_type",
                        "patch_name",
                        "patch_description",
                        "model_name",
                        "api_costs",
                        "diff_summary",
                    )
                },
            }
        )
    record = {
        "schema_version": 1,
        "program_count": len(programs),
        "correct_program_count": sum(item["correct"] for item in programs),
        "task_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                TASK_DIR / "initial.py",
                TASK_DIR / "evaluate.py",
                TASK_DIR / "run_evo.py",
            )
        },
        "programs": programs,
    }
    _write_atomic(output, _json_bytes(record))
    return record


def _dependencies() -> dict[str, Any]:
    """Load only the trusted development evaluation boundary."""
    import sys

    root = str(MICROCOSMOS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from experiments.evo2_ecosystem.episode import (  # noqa: PLC0415
        SimulatorConfig,
        evaluate_manifest,
        simulator_config_sha256,
        simulator_source_sha256,
    )
    from experiments.evo2_ecosystem.manifests import (  # noqa: PLC0415
        load_bound_manifest,
        load_development_manifest,
    )
    from experiments.evo2_ecosystem.founder_artifacts import (  # noqa: PLC0415
        founder_index_sha256,
        load_founder_index,
    )
    from experiments.evo2_ecosystem.protocol import manifest_sha256  # noqa: PLC0415
    from microcosmos.heredity import mutate_cppn  # noqa: PLC0415

    try:
        from microcosmos.heredity import make_r4_offspring_policy  # noqa: PLC0415
        from experiments.evo2_ecosystem.episode import (  # noqa: PLC0415
            evaluate_manifest_paired_delta,
        )
    except ImportError:
        make_r4_offspring_policy = None
        evaluate_manifest_paired_delta = None

    return {
        "SimulatorConfig": SimulatorConfig,
        "evaluate_manifest": evaluate_manifest,
        "load_bound_manifest": load_bound_manifest,
        "load_development_manifest": load_development_manifest,
        "manifest_sha256": manifest_sha256,
        "founder_index_sha256": founder_index_sha256,
        "load_founder_index": load_founder_index,
        "mutate_cppn": mutate_cppn,
        "make_r4_offspring_policy": make_r4_offspring_policy,
        "evaluate_manifest_paired_delta": evaluate_manifest_paired_delta,
        "simulator_config_sha256": simulator_config_sha256,
        "simulator_source_sha256": simulator_source_sha256,
    }


def _load_development_manifest(
    spec: dict[str, Any], dependencies: dict[str, Any]
) -> Any:
    if run_spec.schema_version(spec) == 1:
        return dependencies["load_development_manifest"]()
    return dependencies["load_bound_manifest"](
        run_spec.manifest_path(spec, "development"),
        expected_sha256=run_spec.manifest_hash(spec, "development"),
        expected_partition="development",
    )


def _verified_founder_index_path(
    spec: dict[str, Any], dependencies: dict[str, Any]
) -> Path | None:
    path = run_spec.founder_index_path(spec)
    if path is None:
        return None
    # Development evaluation verifies only development founders; sealed founder
    # artifacts remain unreadable until the final examination.
    index = dependencies["load_founder_index"](path, verify_artifacts=False)
    if dependencies["founder_index_sha256"](index) != spec["founder_index"]["sha256"]:
        raise RuntimeError("founder index does not match the run spec")
    return path


def _episode_record(world: Any, episode: Any) -> dict[str, Any]:
    event = episode.event_record
    record = {
        "scenario_id": world.scenario_id,
        "pair_id": world.pair_id,
        "scenario_family": world.scenario_family,
        "world_seed": world.world_seed,
        "event_kind": world.event_kind.value,
        "primary_score": float(np.asarray(episode.primary_score)),
        "post_event_productivity": [
            float(value) for value in np.asarray(episode.post_event_productivity)
        ],
        "survived": bool(np.asarray(episode.survived)),
        "final_alive": int(np.asarray(episode.final_alive)),
        "minimum_population": int(np.asarray(episode.minimum_population)),
        "maximum_generation": int(np.asarray(episode.maximum_generation)),
        "birth_count": int(np.asarray(episode.birth_count)),
        "natural_death_count": int(np.asarray(episode.natural_death_count)),
        "operator_counts": [
            int(value) for value in np.asarray(episode.operator_counts)
        ],
        "catastrophe_death_count": int(np.asarray(event.catastrophe_death_count)),
        "targeted_founder_lineage": int(np.asarray(event.targeted_founder_lineage)),
        "policy_violation_count": int(np.asarray(episode.policy_violation_count)),
        "integrity_valid": bool(np.asarray(episode.integrity_valid)),
    }
    for name in (
        "generation_gain",
        "pre_selection_probability_count",
        "post_selection_probability_count",
    ):
        if hasattr(episode, name):
            record[name] = int(np.asarray(getattr(episode, name)))
    for name in (
        "pre_selection_probability_sum",
        "post_selection_probability_sum",
        "operator_success_ema",
        "operator_usage_ema",
        "operator_evidence_ema",
    ):
        if hasattr(episode, name):
            values = np.asarray(getattr(episode, name), dtype=float)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid r4 episode field: {name}")
            record[name] = [float(value) for value in values]
    return record


def _development_record(
    regime: str,
    candidate: Candidate,
    manifest: Any,
    config: Any,
    evaluation: Any,
    dependencies: dict[str, Any],
    spec_hash: str,
) -> dict[str, Any]:
    repeat_scores = [float(value) for value in np.asarray(evaluation.repeat_scores)]
    selected_index = int(np.asarray(evaluation.selected_repeat_index))
    if (
        not repeat_scores
        or not 0 <= selected_index < len(repeat_scores)
        or not all(math.isfinite(value) for value in repeat_scores)
    ):
        raise ValueError("development repeat audit is invalid")
    if len(manifest.worlds) != len(evaluation.episodes):
        raise ValueError("development manifest and evaluation world counts differ")
    score = float(np.asarray(evaluation.candidate_score))
    if score != repeat_scores[selected_index]:
        raise ValueError("development score is not the selected coherent repeat")
    integrity_valid = bool(np.asarray(evaluation.integrity_valid))
    if not math.isfinite(score):
        raise ValueError("development score is non-finite")
    training_manifest_hash = candidate.training_metrics.get("private", {}).get(
        "manifest_sha256"
    )
    if not isinstance(training_manifest_hash, str):
        raise ValueError("training metrics do not identify their manifest")
    record = {
        "schema_version": 1,
        "regime": regime,
        "generation": candidate.generation,
        "candidate_source_sha256": candidate.source_sha256,
        "run_spec_sha256": spec_hash,
        "training_score": candidate.training_score,
        "training_metrics": candidate.training_metrics,
        "training_manifest_sha256": training_manifest_hash,
        "development_score": score,
        "development_integrity_valid": integrity_valid,
        "numerical_repeats": len(repeat_scores),
        "repeat_scores": repeat_scores,
        "selected_repeat_index": selected_index,
        "development_manifest_sha256": dependencies["manifest_sha256"](manifest),
        "simulator_config_sha256": dependencies["simulator_config_sha256"](config),
        "episodes": [
            _episode_record(world, episode)
            for world, episode in zip(
                manifest.worlds,
                evaluation.episodes,
                strict=True,
            )
        ],
    }
    observations = getattr(evaluation, "adaptive_observations", ())
    if observations:
        normalized = []
        for observation in observations:
            value = dict(observation)
            # Validation and bootstrap are centralized in r4_selection.
            normalized.append(value)
        record["adaptive_observations"] = normalized
    return record


def reevaluate_candidate(
    regime: str,
    candidate: Candidate,
    manifest: Any,
    config: Any,
    dependencies: dict[str, Any],
    *,
    numerical_repeats: int,
    spec_hash: str,
    candidate_output_width: int = 4,
    founder_index_path: Path | None = None,
    contract_version: str = run_spec.LEGACY_CONTRACT,
    initial_source_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and evaluate one candidate against the development worlds."""
    if contract_version == run_spec.LEGACY_CONTRACT:
        module: ModuleType = task_evaluator._load_candidate(candidate.source_path)
        if candidate_output_width == 4:
            task_evaluator._smoke_validate_candidate(module)
        else:
            task_evaluator._smoke_validate_candidate(module, candidate_output_width)
    else:
        module = task_evaluator._load_candidate(
            candidate.source_path,
            contract_version,
            initial_source_path,
        )
        task_evaluator._smoke_validate_candidate(
            module,
            candidate_output_width,
            contract_version=contract_version,
        )
    mutate = (
        dependencies["make_r4_offspring_policy"]
        if contract_version == run_spec.R4_CONTRACT
        else dependencies["mutate_cppn"]
    )
    if mutate is None:
        raise RuntimeError("Microcosmos r4 heredity contract is unavailable")
    policy = (
        task_evaluator._build_policy(module, mutate)
        if contract_version == run_spec.LEGACY_CONTRACT
        else task_evaluator._build_policy(
            module,
            mutate,
            contract_version=contract_version,
        )
    )
    evaluation_kwargs = {"numerical_repeats": numerical_repeats}
    if founder_index_path is not None:
        evaluation_kwargs["founder_index_path"] = founder_index_path
    if contract_version == run_spec.R4_CONTRACT:
        paired = dependencies["evaluate_manifest_paired_delta"]
        if paired is None or initial_source_path is None:
            raise RuntimeError("Microcosmos paired r4 evaluator is unavailable")
        ancestor = task_evaluator._load_candidate(
            initial_source_path,
            contract_version,
            initial_source_path,
        )
        ancestor_policy = task_evaluator._build_policy(
            ancestor,
            mutate,
            contract_version=contract_version,
        )
        evaluation = paired(
            manifest,
            config,
            policy,
            ancestor_policy,
            regime=regime,
            **evaluation_kwargs,
        )
    else:
        evaluation = dependencies["evaluate_manifest"](
            manifest, config, policy, **evaluation_kwargs
        )
    record = _development_record(
        regime,
        candidate,
        manifest,
        config,
        evaluation,
        dependencies,
        spec_hash,
    )
    if record["numerical_repeats"] != numerical_repeats:
        raise ValueError("development evaluator used the wrong numerical repeats")
    return record


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        + b"\n"
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to replace audit record at {path}")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_terminal_directory(
    destination: Path,
    source: bytes,
    metrics: dict[str, Any],
    correct: dict[str, Any],
) -> None:
    """Publish one complete structured-random slot without partial records."""
    expected = {
        "main.py": source,
        "results/metrics.json": _json_bytes(metrics),
        "results/correct.json": _json_bytes(correct),
    }
    if destination.exists():
        if all(
            (destination / relative).is_file()
            and (destination / relative).read_bytes() == payload
            for relative, payload in expected.items()
        ):
            return
        raise FileExistsError(
            f"refusing to replace structured-random terminal record at {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for relative, payload in expected.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _structured_expected_private(
    spec: dict[str, Any],
    spec_hash: str,
) -> dict[str, Any]:
    return {
        "manifest_sha256": run_spec.manifest_hash(spec, "training_punctuated"),
        "evaluator_source_sha256": spec["source_sha256"]["evaluator"],
        "simulator_source_sha256": spec["source_sha256"]["simulator"],
        "simulator_config_sha256": spec["simulator_config_sha256"],
        "run_spec_sha256": spec_hash,
        "run_spec_schema_version": run_spec.schema_version(spec),
        "candidate_output_width": run_spec.candidate_output_width(spec),
        "candidate_contract_version": run_spec.candidate_contract_version(spec),
        "initial_program_sha256": spec["initial_program"]["sha256"],
        "founder_index_sha256": spec["founder_index"]["sha256"],
        "full_evaluation_performed": True,
    }


def _structured_terminal_records(
    training: Path,
    roster: tuple[Any, ...],
    *,
    spec: dict[str, Any],
    spec_hash: str,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Authenticate every terminal training slot against its frozen source."""
    if require_complete and len(roster) != 50:
        raise RuntimeError("structured-random roster must contain exactly 50 sources")
    expected_private = _structured_expected_private(spec, spec_hash)
    records: list[dict[str, Any]] = []
    for generation, source_record in enumerate(roster):
        generation_dir = training / f"gen_{generation}"
        source_path = generation_dir / "main.py"
        metrics_path = generation_dir / "results" / "metrics.json"
        correct_path = generation_dir / "results" / "correct.json"
        if not all(
            path.is_file() for path in (source_path, metrics_path, correct_path)
        ):
            raise RuntimeError(f"structured-random slot {generation} is not terminal")
        source = source_path.read_bytes()
        if (
            source != source_record.source.encode("utf-8")
            or hashlib.sha256(source).hexdigest() != source_record.sha256
        ):
            raise RuntimeError(
                f"structured-random slot {generation} source does not match roster"
            )
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            correct = json.loads(correct_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"structured-random slot {generation} has invalid JSON"
            ) from error
        if not isinstance(metrics, dict) or not isinstance(correct, dict):
            raise RuntimeError(
                f"structured-random slot {generation} has invalid records"
            )
        private = metrics.get("private", {})
        if (
            not isinstance(private, dict)
            or private.get("candidate_sha256") != source_record.sha256
            or private.get("structured_random_candidate_id")
            != source_record.candidate_id
            or private.get("structured_random_generation") != generation
            or any(
                private.get(key) != value
                for key, value in expected_private.items()
                if key != "full_evaluation_performed"
            )
        ):
            raise RuntimeError(f"structured-random slot {generation} protocol mismatch")
        was_evaluated = private.get("full_evaluation_performed") is True
        is_correct = correct.get("correct") is True
        if is_correct != (private.get("integrity_valid") is True) or (
            is_correct and not was_evaluated
        ):
            raise RuntimeError(
                f"structured-random slot {generation} correctness mismatch"
            )
        score = metrics.get("combined_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise RuntimeError(f"structured-random slot {generation} has invalid score")
        if was_evaluated:
            repeat_scores = private.get("repeat_scores")
            selected_repeat = private.get("selected_repeat_index")
            if (
                private.get("numerical_repeats") != spec["numerical_repeats"]
                or not isinstance(repeat_scores, list)
                or len(repeat_scores) != spec["numerical_repeats"]
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in repeat_scores
                )
                or isinstance(selected_repeat, bool)
                or not isinstance(selected_repeat, int)
                or not 0 <= selected_repeat < len(repeat_scores)
                or float(score) != float(repeat_scores[selected_repeat])
            ):
                raise RuntimeError(
                    f"structured-random slot {generation} full audit mismatch"
                )
        elif (
            private.get("numerical_repeats") != 0
            or private.get("repeat_scores") != []
            or private.get("selected_repeat_index") is not None
        ):
            raise RuntimeError(
                f"structured-random slot {generation} invalid audit mismatch"
            )
        records.append(
            {
                "candidate_id": source_record.candidate_id,
                "family": source_record.family,
                "generation": generation,
                "source_sha256": source_record.sha256,
                "metrics_sha256": run_spec.sha256_file(metrics_path),
                "correct_sha256": run_spec.sha256_file(correct_path),
                "correct": is_correct,
                "full_evaluation_performed": was_evaluated,
                "training_score": float(score),
            }
        )
    if require_complete:
        unexpected = {
            path.name
            for path in training.glob("gen_*")
            if path.name not in {f"gen_{index}" for index in range(50)}
        }
        if unexpected:
            raise RuntimeError(
                f"unexpected structured-random training slots: {sorted(unexpected)}"
            )
    return records


def _structured_completion(
    paths: run_spec.StructuredRandomPaths,
    roster: tuple[Any, ...],
    *,
    spec: dict[str, Any],
    spec_hash: str,
) -> dict[str, Any]:
    index_path = paths.roster / "index.json"
    records = _structured_terminal_records(
        paths.training,
        roster,
        spec=spec,
        spec_hash=spec_hash,
    )
    unique_valid = {record["source_sha256"] for record in records if record["correct"]}
    return {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "control": "structured_random",
        "regime": "punctuated",
        "run_spec_sha256": spec_hash,
        "roster_index_sha256": run_spec.sha256_file(index_path),
        "terminal_evaluation_count": len(records),
        "full_evaluation_count": sum(
            record["full_evaluation_performed"] for record in records
        ),
        "valid_evaluation_count": sum(record["correct"] for record in records),
        "unique_valid_candidate_count": len(unique_valid),
        "passed": len(unique_valid) >= spec["top_k"],
        "terminal_records": records,
    }


def _evaluation_timeout_seconds(spec: dict[str, Any]) -> int:
    """Parse the already-frozen Shinka wall-clock timeout for one control slot."""
    value = spec["search"]["evaluation_timeout"]
    try:
        hours, minutes, seconds = (int(part) for part in value.split(":"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("invalid structured-random evaluation timeout") from error
    timeout = hours * 3600 + minutes * 60 + seconds
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60 or timeout <= 0:
        raise ValueError("invalid structured-random evaluation timeout")
    return timeout


def _evaluate_structured_source(
    source_path: Path,
    *,
    stored_spec: Path,
    spec: dict[str, Any],
    spec_hash: str,
) -> tuple[dict[str, Any], bool, str | None]:
    """Run one source through the existing evaluator in a fresh JAX process."""
    with tempfile.TemporaryDirectory(prefix="evo2-r5-structured-") as directory:
        results = Path(directory) / "results"
        command = [
            sys.executable,
            str(TASK_DIR / "evaluate.py"),
            "--program_path",
            str(source_path),
            "--results_dir",
            str(results),
            "--training_regime",
            "punctuated",
            "--run_spec_path",
            str(stored_spec),
            "--run_spec_sha256",
            spec_hash,
        ]
        error_code: str | None = None
        error: str | None = None
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=_evaluation_timeout_seconds(spec),
            )
            if completed.returncode:
                error_code = "evaluation_failed"
                error = "candidate evaluator process failed"
        except subprocess.TimeoutExpired:
            error_code = "timeout"
            error = "candidate evaluation timed out"

        metrics_path = results / "metrics.json"
        correct_path = results / "correct.json"
        if error_code is None and metrics_path.is_file() and correct_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                correct = json.loads(correct_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                error_code = "evaluation_failed"
                error = "candidate evaluator emitted invalid records"
            else:
                if isinstance(metrics, dict) and isinstance(correct, dict):
                    return metrics, correct.get("correct") is True, correct.get("error")
                error_code = "evaluation_failed"
                error = "candidate evaluator emitted invalid records"
        elif error_code is None:
            error_code = "evaluation_failed"
            error = "candidate evaluator emitted incomplete records"

        assert error_code is not None
        metrics = task_evaluator._failure_metrics(
            error_code,
            source_path,
            "punctuated",
            run_spec_path=stored_spec,
            expected_run_spec_sha256=spec_hash,
        )
        return metrics, False, error


def run_structured_random_training(
    *,
    spec: dict[str, Any],
    spec_hash: str,
    spec_raw: bytes,
) -> dict[str, Any]:
    """Evaluate the prospectively frozen 50-source control exactly once each."""
    if run_spec.schema_version(spec) not in {4, 5} or spec["top_k"] != 5:
        raise ValueError("structured-random training requires schema 4 or 5")
    paths = run_spec.structured_random_paths(spec)
    stored_spec = _require_stored_run_spec(paths.run_root, spec_raw, spec_hash)
    _verify_structured_training_inputs(spec)
    baselines = _load_bound_baselines(spec)
    baselines.publish_structured_random_roster(paths.roster)
    roster = tuple(baselines.load_structured_random_roster(paths.roster))
    if len(roster) != 50:
        raise RuntimeError("structured-random roster has the wrong frozen size")

    for generation, source_record in enumerate(roster):
        destination = paths.training / f"gen_{generation}"
        if destination.exists():
            # Resume accepts only a complete, authenticated terminal directory.
            _structured_terminal_records(
                paths.training,
                roster[: generation + 1],
                spec=spec,
                spec_hash=spec_hash,
                require_complete=False,
            )
            continue
        metrics, correct_value, error = _evaluate_structured_source(
            paths.roster / f"{source_record.candidate_id}.py",
            stored_spec=stored_spec,
            spec=spec,
            spec_hash=spec_hash,
        )
        metrics = dict(metrics)
        private = dict(metrics.get("private", {}))
        private.update(
            {
                "structured_random_candidate_id": source_record.candidate_id,
                "structured_random_generation": generation,
            }
        )
        metrics["private"] = private
        _write_terminal_directory(
            destination,
            source_record.source.encode("utf-8"),
            metrics,
            {"correct": correct_value, "error": error},
        )

    completion = _structured_completion(
        paths,
        roster,
        spec=spec,
        spec_hash=spec_hash,
    )
    completion_path = paths.control_root / "complete.json"
    _write_atomic(completion_path, _json_bytes(completion))
    if not completion["passed"]:
        failure = {
            "schema_version": 1,
            "run_id": spec["run_id"],
            "control": "structured_random",
            "reason": "fewer_than_five_unique_valid_candidates",
            "run_spec_sha256": spec_hash,
            "completion_sha256": run_spec.sha256_file(completion_path),
            "unique_valid_candidate_count": completion["unique_valid_candidate_count"],
            "passed": False,
        }
        _write_atomic(paths.control_root / "search_failure.json", _json_bytes(failure))
    return completion


def _load_development_record(
    path: Path,
    *,
    regime: str,
    candidate: Candidate,
    development_manifest_sha256: str,
    simulator_config_sha256: str,
    numerical_repeats: int,
    spec_hash: str,
) -> dict[str, Any] | None:
    """Resume only an exact immutable evaluation of the same source/protocol."""
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid existing development record at {path}") from error
    expected = {
        "regime": regime,
        "generation": candidate.generation,
        "candidate_source_sha256": candidate.source_sha256,
        "development_manifest_sha256": development_manifest_sha256,
        "simulator_config_sha256": simulator_config_sha256,
        "numerical_repeats": numerical_repeats,
        "run_spec_sha256": spec_hash,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"stale existing development record at {path}")
    score = record.get("development_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"invalid existing development score at {path}")
    if not math.isfinite(float(score)):
        raise ValueError(f"non-finite existing development score at {path}")
    return record


def _provenance(spec: dict[str, Any], dependencies: dict[str, Any]) -> dict[str, Any]:
    analysis_path = ANALYSIS_PATH
    baseline_path = BASELINE_PATH
    if run_spec.schema_version(spec) in {4, 5}:
        analysis_path = run_spec.protocol_tool_paths(spec)["final_analysis"]
        baseline_path = run_spec.baseline_source_path(spec)
    all_sources = {
        "initial": run_spec.sha256_file(run_spec.initial_program_path(spec)),
        "evaluator": run_spec.sha256_file(TASK_DIR / "evaluate.py"),
        "simulator": dependencies["simulator_source_sha256"](),
        "analysis": run_spec.sha256_file(analysis_path),
        "baseline": run_spec.sha256_file(baseline_path),
        "dependency_lock": run_spec.sha256_file(run_spec.DEPENDENCY_LOCK_PATH),
        "launcher": run_spec.sha256_file(TASK_DIR / "run_evo.py"),
        "finalist_selector": run_spec.sha256_file(Path(__file__)),
        "lineage_selector": run_spec.sha256_file(TASK_DIR / "program_lineage.py"),
        "run_spec_module": run_spec.sha256_file(TASK_DIR / "run_spec.py"),
        "adaptive_selector": run_spec.sha256_file(TASK_DIR / "r4_selection.py"),
    }
    if run_spec.schema_version(spec) not in {4, 5}:
        all_sources.update(
            {
                "preregistration": run_spec.sha256_file(run_spec.PREREGISTRATION_PATH),
                "r4_final_analysis": run_spec.sha256_file(R4_TOOL_ROOT / "analysis.py"),
                "r4_manifest_generator": run_spec.sha256_file(
                    R4_TOOL_ROOT / "manifest_generator.py"
                ),
                "r4_qualification": run_spec.sha256_file(
                    R4_TOOL_ROOT / "qualification.py"
                ),
                "r4_opportunity": run_spec.sha256_file(R4_TOOL_ROOT / "opportunity.py"),
            }
        )
    actual_sources = {name: all_sources[name] for name in spec["source_sha256"]}
    if actual_sources != spec["source_sha256"]:
        raise RuntimeError("trusted source no longer matches the production run spec")
    sealed_path = run_spec.manifest_path(spec, "sealed")
    sealed_sidecar_path = sealed_path.with_suffix(".sha256")
    sealed_sidecar = sealed_sidecar_path.read_text(encoding="utf-8")
    expected_sidecar = f"{run_spec.manifest_hash(spec, 'sealed')}  {sealed_path.name}\n"
    if sealed_sidecar != expected_sidecar:
        raise RuntimeError("sealed manifest sidecar does not match the run spec")
    if os.access(sealed_path, os.R_OK):
        raise RuntimeError(
            "sealed manifest must remain unreadable during finalist selection"
        )
    if run_spec.schema_version(spec) >= 2:
        founder_path = run_spec.founder_index_path(spec)
        assert founder_path is not None
        founder_index = dependencies["load_founder_index"](
            founder_path,
            verify_artifacts=False,
        )
        if (
            dependencies["founder_index_sha256"](founder_index)
            != spec["founder_index"]["sha256"]
        ):
            raise RuntimeError("founder index does not match the run spec")
        bank_root = founder_path.parent.resolve()
        for record in founder_index.founders:
            if record.partition != "sealed":
                continue
            artifact = (bank_root / record.artifact).resolve()
            if bank_root not in artifact.parents:
                raise RuntimeError("sealed founder path escapes the founder bank")
            if not artifact.is_file():
                raise RuntimeError(f"required sealed founder is missing: {artifact}")
            if os.access(artifact, os.R_OK):
                raise RuntimeError(
                    "sealed founder must remain unreadable during finalist "
                    f"selection: {artifact}"
                )
    protocol_hashes = {
        name: run_spec.sha256_file(path)
        for name, path in run_spec.bound_protocol_paths(spec).items()
    }
    if run_spec.schema_version(spec) >= 2:
        for name, digest in protocol_hashes.items():
            if digest != spec[name]["sha256"]:
                raise RuntimeError(f"{name} no longer matches the run spec")
    provenance = {**actual_sources, **protocol_hashes}
    if run_spec.schema_version(spec) == 1:
        provenance.update(
            {
                "sealed_workflow": run_spec.sha256_file(SEALED_WORKFLOW_PATH),
                "heredity": run_spec.sha256_file(HEREDITY_PATH),
                "launcher": run_spec.sha256_file(TASK_DIR / "run_evo.py"),
                "finalist_selector": run_spec.sha256_file(Path(__file__)),
                "run_spec_module": run_spec.sha256_file(TASK_DIR / "run_spec.py"),
            }
        )
    return provenance


def _freeze(
    frozen_dir: Path,
    winner: Candidate,
    freeze_record: dict[str, Any],
    archive_lineage: bytes,
) -> None:
    source = winner.source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != winner.source_sha256:
        raise ValueError("winning candidate changed after discovery")
    record = _json_bytes(freeze_record)
    if frozen_dir.exists():
        if (
            (frozen_dir / "main.py").read_bytes() == source
            and (frozen_dir / "freeze_record.json").read_bytes() == record
            and (frozen_dir / "archive_lineage.json").read_bytes() == archive_lineage
        ):
            return
        raise FileExistsError(f"refusing to replace frozen finalist at {frozen_dir}")

    frozen_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{frozen_dir.name}.", dir=frozen_dir.parent)
    )
    try:
        (staging / "main.py").write_bytes(source)
        (staging / "freeze_record.json").write_bytes(record)
        (staging / "archive_lineage.json").write_bytes(archive_lineage)
        os.replace(staging, frozen_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _protocol_freeze_fields(
    spec: dict[str, Any],
    source_hashes: dict[str, Any],
) -> dict[str, Any]:
    """Preserve legacy records while using r5's single protocol document."""
    version = run_spec.schema_version(spec)
    if version in {4, 5}:
        return {
            "protocol_document_sha256": spec["protocol_document"]["sha256"],
            "protocol_document_complete": True,
        }
    return {
        "preregistration_sha256": (
            source_hashes["preregistration"]
            if version == 1
            else spec["preregistration"]["sha256"]
        ),
        "implementation_plan_sha256": (
            None if version == 1 else spec["implementation_plan"]["sha256"]
        ),
        "preregistration_complete": True,
    }


_DEVELOPMENT_COMPLETE_NAME = "development.complete.json"


def _development_complete_path(spec: dict[str, Any], label: str) -> Path:
    if label in run_spec.REGIMES:
        return run_spec.paths_for(spec, label).selection / _DEVELOPMENT_COMPLETE_NAME
    if label == "structured_random":
        return (
            run_spec.structured_random_paths(spec).selection
            / _DEVELOPMENT_COMPLETE_NAME
        )
    raise ValueError(f"unknown development label: {label}")


def _publish_development_completion(
    spec: dict[str, Any],
    spec_hash: str,
    label: str,
    candidates: list[Candidate],
    selection_directory: Path,
) -> dict[str, Any]:
    records = []
    for candidate in candidates:
        filename = f"gen_{candidate.generation}_{candidate.source_sha256[:12]}.json"
        path = selection_directory / filename
        if not path.is_file():
            raise RuntimeError(f"missing development evaluation: {path}")
        records.append(
            {
                "generation": candidate.generation,
                "candidate_source_sha256": candidate.source_sha256,
                "record": filename,
                "record_sha256": run_spec.sha256_file(path),
            }
        )
    archive = selection_directory / "archive_lineage.json"
    if not archive.is_file():
        raise RuntimeError("development archive lineage is missing")
    completion = {
        "schema_version": 1,
        "complete": True,
        "run_id": spec["run_id"],
        "label": label,
        "run_spec_sha256": spec_hash,
        "top_k": spec["top_k"],
        "archive_lineage_sha256": run_spec.sha256_file(archive),
        "records": records,
    }
    _write_atomic(
        selection_directory / _DEVELOPMENT_COMPLETE_NAME,
        _json_bytes(completion),
    )
    return completion


def _load_development_completion(
    spec: dict[str, Any],
    spec_hash: str,
    label: str,
) -> dict[str, Any]:
    path = _development_complete_path(spec, label)
    try:
        completion = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"development evaluation is incomplete for {label}"
        ) from error
    expected = {
        "schema_version": 1,
        "complete": True,
        "run_id": spec["run_id"],
        "label": label,
        "run_spec_sha256": spec_hash,
        "top_k": spec["top_k"],
    }
    if not isinstance(completion, dict) or any(
        completion.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError(f"development completion marker is invalid for {label}")
    selection = path.parent
    archive = selection / "archive_lineage.json"
    records = completion.get("records")
    if (
        not archive.is_file()
        or run_spec.sha256_file(archive) != completion.get("archive_lineage_sha256")
        or not isinstance(records, list)
        or len(records) != spec["top_k"]
    ):
        raise RuntimeError(f"development artifacts do not authenticate for {label}")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "generation",
            "candidate_source_sha256",
            "record",
            "record_sha256",
        }:
            raise RuntimeError(f"development record index is invalid for {label}")
        record_path = selection / record["record"]
        source_hash = record["candidate_source_sha256"]
        if (
            Path(record["record"]).name != record["record"]
            or source_hash in seen
            or not record_path.is_file()
            or run_spec.sha256_file(record_path) != record["record_sha256"]
        ):
            raise RuntimeError(f"development record does not authenticate for {label}")
        seen.add(source_hash)
    if path.read_bytes() != _json_bytes(completion):
        raise RuntimeError(f"development completion is not canonical for {label}")
    return completion


def _require_all_development_evaluations(spec: dict[str, Any], spec_hash: str) -> None:
    """Prevent any r5 freeze until all 15 development records authenticate."""
    for label in (*run_spec.REGIMES, "structured_random"):
        _load_development_completion(spec, spec_hash, label)


def select_and_freeze(
    regime: str,
    *,
    spec: dict[str, Any],
    spec_hash: str,
    spec_raw: bytes,
    _phase: str | None = None,
) -> dict[str, Any]:
    """Sequentially evaluate the fixed top-K and freeze the development winner."""
    version = run_spec.schema_version(spec)
    if _phase not in {None, "evaluate", "freeze"}:
        raise ValueError("unknown development phase")
    if version in {4, 5} and _phase is None:
        raise RuntimeError("bounded development requires an explicit phase")
    if version not in {4, 5} and _phase is not None:
        raise ValueError("split development phases require schema 4 or 5")
    if _phase == "freeze":
        _require_all_development_evaluations(spec, spec_hash)
    paths = run_spec.paths_for(spec, regime)
    _require_stored_run_spec(paths.run_root, spec_raw, spec_hash)
    matched_counts = audit_matched_run(spec)
    dependencies = _dependencies()
    source_hashes = _provenance(spec, dependencies)
    expected_private = {
        "manifest_sha256": run_spec.manifest_hash(spec, f"training_{regime}"),
        "evaluator_source_sha256": spec["source_sha256"]["evaluator"],
        "simulator_source_sha256": spec["source_sha256"]["simulator"],
        "simulator_config_sha256": spec["simulator_config_sha256"],
        "numerical_repeats": spec["numerical_repeats"],
        "full_evaluation_performed": True,
    }
    if run_spec.schema_version(spec) >= 2:
        expected_private.update(
            {
                "run_spec_sha256": spec_hash,
                "run_spec_schema_version": run_spec.schema_version(spec),
                "candidate_output_width": run_spec.candidate_output_width(spec),
                "founder_index_sha256": spec["founder_index"]["sha256"],
            }
        )
        if run_spec.schema_version(spec) >= 3:
            expected_private["candidate_contract_version"] = (
                run_spec.candidate_contract_version(spec)
            )
            expected_private["initial_program_sha256"] = spec["initial_program"][
                "sha256"
            ]
    candidates = discover_candidates(
        paths.arm_results,
        expected_private,
        minimum_generation=1 if version == 5 else 0,
    )[: spec["top_k"]]
    if len(candidates) != spec["top_k"]:
        raise RuntimeError("arm does not contain the required unique correct top-K")
    archive_record = export_archive_lineage(
        paths.arm_results,
        paths.selection / "archive_lineage.json",
    )
    manifest = _load_development_manifest(spec, dependencies)
    if manifest.partition != "development":
        raise RuntimeError("development loader returned the wrong partition")
    config = dependencies["SimulatorConfig"]()
    development_manifest_hash = dependencies["manifest_sha256"](manifest)
    simulator_config_hash = dependencies["simulator_config_sha256"](config)
    if development_manifest_hash != run_spec.manifest_hash(spec, "development"):
        raise RuntimeError("development manifest does not match the run spec")
    if simulator_config_hash != spec["simulator_config_sha256"]:
        raise RuntimeError("simulator configuration does not match the run spec")
    founder_index_path = _verified_founder_index_path(spec, dependencies)

    evaluations: list[dict[str, Any]] = []
    candidate_by_hash = {candidate.source_sha256: candidate for candidate in candidates}
    for candidate in candidates:
        filename = f"gen_{candidate.generation}_{candidate.source_sha256[:12]}.json"
        path = paths.selection / filename
        record = _load_development_record(
            path,
            regime=regime,
            candidate=candidate,
            development_manifest_sha256=development_manifest_hash,
            simulator_config_sha256=simulator_config_hash,
            numerical_repeats=spec["numerical_repeats"],
            spec_hash=spec_hash,
        )
        if record is None:
            if _phase == "freeze":
                raise RuntimeError(
                    f"freeze-only cannot fill missing development record: {path}"
                )
            record = reevaluate_candidate(
                regime,
                candidate,
                manifest,
                config,
                dependencies,
                numerical_repeats=spec["numerical_repeats"],
                spec_hash=spec_hash,
                candidate_output_width=run_spec.candidate_output_width(spec),
                founder_index_path=founder_index_path,
                contract_version=run_spec.candidate_contract_version(spec),
                initial_source_path=run_spec.initial_program_path(spec),
            )
            if version in {4, 5}:
                try:
                    eligibility = r4_selection.classify_adaptive(
                        record.get("adaptive_observations", ()),
                        bootstrap_replicates=spec["bootstrap"]["replicates"],
                        bootstrap_seed=spec["bootstrap"]["seed"],
                    )
                    record["adaptive_eligibility"] = asdict(eligibility)
                except ValueError as error:
                    record["adaptive_eligibility"] = {
                        "eligible": False,
                        "reason": str(error),
                    }
            _write_atomic(path, _json_bytes(record))
        elif version in {4, 5} and "adaptive_eligibility" not in record:
            raise RuntimeError(
                f"r5 development record lacks frozen adaptive eligibility: {path}"
            )
        evaluations.append(record)

    if not any(item["development_integrity_valid"] for item in evaluations):
        raise RuntimeError("no development candidate passed integrity checks")
    if version >= 3:
        for item in evaluations:
            if "adaptive_eligibility" in item:
                continue
            try:
                eligibility = r4_selection.classify_adaptive(
                    item.get("adaptive_observations", ()),
                    bootstrap_replicates=spec["bootstrap"]["replicates"],
                    bootstrap_seed=spec["bootstrap"]["seed"],
                )
                item["adaptive_eligibility"] = asdict(eligibility)
            except ValueError as error:
                item["adaptive_eligibility"] = {
                    "eligible": False,
                    "reason": str(error),
                }
    evaluations.sort(
        key=lambda item: (
            not item["development_integrity_valid"],
            -item["development_score"],
            item["generation"],
        )
    )
    if _phase == "evaluate":
        return _publish_development_completion(
            spec,
            spec_hash,
            regime,
            candidates,
            paths.selection,
        )
    winner_record = evaluations[0]
    winner = candidate_by_hash[winner_record["candidate_source_sha256"]]
    archive_path = paths.selection / "archive_lineage.json"
    archive_bytes = archive_path.read_bytes()
    selection_artifacts = {
        path.name: run_spec.sha256_file(path)
        for path in sorted(paths.selection.glob("*.json"))
    }
    freeze_record = {
        "schema_version": 2,
        "run_id": spec["run_id"],
        "regime": regime,
        "run_spec_sha256": spec_hash,
        "run_spec": spec,
        "selection_rule": {
            "name": "development_score_desc_then_generation_asc",
            "training_rank": "descending combined_score, then generation",
            "top_k": spec["top_k"],
            "development_rank": "integrity, median repeated score, then generation",
            "numerical_repeats": spec["numerical_repeats"],
        },
        "selected_generation": winner.generation,
        "candidate_source_sha256": winner.source_sha256,
        "training_score": winner.training_score,
        "training_manifest_sha256": winner_record["training_manifest_sha256"],
        "development_score": winner_record["development_score"],
        "development_integrity_valid": winner_record["development_integrity_valid"],
        "numerical_repeats": winner_record["numerical_repeats"],
        "repeat_scores": winner_record["repeat_scores"],
        "selected_repeat_index": winner_record["selected_repeat_index"],
        "development_manifest_sha256": winner_record["development_manifest_sha256"],
        "simulator_config_sha256": winner_record["simulator_config_sha256"],
        "simulator_config": asdict(config),
        "sealed_manifest_sha256": run_spec.manifest_hash(spec, "sealed"),
        "sealed_manifest_locked": True,
        "source_sha256": source_hashes,
        "founder_index_sha256": (
            None
            if run_spec.schema_version(spec) == 1
            else spec["founder_index"]["sha256"]
        ),
        "candidate_output_width": run_spec.candidate_output_width(spec),
        "matched_run_counts": matched_counts,
        "archive_config": spec["archive"],
        "baselines": spec["baselines"],
        "bootstrap_config": spec["bootstrap"],
        "database_sha256": matched_counts[regime]["database_sha256"],
        "archive_lineage_record": "archive_lineage.json",
        "archive_lineage_sha256": run_spec.sha256_bytes(archive_bytes),
        "archive_program_count": archive_record["program_count"],
        "archive_correct_program_count": archive_record["correct_program_count"],
        "selection_artifact_sha256": selection_artifacts,
        "development_evaluation_record": (
            f"gen_{winner.generation}_{winner.source_sha256[:12]}.json"
        ),
        "ranked_development_candidates": [
            {
                "generation": item["generation"],
                "candidate_source_sha256": item["candidate_source_sha256"],
                "training_score": item["training_score"],
                "development_score": item["development_score"],
                "development_integrity_valid": item["development_integrity_valid"],
                "numerical_repeats": item["numerical_repeats"],
                "repeat_scores": item["repeat_scores"],
                "selected_repeat_index": item["selected_repeat_index"],
                "adaptive_eligibility": item.get("adaptive_eligibility"),
            }
            for item in evaluations
        ],
    }
    freeze_record.update(_protocol_freeze_fields(spec, source_hashes))
    if version < 3:
        _freeze(paths.frozen, winner, freeze_record, archive_bytes)
        return freeze_record

    freeze_record["finalist_type"] = "unrestricted"
    freeze_record["adaptive_eligibility"] = winner_record.get("adaptive_eligibility")
    _freeze(paths.frozen / "unrestricted", winner, freeze_record, archive_bytes)
    finalists = r4_selection.select_finalists(evaluations)
    result: dict[str, Any] = {"unrestricted": freeze_record}
    adaptive = finalists.get("adaptive")
    if adaptive is not None:
        adaptive_winner = candidate_by_hash[adaptive["candidate_source_sha256"]]
        adaptive_record = {
            **freeze_record,
            "finalist_type": "adaptive",
            "selected_generation": adaptive_winner.generation,
            "candidate_source_sha256": adaptive_winner.source_sha256,
            "training_score": adaptive_winner.training_score,
            "training_manifest_sha256": adaptive["training_manifest_sha256"],
            "development_score": adaptive["development_score"],
            "development_integrity_valid": adaptive["development_integrity_valid"],
            "repeat_scores": adaptive["repeat_scores"],
            "selected_repeat_index": adaptive["selected_repeat_index"],
            "adaptive_eligibility": adaptive["adaptive_eligibility"],
            "development_evaluation_record": (
                f"gen_{adaptive_winner.generation}_"
                f"{adaptive_winner.source_sha256[:12]}.json"
            ),
        }
        _freeze(
            paths.frozen / "adaptive",
            adaptive_winner,
            adaptive_record,
            archive_bytes,
        )
        result["adaptive"] = adaptive_record
    return result


def evaluate_development_candidates(
    regime: str,
    *,
    spec: dict[str, Any],
    spec_hash: str,
    spec_raw: bytes,
) -> dict[str, Any]:
    """Write and authenticate one arm's five r5 development evaluations only."""
    return select_and_freeze(
        regime,
        spec=spec,
        spec_hash=spec_hash,
        spec_raw=spec_raw,
        _phase="evaluate",
    )


def freeze_development_finalist(
    regime: str,
    *,
    spec: dict[str, Any],
    spec_hash: str,
    spec_raw: bytes,
) -> dict[str, Any]:
    """Freeze from complete existing r5 records without running evaluation."""
    return select_and_freeze(
        regime,
        spec=spec,
        spec_hash=spec_hash,
        spec_raw=spec_raw,
        _phase="freeze",
    )


def select_and_freeze_structured_random(
    *,
    spec: dict[str, Any],
    spec_hash: str,
    spec_raw: bytes,
    _phase: str,
) -> dict[str, Any]:
    """Development-rank the fixed control top five and freeze one champion."""
    if run_spec.schema_version(spec) not in {4, 5} or spec["top_k"] != 5:
        raise ValueError("structured-random selection requires schema 4 or 5")
    if _phase not in {"evaluate", "freeze"}:
        raise ValueError("structured-random development requires an explicit phase")
    if _phase == "freeze":
        _require_all_development_evaluations(spec, spec_hash)
    paths = run_spec.structured_random_paths(spec)
    _require_stored_run_spec(paths.run_root, spec_raw, spec_hash)
    baselines = _load_bound_baselines(spec)
    roster = tuple(baselines.load_structured_random_roster(paths.roster))
    completion = _structured_completion(
        paths,
        roster,
        spec=spec,
        spec_hash=spec_hash,
    )
    completion_path = paths.control_root / "complete.json"
    if not completion_path.is_file() or completion_path.read_bytes() != _json_bytes(
        completion
    ):
        raise RuntimeError("structured-random completion marker does not authenticate")
    if not completion["passed"]:
        raise RuntimeError("structured-random search has fewer than five valid sources")

    matched_counts = audit_matched_run(spec)
    dependencies = _dependencies()
    source_hashes = _provenance(spec, dependencies)
    candidates = discover_candidates(
        paths.training,
        _structured_expected_private(spec, spec_hash),
    )[: spec["top_k"]]
    if len(candidates) != spec["top_k"]:
        raise RuntimeError(
            "structured-random control lacks five unique valid training candidates"
        )

    archive_record = {
        "schema_version": 1,
        "control": "structured_random",
        "run_spec_sha256": spec_hash,
        "roster_index_sha256": completion["roster_index_sha256"],
        "terminal_evaluation_count": completion["terminal_evaluation_count"],
        "records": completion["terminal_records"],
    }
    archive_path = paths.selection / "archive_lineage.json"
    _write_atomic(archive_path, _json_bytes(archive_record))

    manifest = _load_development_manifest(spec, dependencies)
    if manifest.partition != "development":
        raise RuntimeError("development loader returned the wrong partition")
    config = dependencies["SimulatorConfig"]()
    development_manifest_hash = dependencies["manifest_sha256"](manifest)
    simulator_config_hash = dependencies["simulator_config_sha256"](config)
    if development_manifest_hash != run_spec.manifest_hash(spec, "development"):
        raise RuntimeError("development manifest does not match the run spec")
    if simulator_config_hash != spec["simulator_config_sha256"]:
        raise RuntimeError("simulator configuration does not match the run spec")
    founder_index_path = _verified_founder_index_path(spec, dependencies)

    evaluations: list[dict[str, Any]] = []
    candidate_by_hash = {candidate.source_sha256: candidate for candidate in candidates}
    for candidate in candidates:
        path = paths.selection / (
            f"gen_{candidate.generation}_{candidate.source_sha256[:12]}.json"
        )
        record = _load_development_record(
            path,
            regime="punctuated",
            candidate=candidate,
            development_manifest_sha256=development_manifest_hash,
            simulator_config_sha256=simulator_config_hash,
            numerical_repeats=spec["numerical_repeats"],
            spec_hash=spec_hash,
        )
        if record is None:
            if _phase == "freeze":
                raise RuntimeError(
                    f"freeze-only cannot fill missing development record: {path}"
                )
            record = reevaluate_candidate(
                "punctuated",
                candidate,
                manifest,
                config,
                dependencies,
                numerical_repeats=spec["numerical_repeats"],
                spec_hash=spec_hash,
                candidate_output_width=run_spec.candidate_output_width(spec),
                founder_index_path=founder_index_path,
                contract_version=run_spec.candidate_contract_version(spec),
                initial_source_path=run_spec.initial_program_path(spec),
            )
            _write_atomic(path, _json_bytes(record))
        evaluations.append(record)

    evaluations.sort(
        key=lambda item: (
            not item["development_integrity_valid"],
            -item["development_score"],
            item["generation"],
        )
    )
    if not evaluations[0]["development_integrity_valid"]:
        raise RuntimeError(
            "no structured-random candidate passed development integrity"
        )
    if _phase == "evaluate":
        return _publish_development_completion(
            spec,
            spec_hash,
            "structured_random",
            candidates,
            paths.selection,
        )
    winner_record = evaluations[0]
    winner = candidate_by_hash[winner_record["candidate_source_sha256"]]
    archive_bytes = archive_path.read_bytes()
    selection_artifacts = {
        path.name: run_spec.sha256_file(path)
        for path in sorted(paths.selection.glob("*.json"))
    }
    freeze_record = {
        "schema_version": 2,
        "run_id": spec["run_id"],
        "regime": "punctuated",
        "control": "structured_random",
        "run_spec_sha256": spec_hash,
        "run_spec": spec,
        "selection_rule": {
            "name": "development_score_desc_then_generation_asc",
            "training_rank": "descending combined_score, then generation",
            "top_k": spec["top_k"],
            "development_rank": "integrity, median repeated score, then generation",
            "numerical_repeats": spec["numerical_repeats"],
        },
        "selected_generation": winner.generation,
        "candidate_source_sha256": winner.source_sha256,
        "training_score": winner.training_score,
        "training_manifest_sha256": winner_record["training_manifest_sha256"],
        "development_score": winner_record["development_score"],
        "development_integrity_valid": winner_record["development_integrity_valid"],
        "numerical_repeats": winner_record["numerical_repeats"],
        "repeat_scores": winner_record["repeat_scores"],
        "selected_repeat_index": winner_record["selected_repeat_index"],
        "development_manifest_sha256": winner_record["development_manifest_sha256"],
        "simulator_config_sha256": winner_record["simulator_config_sha256"],
        "simulator_config": asdict(config),
        "sealed_manifest_sha256": run_spec.manifest_hash(spec, "sealed"),
        "sealed_manifest_locked": True,
        "source_sha256": source_hashes,
        "founder_index_sha256": spec["founder_index"]["sha256"],
        "candidate_output_width": run_spec.candidate_output_width(spec),
        "matched_run_counts": matched_counts,
        "structured_random_completion_sha256": run_spec.sha256_file(completion_path),
        "archive_config": spec["archive"],
        "baselines": spec["baselines"],
        "bootstrap_config": spec["bootstrap"],
        "archive_lineage_record": "archive_lineage.json",
        "archive_lineage_sha256": run_spec.sha256_bytes(archive_bytes),
        "archive_program_count": len(archive_record["records"]),
        "archive_correct_program_count": completion["valid_evaluation_count"],
        "selection_artifact_sha256": selection_artifacts,
        "development_evaluation_record": (
            f"gen_{winner.generation}_{winner.source_sha256[:12]}.json"
        ),
        "ranked_development_candidates": [
            {
                "generation": item["generation"],
                "candidate_source_sha256": item["candidate_source_sha256"],
                "training_score": item["training_score"],
                "development_score": item["development_score"],
                "development_integrity_valid": item["development_integrity_valid"],
                "numerical_repeats": item["numerical_repeats"],
                "repeat_scores": item["repeat_scores"],
                "selected_repeat_index": item["selected_repeat_index"],
            }
            for item in evaluations
        ],
    }
    freeze_record.update(_protocol_freeze_fields(spec, source_hashes))
    _freeze(paths.frozen, winner, freeze_record, archive_bytes)
    return freeze_record


def evaluate_structured_random_development(
    *,
    spec: dict[str, Any],
    spec_hash: str,
    spec_raw: bytes,
) -> dict[str, Any]:
    """Write and authenticate the control's five development evaluations only."""
    return select_and_freeze_structured_random(
        spec=spec,
        spec_hash=spec_hash,
        spec_raw=spec_raw,
        _phase="evaluate",
    )


def freeze_structured_random_finalist(
    *,
    spec: dict[str, Any],
    spec_hash: str,
    spec_raw: bytes,
) -> dict[str, Any]:
    """Freeze the control champion from complete existing records only."""
    return select_and_freeze_structured_random(
        spec=spec,
        spec_hash=spec_hash,
        spec_raw=spec_raw,
        _phase="freeze",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--regime", choices=task_evaluator.REGIMES)
    operation.add_argument(
        "--structured-random-phase",
        choices=("training", "evaluate", "freeze"),
    )
    parser.add_argument("--development-phase", choices=("evaluate", "freeze"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run-spec", type=Path)
    selection.add_argument("--profile")
    arguments = parser.parse_args()
    if (
        arguments.structured_random_phase is not None
        and arguments.development_phase is not None
    ):
        parser.error("--development-phase applies only with --regime")
    return arguments


def main() -> None:
    arguments = _parse_args()
    spec, spec_hash, spec_raw = run_spec.load_run_spec(
        arguments.run_spec,
        profile=arguments.profile,
    )
    if arguments.structured_random_phase == "training":
        freeze_record = run_structured_random_training(
            spec=spec,
            spec_hash=spec_hash,
            spec_raw=spec_raw,
        )
    elif arguments.structured_random_phase == "evaluate":
        freeze_record = evaluate_structured_random_development(
            spec=spec,
            spec_hash=spec_hash,
            spec_raw=spec_raw,
        )
    elif arguments.structured_random_phase == "freeze":
        freeze_record = freeze_structured_random_finalist(
            spec=spec,
            spec_hash=spec_hash,
            spec_raw=spec_raw,
        )
    elif arguments.development_phase == "evaluate":
        freeze_record = evaluate_development_candidates(
            arguments.regime,
            spec=spec,
            spec_hash=spec_hash,
            spec_raw=spec_raw,
        )
    elif arguments.development_phase == "freeze":
        freeze_record = freeze_development_finalist(
            arguments.regime,
            spec=spec,
            spec_hash=spec_hash,
            spec_raw=spec_raw,
        )
    else:
        freeze_record = select_and_freeze(
            arguments.regime,
            spec=spec,
            spec_hash=spec_hash,
            spec_raw=spec_raw,
        )
    print(json.dumps(freeze_record, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
