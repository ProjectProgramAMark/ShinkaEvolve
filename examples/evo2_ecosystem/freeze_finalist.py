"""Re-evaluate completed Shinka candidates and freeze one development winner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
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
R4_TOOL_ROOT = MICROCOSMOS_ROOT / "experiments" / "evo2_ecosystem" / "heredity_adaptation_v4"
_GENERATION = re.compile(r"gen_(\d+)")


@dataclass(frozen=True)
class Candidate:
    """One complete, integrity-valid generation result."""

    generation: int
    source_path: Path
    source_sha256: str
    training_score: float
    training_metrics: dict[str, Any]


def discover_candidates(
    results_dir: Path,
    expected_private: dict[str, Any] | None = None,
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
        candidates.append(
            Candidate(
                generation=int(match.group(1)),
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
    all_sources = {
        "initial": run_spec.sha256_file(run_spec.initial_program_path(spec)),
        "evaluator": run_spec.sha256_file(TASK_DIR / "evaluate.py"),
        "simulator": dependencies["simulator_source_sha256"](),
        "analysis": run_spec.sha256_file(ANALYSIS_PATH),
        "baseline": run_spec.sha256_file(BASELINE_PATH),
        "dependency_lock": run_spec.sha256_file(run_spec.DEPENDENCY_LOCK_PATH),
        "preregistration": run_spec.sha256_file(run_spec.PREREGISTRATION_PATH),
        "launcher": run_spec.sha256_file(TASK_DIR / "run_evo.py"),
        "finalist_selector": run_spec.sha256_file(Path(__file__)),
        "lineage_selector": run_spec.sha256_file(TASK_DIR / "program_lineage.py"),
        "run_spec_module": run_spec.sha256_file(TASK_DIR / "run_spec.py"),
        "adaptive_selector": run_spec.sha256_file(TASK_DIR / "r4_selection.py"),
        "r4_final_analysis": run_spec.sha256_file(R4_TOOL_ROOT / "analysis.py"),
        "r4_manifest_generator": run_spec.sha256_file(R4_TOOL_ROOT / "manifest_generator.py"),
        "r4_qualification": run_spec.sha256_file(R4_TOOL_ROOT / "qualification.py"),
        "r4_opportunity": run_spec.sha256_file(R4_TOOL_ROOT / "opportunity.py"),
    }
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


def select_and_freeze(
    regime: str,
    *,
    spec: dict[str, Any],
    spec_hash: str,
    spec_raw: bytes,
) -> dict[str, Any]:
    """Sequentially evaluate the fixed top-K and freeze the development winner."""
    paths = run_spec.paths_for(spec, regime)
    stored_spec = paths.run_root / "run_spec.json"
    stored_hash = paths.run_root / "run_spec.sha256"
    if (
        not stored_spec.is_file()
        or stored_spec.read_bytes() != spec_raw
        or not stored_hash.is_file()
        or stored_hash.read_text(encoding="utf-8") != f"{spec_hash}  run_spec.json\n"
    ):
        raise RuntimeError("stored run specification does not match the canonical spec")
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
    candidates = discover_candidates(paths.arm_results, expected_private)[
        : spec["top_k"]
    ]
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
            _write_atomic(path, _json_bytes(record))
        evaluations.append(record)

    if not any(item["development_integrity_valid"] for item in evaluations):
        raise RuntimeError("no development candidate passed integrity checks")
    if run_spec.schema_version(spec) >= 3:
        for item in evaluations:
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
        "preregistration_sha256": (
            source_hashes["preregistration"]
            if run_spec.schema_version(spec) == 1
            else spec["preregistration"]["sha256"]
        ),
        "implementation_plan_sha256": (
            None
            if run_spec.schema_version(spec) == 1
            else spec["implementation_plan"]["sha256"]
        ),
        "founder_index_sha256": (
            None
            if run_spec.schema_version(spec) == 1
            else spec["founder_index"]["sha256"]
        ),
        "candidate_output_width": run_spec.candidate_output_width(spec),
        "preregistration_complete": True,
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
    if run_spec.schema_version(spec) < 3:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=task_evaluator.REGIMES, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run-spec", type=Path)
    selection.add_argument("--profile")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    spec, spec_hash, spec_raw = run_spec.load_run_spec(
        arguments.run_spec,
        profile=arguments.profile,
    )
    freeze_record = select_and_freeze(
        arguments.regime,
        spec=spec,
        spec_hash=spec_hash,
        spec_raw=spec_raw,
    )
    print(json.dumps(freeze_record, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
