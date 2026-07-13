"""Launch one arm of the canonical matched Evo² ShinkaEvolve run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shlex
import sqlite3
import sys
from typing import Any, TYPE_CHECKING

import numpy as np

if __package__:
    from examples.evo2_ecosystem import run_spec
else:
    import run_spec

if TYPE_CHECKING:
    from shinka.core import ShinkaEvolveRunner


TASK_DIR = Path(__file__).resolve().parent
MICROCOSMOS_ROOT = TASK_DIR.parents[2] / "microcosmos"
HOLDOUT_PATHS = (
    MICROCOSMOS_ROOT
    / "experiments"
    / "evo2_ecosystem"
    / "manifests"
    / "development.json",
    MICROCOSMOS_ROOT / "experiments" / "evo2_sealed" / "final.json",
)
HEADLESS_COMMAND_ENV = "SHINKA_HEADLESS_COMMAND"
ANALYSIS_PATH = MICROCOSMOS_ROOT / "experiments" / "evo2_ecosystem" / "analysis.py"
BASELINE_PATH = MICROCOSMOS_ROOT / "experiments" / "evo2_ecosystem" / "run_baselines.py"


def _require_holdouts_locked(spec: dict[str, Any] | None = None) -> None:
    """Fail closed if the read-only headless proposer could inspect holdouts."""
    paths = (
        HOLDOUT_PATHS
        if spec is None
        else (*run_spec.holdout_paths(spec), *_founder_holdout_paths(spec))
    )
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"required holdout is missing: {path}")
        if os.access(path, os.R_OK):
            raise RuntimeError(
                f"holdout must be unreadable during outer search: {path}"
            )


def _founder_holdout_paths(spec: dict[str, Any]) -> tuple[Path, ...]:
    """Resolve development/sealed founder artifacts without opening their bytes."""
    if run_spec.schema_version(spec) == 1:
        return ()
    import sys as _sys

    root = str(MICROCOSMOS_ROOT)
    if root not in _sys.path:
        _sys.path.insert(0, root)
    from experiments.evo2_ecosystem.founder_artifacts import (  # noqa: PLC0415
        founder_index_sha256,
        load_founder_index,
    )

    index_path = run_spec.founder_index_path(spec)
    assert index_path is not None
    index = load_founder_index(index_path, verify_artifacts=False)
    if founder_index_sha256(index) != spec["founder_index"]["sha256"]:
        raise RuntimeError("run specification does not match the founder index")
    bank_root = index_path.parent.resolve()
    paths = []
    for record in index.founders:
        if record.partition not in {"development", "sealed"}:
            continue
        path = (bank_root / record.artifact).resolve()
        if bank_root not in path.parents:
            raise RuntimeError("founder holdout path escapes the founder bank")
        paths.append(path)
    return tuple(paths)


def _verify_search_inputs(spec: dict[str, Any], regime: str) -> None:
    import sys as _sys

    root = str(MICROCOSMOS_ROOT)
    if root not in _sys.path:
        _sys.path.insert(0, root)
    from experiments.evo2_ecosystem.episode import (  # noqa: PLC0415
        SimulatorConfig,
        simulator_config_sha256,
        simulator_source_sha256,
    )
    from experiments.evo2_ecosystem.manifests import (  # noqa: PLC0415
        load_bound_manifest,
        load_training_manifest,
    )
    from experiments.evo2_ecosystem.founder_artifacts import (  # noqa: PLC0415
        founder_index_sha256,
        load_founder_artifact,
        load_founder_index,
    )
    from experiments.evo2_ecosystem.frozen_config import (  # noqa: PLC0415
        NUMERICAL_REPEATS,
    )
    from experiments.evo2_ecosystem.protocol import manifest_sha256  # noqa: PLC0415

    source_paths = {
        "initial": run_spec.sha256_file(TASK_DIR / "initial.py"),
        "evaluator": run_spec.sha256_file(TASK_DIR / "evaluate.py"),
        "simulator": simulator_source_sha256(),
        "analysis": run_spec.sha256_file(ANALYSIS_PATH),
        "baseline": run_spec.sha256_file(BASELINE_PATH),
        "dependency_lock": run_spec.sha256_file(run_spec.DEPENDENCY_LOCK_PATH),
        "launcher": run_spec.sha256_file(Path(__file__)),
        "finalist_selector": run_spec.sha256_file(TASK_DIR / "freeze_finalist.py"),
        "lineage_selector": run_spec.sha256_file(TASK_DIR / "program_lineage.py"),
        "run_spec_module": run_spec.sha256_file(TASK_DIR / "run_spec.py"),
    }
    expected_sources = spec["source_sha256"]
    actual_sources = {
        name: source_paths[name]
        for name in expected_sources
        if name != "preregistration"
    }
    if "preregistration" in expected_sources:
        actual_sources["preregistration"] = run_spec.sha256_file(
            run_spec.PREREGISTRATION_PATH
        )
    if any(actual_sources[name] != expected_sources[name] for name in actual_sources):
        raise RuntimeError("run specification does not match trusted search source")
    if run_spec.schema_version(spec) == 1:
        manifest = load_training_manifest(regime)
    else:
        manifest = load_bound_manifest(
            run_spec.manifest_path(spec, f"training_{regime}"),
            expected_sha256=run_spec.manifest_hash(spec, f"training_{regime}"),
            expected_partition="training",
        )
        founder_path = run_spec.founder_index_path(spec)
        assert founder_path is not None
        founder_index = load_founder_index(founder_path, verify_artifacts=False)
        if founder_index_sha256(founder_index) != spec["founder_index"]["sha256"]:
            raise RuntimeError("run specification does not match the founder index")
        for record in founder_index.founders:
            if record.partition == "training":
                load_founder_artifact(
                    founder_path.parent,
                    record,
                    expected_partition="training",
                )
        for label, path in run_spec.bound_protocol_paths(spec).items():
            if run_spec.sha256_file(path) != spec[label]["sha256"]:
                raise RuntimeError(
                    f"run specification does not match {label.replace('_', ' ')}"
                )
    if manifest_sha256(manifest) != run_spec.manifest_hash(spec, f"training_{regime}"):
        raise RuntimeError("run specification does not match the training manifest")
    if simulator_config_sha256(SimulatorConfig()) != spec["simulator_config_sha256"]:
        raise RuntimeError("run specification does not match simulator configuration")
    if NUMERICAL_REPEATS != spec["numerical_repeats"]:
        raise RuntimeError("run specification does not match numerical repeats")


def _pin_headless(spec: dict[str, Any]) -> None:
    expected = spec["headless_command"]
    configured = os.environ.get(HEADLESS_COMMAND_ENV)
    if configured is not None and configured != expected:
        raise RuntimeError(
            f"{HEADLESS_COMMAND_ENV} conflicts with the run specification"
        )
    os.environ[HEADLESS_COMMAND_ENV] = expected


def _canonical_command(
    regime: str,
    *,
    resume: bool,
    selected_spec_path: Path | None = None,
    profile: str | None = None,
) -> str:
    command = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        "MemoryHigh=20G",
        "-p",
        "MemoryMax=24G",
        "-p",
        "MemorySwapMax=2G",
        "conda",
        "run",
        "-n",
        "sakana",
        "python",
        "examples/evo2_ecosystem/run_evo.py",
        "--regime",
        regime,
    ]
    if profile is not None:
        command.extend(["--profile", profile])
    elif (
        selected_spec_path is not None
        and selected_spec_path != run_spec.SPEC_PATH.resolve()
    ):
        command.extend(["--run-spec", str(selected_spec_path)])
    if resume:
        command.append("--resume")
    return shlex.join(command)


def _prepare_arm(
    spec: dict[str, Any],
    spec_hash: str,
    spec_raw: bytes,
    regime: str,
    *,
    resume: bool,
    selected_spec_path: Path | None = None,
    profile: str | None = None,
) -> run_spec.RunPaths:
    paths = run_spec.paths_for(spec, regime)
    if (paths.run_root / f"{regime}.complete.json").exists():
        raise RuntimeError(f"{regime} arm is already complete")
    run_spec.materialize_run_spec(paths.run_root, spec_raw, spec_hash)
    nonempty = paths.arm_results.is_dir() and any(paths.arm_results.iterdir())
    if resume and not nonempty:
        raise RuntimeError("--resume requires a nonempty arm directory")
    if not resume and nonempty:
        raise RuntimeError(
            "arm directory is nonempty; use --resume only for this exact run"
        )
    paths.arm_results.mkdir(parents=True, exist_ok=True)

    if resume:
        index = 1
        while (paths.arm_results / f"resume_{index:03d}.json").exists():
            index += 1
        launch_path = paths.arm_results / f"resume_{index:03d}.json"
    else:
        launch_path = paths.arm_results / "launch.json"
    launch = {
        "schema_version": 1,
        "regime": regime,
        "run_id": spec["run_id"],
        "run_spec_sha256": spec_hash,
        "canonical_command": _canonical_command(
            regime,
            resume=resume,
            selected_spec_path=selected_spec_path,
            profile=profile,
        ),
        "process_argv": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd().resolve()),
        "headless_command": spec["headless_command"],
        "resume": resume,
        "selected_run_spec_path": str(
            (selected_spec_path or run_spec.SPEC_PATH.resolve()).resolve()
        ),
        "profile": profile,
    }
    run_spec.write_once(launch_path, run_spec.canonical_json_bytes(launch))
    return paths


def _require_sibling_isolated(
    spec: dict[str, Any],
    spec_hash: str,
    regime: str,
) -> None:
    sibling = next(item for item in run_spec.REGIMES if item != regime)
    sibling_paths = run_spec.paths_for(spec, sibling)
    if not sibling_paths.arm_results.exists():
        return
    marker_path = sibling_paths.run_root / f"{sibling}.complete.json"
    if not marker_path.is_file():
        raise RuntimeError("sibling arm exists but has not completed")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "run_id": spec["run_id"],
        "regime": sibling,
        "run_spec_sha256": spec_hash,
        "completed_evaluation_count": spec["generations"],
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise RuntimeError("sibling completion marker does not match the run spec")
    permissions = sibling_paths.arm_results.stat().st_mode & 0o777
    if permissions & 0o555 or os.access(sibling_paths.arm_results, os.R_OK):
        raise RuntimeError(
            "completed sibling arm must be unreadable before second launch"
        )


def _write_completion(
    spec: dict[str, Any],
    spec_hash: str,
    regime: str,
    paths: run_spec.RunPaths,
) -> None:
    completed = []
    for generation_dir in paths.arm_results.glob("gen_*"):
        try:
            generation = int(generation_dir.name.removeprefix("gen_"))
        except ValueError:
            continue
        if all(
            path.is_file()
            for path in (
                generation_dir / "main.py",
                generation_dir / "results" / "metrics.json",
                generation_dir / "results" / "correct.json",
            )
        ):
            completed.append(generation)
    if sorted(completed) != list(range(spec["generations"])):
        raise RuntimeError("outer search returned without completing its fixed budget")
    database = paths.arm_results / "programs.sqlite"
    if not database.is_file():
        raise RuntimeError("outer search returned without its archive database")
    # Shinka uses WAL mode. Its runner can return just before SQLite applies the
    # final checkpoint to the main file, so hash only after an explicit, closed
    # checkpoint. Otherwise the marker authenticates the common pre-checkpoint
    # database image instead of the completed archive.
    connection = sqlite3.connect(database)
    try:
        busy, _, _ = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        connection.close()
    if busy:
        raise RuntimeError("archive database is still busy after outer search")
    wal = Path(f"{database}-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("archive database still has an active WAL")
    marker = {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "regime": regime,
        "run_spec_sha256": spec_hash,
        "completed_evaluation_count": len(completed),
        "database_sha256": run_spec.sha256_file(database),
    }
    run_spec.write_once(
        paths.run_root / f"{regime}.complete.json",
        run_spec.canonical_json_bytes(marker),
    )


def build_runner(
    regime: str,
    spec: dict[str, Any],
    paths: run_spec.RunPaths,
    *,
    spec_hash: str | None = None,
) -> ShinkaEvolveRunner:
    from shinka.core import EvolutionConfig, ShinkaEvolveRunner
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig

    random.seed(spec["outer_seed"])
    np.random.seed(spec["outer_seed"])
    archive = spec["archive"]
    search = spec["search"]

    materialized_spec = paths.run_root / "run_spec.json"
    if spec_hash is None:
        spec_hash = run_spec.sha256_file(materialized_spec)
    job_config = LocalJobConfig(
        eval_program_path=str(TASK_DIR / "evaluate.py"),
        extra_cmd_args={
            "training_regime": regime,
            "run_spec_path": str(materialized_spec.resolve()),
            "run_spec_sha256": spec_hash,
        },
        time=search["evaluation_timeout"],
        python_executable=sys.executable,
        numeric_threads_per_job=1,
        eval_verbose=False,
    )
    database_config = DatabaseConfig(
        db_path=str(paths.arm_results / "programs.sqlite"),
        num_islands=archive["num_islands"],
        archive_size=archive["archive_size"],
        num_archive_inspirations=archive["num_archive_inspirations"],
        num_top_k_inspirations=archive["num_top_k_inspirations"],
    )
    output_width = run_spec.candidate_output_width(spec)
    task_message = f"""
Improve make_offspring for Evo² heredity-policy training.

The four inputs are fixed-shape summaries:
- parent_genome: [node_fraction, connection_fraction]
- parent_stats: [energy_fraction, intake_ema]
- population_stats: [alive_fraction, population_change_ema,
  action_diversity, lineage_entropy]
- rng: opaque reserved argument; candidate code must not read it

Return exactly {output_width} finite scores. The first four are
[clone, parametric, structural, mixed]; any additional trusted actions are
defined only by this run's immutable protocol.
The trusted evaluator performs the selected TensorNEAT operation. Only pure,
bounded jax.numpy expressions are accepted; no loops, imports, I/O, mutation
APIs, simulator state, scenario identity, or hidden manifests are available.
Maximize post-event resource-productivity AUC without integrity violations.
"""
    evolution_config = EvolutionConfig(
        task_sys_msg=task_message,
        patch_types=search["patch_types"],
        patch_type_probs=search["patch_type_probs"],
        num_generations=spec["generations"],
        max_patch_resamples=search["max_patch_resamples"],
        max_patch_attempts=search["max_patch_attempts"],
        job_type="local",
        language="python",
        llm_models=[spec["model"]],
        llm_kwargs={
            "temperatures": [search["temperature"]],
            "reasoning_efforts": [search["reasoning_effort"]],
            "max_tokens": search["max_tokens"],
        },
        embedding_model=None,
        llm_dynamic_selection="fixed",
        init_program_path=str(TASK_DIR / "initial.py"),
        results_dir=str(paths.arm_results),
        max_novelty_attempts=search["max_novelty_attempts"],
        use_text_feedback=True,
    )
    return ShinkaEvolveRunner(
        evo_config=evolution_config,
        job_config=job_config,
        db_config=database_config,
        max_evaluation_jobs=search["max_evaluation_jobs"],
        max_proposal_jobs=search["max_proposal_jobs"],
        max_db_workers=search["max_db_workers"],
        verbose=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=run_spec.REGIMES, required=True)
    parser.add_argument("--resume", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run-spec", type=Path)
    selection.add_argument("--profile")
    arguments = parser.parse_args()
    selected_spec_path = run_spec.resolve_run_spec_path(
        arguments.run_spec,
        profile=arguments.profile,
    )
    spec, spec_hash, spec_raw = run_spec.load_run_spec(selected_spec_path)
    _require_holdouts_locked(spec)
    _pin_headless(spec)
    _verify_search_inputs(spec, arguments.regime)
    _require_sibling_isolated(spec, spec_hash, arguments.regime)
    paths = _prepare_arm(
        spec,
        spec_hash,
        spec_raw,
        arguments.regime,
        resume=arguments.resume,
        selected_spec_path=selected_spec_path,
        profile=arguments.profile,
    )
    build_runner(arguments.regime, spec, paths, spec_hash=spec_hash).run()
    _write_completion(spec, spec_hash, arguments.regime, paths)


if __name__ == "__main__":
    main()
