"""Run a small, fully logged exploratory Shinka search on R6 ecology."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys

import numpy as np


TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[2]
MICROCOSMOS_ROOT = PROJECT_ROOT / "microcosmos"
RESULTS_ROOT = TASK_DIR / "results"
FROZEN_MODEL = "headless/codex@gpt-5.5?effort=high"
HEADLESS_COMMAND = "npx -y @roberttlange/headless@0.4.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace {path}")
        return
    path.write_bytes(payload)


def _git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _prepare_manifest(run_root: Path) -> tuple[Path, Path]:
    for path in (MICROCOSMOS_ROOT, MICROCOSMOS_ROOT / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from experiments.evo2_ecosystem.r6.protocol import (  # noqa: PLC0415
        FOUNDER_INDEX_PATH,
        WORLD_QUALIFICATION_PATH,
    )
    from experiments.evo2_ecosystem.r6.qualification import (  # noqa: PLC0415
        _selected_world_seeds,
        build_manifest_bundle,
    )
    from experiments.evo2_ecosystem.r6.qualify_worlds import (  # noqa: PLC0415
        validate_world_qualification,
    )
    from experiments.evo2_ecosystem.protocol import (  # noqa: PLC0415
        canonical_manifest_bytes,
    )

    world = validate_world_qualification(WORLD_QUALIFICATION_PATH)
    seeds = _selected_world_seeds(world)
    bundle = build_manifest_bundle(
        FOUNDER_INDEX_PATH,
        {name: seeds[name] for name in ("training", "development", "sealed")},
    )
    manifest_path = run_root / "inputs" / "training_punctuated.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_manifest_bytes(bundle.training_punctuated) + b"\n"
    if manifest_path.exists() and manifest_path.read_bytes() != payload:
        raise FileExistsError("exploratory manifest changed")
    manifest_path.write_bytes(payload)
    return manifest_path, FOUNDER_INDEX_PATH


def _runner(
    run_root: Path,
    manifest_path: Path,
    founder_index_path: Path,
    *,
    generations: int,
    regime: str,
):
    from shinka.core import EvolutionConfig, ShinkaEvolveRunner
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig

    arm = run_root / regime
    arm.mkdir(parents=True, exist_ok=False)
    job = LocalJobConfig(
        eval_program_path=str(TASK_DIR / "evaluate_exploratory.py"),
        extra_cmd_args={
            "manifest_path": str(manifest_path.resolve()),
            "founder_index_path": str(founder_index_path.resolve()),
            "regime": regime,
            "numerical_repeats": "1",
        },
        time="00:12:00",
        python_executable=sys.executable,
        numeric_threads_per_job=1,
        eval_verbose=False,
    )
    database = DatabaseConfig(
        db_path=str(arm / "programs.sqlite"),
        num_islands=2,
        archive_size=16,
        num_archive_inspirations=1,
        num_top_k_inspirations=1,
    )
    task = """
Discover a better six-action heredity scheduler for an embodied CPPN ecosystem.

make_offspring receives bounded parent, population, and per-operator success,
usage, and evidence summaries. Return six finite logits for clone,
conservative parametric, standard parametric, exploratory parametric,
structural, and mixed mutation. The ancestor always chooses standard
parametric mutation. Maximize the paired candidate-minus-ancestor ecological
score across matched resource-refresh and resource-relocation worlds. Use
operator evidence to adapt choices rather than merely returning the ancestor.
Only pure bounded jax.numpy expressions are valid.
"""
    evolution = EvolutionConfig(
        task_sys_msg=task,
        patch_types=["diff", "full"],
        patch_type_probs=[0.7, 0.3],
        num_generations=generations,
        max_patch_resamples=2,
        max_patch_attempts=2,
        job_type="local",
        language="python",
        llm_models=[FROZEN_MODEL],
        llm_kwargs={
            "temperatures": [0.0],
            "reasoning_efforts": ["high"],
            "max_tokens": 8192,
        },
        embedding_model=None,
        llm_dynamic_selection="fixed",
        init_program_path=str(TASK_DIR / "initial_r4.py"),
        results_dir=str(arm),
        max_novelty_attempts=1,
        use_text_feedback=True,
    )
    return ShinkaEvolveRunner(
        evo_config=evolution,
        job_config=job,
        db_config=database,
        max_evaluation_jobs=1,
        max_proposal_jobs=1,
        max_db_workers=1,
        verbose=True,
    ), arm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--regime", choices=("stable", "punctuated"), default="punctuated")
    arguments = parser.parse_args()
    if not 2 <= arguments.generations <= 20:
        raise ValueError("exploratory generations must be in [2, 20]")

    random.seed(17)
    np.random.seed(17)
    os.environ["SHINKA_HEADLESS_COMMAND"] = HEADLESS_COMMAND
    run_root = RESULTS_ROOT / arguments.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    manifest_path, founder_index_path = _prepare_manifest(run_root)
    launch = {
        "exploratory": True,
        "founder_index_sha256": _sha256(founder_index_path),
        "generations": arguments.generations,
        "manifest_sha256": _sha256(manifest_path),
        "microcosmos_commit": _git_commit(MICROCOSMOS_ROOT),
        "model": FROZEN_MODEL,
        "regime": arguments.regime,
        "run_id": arguments.run_id,
        "shinkaevolve_commit": _git_commit(PROJECT_ROOT / "ShinkaEvolve"),
    }
    _write_once(run_root / "launch.json", launch)
    runner, arm = _runner(
        run_root,
        manifest_path,
        founder_index_path,
        generations=arguments.generations,
        regime=arguments.regime,
    )
    runner.run()

    connection = sqlite3.connect(arm / "programs.sqlite")
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    completed = sorted(
        path.name
        for path in arm.glob("gen_*")
        if (path / "results" / "metrics.json").is_file()
    )
    _write_once(
        run_root / "complete.json",
        {**launch, "completed_generations": completed},
    )


if __name__ == "__main__":
    main()
