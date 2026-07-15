"""Freeze R7 finalists, fixed baselines, and fresh development inputs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[2]
MICROCOSMOS_ROOT = PROJECT_ROOT / "microcosmos"
RESULTS_ROOT = TASK_DIR / "results"
OPERATOR_NAMES = (
    "clone",
    "parametric_conservative",
    "parametric_standard",
    "parametric_exploratory",
    "structural",
    "mixed",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixed_program(index: int) -> str:
    logits = [-8.0] * len(OPERATOR_NAMES)
    logits[index] = 8.0
    values = ", ".join(f"{value:.1f}" for value in logits)
    return f'''import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
): 
    """Always select the frozen {OPERATOR_NAMES[index]} operator."""
    return jnp.array([{values}], dtype=jnp.float32)


# EVOLVE-BLOCK-END
'''


def _confirmation_manifest(
    output: Path,
    *,
    partition: str,
    scenario_family: str,
    event_step: int | None,
) -> tuple[Path, Path]:
    for path in (MICROCOSMOS_ROOT, MICROCOSMOS_ROOT / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from experiments.evo2_ecosystem.protocol import (  # noqa: PLC0415
        ActuatorInjuryParameters,
        EventKind,
        NullEventParameters,
        canonical_manifest_bytes,
    )
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

    seeds = _selected_world_seeds(validate_world_qualification(WORLD_QUALIFICATION_PATH))
    bundle = build_manifest_bundle(
        FOUNDER_INDEX_PATH,
        {name: seeds[name] for name in ("training", "development", "sealed")},
    )
    selected = {
        "training": bundle.training_punctuated,
        "development": bundle.development,
        "sealed": bundle.sealed,
    }[partition]
    if scenario_family == "head_actuator_injury":
        worlds = []
        for item in selected.worlds:
            if item.scenario_id.endswith("-control"):
                worlds.append(
                    replace(
                        item,
                        scenario_id=f"{item.pair_id}-sham",
                        scenario_family="actuator_injury",
                        event_kind=EventKind.NULL,
                        event_parameters=NullEventParameters(),
                    )
                )
            elif item.scenario_id.endswith("-shock"):
                worlds.append(
                    replace(
                        item,
                        scenario_id=f"{item.pair_id}-injured",
                        scenario_family="actuator_injury",
                        event_kind=EventKind.ACTUATOR_INJURY,
                        event_parameters=ActuatorInjuryParameters(
                            (0.1, 0.1, 1.0, 1.0, 1.0, 1.0)
                        ),
                    )
                )
            else:
                raise ValueError("unexpected R6 confirmation scenario role")
        selected = replace(selected, worlds=tuple(worlds))
    elif scenario_family != "resource_relocation":
        raise ValueError("unknown confirmation scenario family")
    if event_step is not None:
        if (
            event_step <= 0
            or event_step >= selected.horizon
            or event_step % selected.chunk_steps
        ):
            raise ValueError(
                "event_step must be a positive chunk boundary before the horizon"
            )
        selected = replace(
            selected,
            worlds=tuple(replace(item, event_step=event_step) for item in selected.worlds),
        )
    manifest = output / "inputs" / f"{partition}_punctuated.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(canonical_manifest_bytes(selected) + b"\n")
    return manifest, FOUNDER_INDEX_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--generations", nargs="+", type=int, required=True)
    parser.add_argument(
        "--partition",
        choices=("training", "development", "sealed"),
        default="development",
    )
    parser.add_argument(
        "--scenario-family",
        choices=("resource_relocation", "head_actuator_injury"),
        default="resource_relocation",
    )
    parser.add_argument(
        "--event-step",
        type=int,
        help="override the paired event time without changing the rollout horizon",
    )
    arguments = parser.parse_args()

    source = RESULTS_ROOT / arguments.source_run_id / "punctuated"
    output = RESULTS_ROOT / arguments.output_run_id
    output.mkdir(parents=True, exist_ok=False)
    programs = output / "programs"
    programs.mkdir()

    program_records = []
    for generation in arguments.generations:
        source_path = source / f"gen_{generation}" / "main.py"
        destination = programs / f"shinka_gen_{generation}.py"
        shutil.copyfile(source_path, destination)
        program_records.append(
            {"name": destination.stem, "sha256": _sha256(destination)}
        )
    for index, name in enumerate(OPERATOR_NAMES):
        destination = programs / f"fixed_{name}.py"
        destination.write_text(_fixed_program(index), encoding="utf-8")
        program_records.append(
            {"name": destination.stem, "sha256": _sha256(destination)}
        )

    manifest, founder_index = _confirmation_manifest(
        output,
        partition=arguments.partition,
        scenario_family=arguments.scenario_family,
        event_step=arguments.event_step,
    )
    launch = {
        "founder_index_sha256": _sha256(founder_index),
        "event_step": arguments.event_step,
        "manifest_sha256": _sha256(manifest),
        "microcosmos_commit": _commit(MICROCOSMOS_ROOT),
        "numerical_repeats": 3,
        "programs": program_records,
        "partition": arguments.partition,
        "scenario_family": arguments.scenario_family,
        "shinkaevolve_commit": _commit(PROJECT_ROOT / "ShinkaEvolve"),
        "source_run_id": arguments.source_run_id,
    }
    (output / "launch.json").write_text(
        json.dumps(launch, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
