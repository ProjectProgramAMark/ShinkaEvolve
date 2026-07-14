from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from examples.evo2_ecosystem import freeze_finalist
from examples.evo2_ecosystem import run_spec


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _r6_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict, Path]:
    task = tmp_path / "ShinkaEvolve/examples/evo2_ecosystem"
    for name in (
        "initial_r4.py",
        "evaluate.py",
        "run_evo.py",
        "freeze_finalist.py",
        "program_lineage.py",
        "r4_selection.py",
    ):
        _write(task / name, f"# {name}\n")
    lock = tmp_path / "microcosmos/uv.lock"
    _write(lock, "frozen\n")
    _write(tmp_path / run_spec.R6_BASELINE_SOURCE_PATH, "# fixed baseline source\n")

    sources_by_path: dict[str, set[str]] = {}
    definitions = {
        "world_qualification": "validate_world_qualification",
        "manifest_generation": "build_and_publish_manifests",
        "disturbance_qualification": "run_disturbance_qualification",
        "opportunity_qualification": "run_opportunity_qualification",
        "final_analysis": "run_final_analysis",
    }
    for role, relative in run_spec.R6_PROTOCOL_TOOL_PATHS.items():
        sources_by_path.setdefault(relative, set()).add(definitions[role])
    for relative, names in sources_by_path.items():
        _write(
            tmp_path / relative,
            "\n".join(
                f"def {name}(*args, **kwargs):\n    return {{}}\n"
                for name in sorted(names)
            ),
        )
    for relative in run_spec.R6_PROTOCOL_SOURCE_PATHS.values():
        path = tmp_path / relative
        if not path.exists():
            _write(path, "# bound R6 protocol source\n")
    for role, relative in run_spec.R6_PREREQUISITE_PATHS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            run_spec.canonical_json_bytes(
                {"passed": True, "role": role, "schema_version": 1}
            )
        )

    monkeypatch.setattr(run_spec, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_spec, "TASK_DIR", task)
    monkeypatch.setattr(run_spec, "DEPENDENCY_LOCK_PATH", lock)
    manifest_hashes = {
        role: str(index) * 64
        for index, role in enumerate(sorted(run_spec.R6_MANIFEST_PATHS), start=1)
    }
    spec = run_spec.build_r6_run_spec(
        manifest_sha256=manifest_hashes,
        founder_index_sha256="5" * 64,
        simulator_source_sha256="6" * 64,
        simulator_config_sha256="7" * 64,
        repository_commits={"microcosmos": "a" * 40, "shinkaevolve": "b" * 40},
    )
    profile = task / "run_specs/evo2-r6-resource-relocation.json"
    run_spec.publish_r6_run_spec(spec, profile)
    return spec, profile


def test_schema_v5_builder_binds_exact_r6_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, profile = _r6_project(monkeypatch, tmp_path)

    loaded, _, _ = run_spec.load_run_spec(profile)

    assert loaded == spec
    assert loaded["schema_version"] == 5
    assert loaded["run_id"] == run_spec.R6_RUN_ID
    assert loaded["proposal_budget"] == 50
    assert loaded["generations"] == 51
    assert loaded["candidate_output_width"] == 6
    assert run_spec.baseline_source_path(loaded) == (
        tmp_path / run_spec.R6_BASELINE_SOURCE_PATH
    )
    assert set(run_spec.protocol_tool_paths(loaded)) == set(
        run_spec.R6_PROTOCOL_TOOL_CALLABLES
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda spec: spec.update(protocol_revision="wrong"),
        lambda spec: spec["manifests"]["sealed"].update(path="elsewhere.json"),
        lambda spec: spec["protocol_sources"]["workflow"].update(
            path="microcosmos/experiments/evo2_ecosystem/r5/workflow.py"
        ),
        lambda spec: spec["baseline_source"].update(path="other.py"),
    ],
)
def test_schema_v5_rejects_protocol_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation,
) -> None:
    spec, _ = _r6_project(monkeypatch, tmp_path)
    changed = copy.deepcopy(spec)
    mutation(changed)

    with pytest.raises(ValueError):
        run_spec._validate(changed)


def test_schema_v5_candidate_roster_excludes_generation_zero(
    tmp_path: Path,
) -> None:
    def add(generation: int, source: str, score: float) -> None:
        directory = tmp_path / f"gen_{generation}"
        results = directory / "results"
        results.mkdir(parents=True)
        source_bytes = source.encode()
        (directory / "main.py").write_bytes(source_bytes)
        digest = run_spec.sha256_bytes(source_bytes)
        (results / "correct.json").write_text(
            json.dumps({"correct": True}), encoding="utf-8"
        )
        (results / "metrics.json").write_text(
            json.dumps(
                {
                    "combined_score": score,
                    "private": {"candidate_sha256": digest},
                }
            ),
            encoding="utf-8",
        )

    add(0, "ancestor", 100.0)
    for generation in range(1, 7):
        add(generation, f"child-{generation}", float(generation))

    historical = freeze_finalist.discover_candidates(tmp_path)[:5]
    r6 = freeze_finalist.discover_candidates(tmp_path, minimum_generation=1)[:5]

    assert 0 in {item.generation for item in historical}
    assert [item.generation for item in r6] == [6, 5, 4, 3, 2]
