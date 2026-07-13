from __future__ import annotations

import json
from pathlib import Path
import subprocess

import jax.numpy as jnp
import numpy as np
import pytest

from examples.evo2_ecosystem import evaluate, run_evo, run_spec


def _binding(path: str, digest: str) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def _write_file(root: Path, relative: str, source: str) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return _binding(relative, run_spec.sha256_file(path))


def _write_spec(path: Path, spec: dict) -> None:
    raw = run_spec.canonical_json_bytes(spec)
    digest = run_spec.sha256_bytes(raw)
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )


def _r5_fixture(tmp_path: Path) -> tuple[dict, Path]:
    pilot, _, _ = run_spec.load_run_spec()
    initial = _write_file(
        tmp_path,
        "ShinkaEvolve/examples/evo2_ecosystem/initial_r4.py",
        "import jax.numpy as jnp\n",
    )
    world_tool = _write_file(
        tmp_path,
        f"{run_spec.R5_PROTOCOL_ROOT}/qualify_worlds.py",
        "import json\n"
        "def validate_world_qualification(path, **kwargs):\n"
        "    return json.loads(path.read_text(encoding='utf-8'))\n",
    )
    shared_tools = _write_file(
        tmp_path,
        f"{run_spec.R5_PROTOCOL_ROOT}/tools.py",
        "def build_and_publish_manifests(*args, **kwargs):\n"
        "    return {}\n"
        "def run_disturbance_qualification(*args, **kwargs):\n"
        "    return {}\n"
        "def run_opportunity_qualification(*args, **kwargs):\n"
        "    return {}\n",
    )
    analysis_tool = _write_file(
        tmp_path,
        f"{run_spec.R5_PROTOCOL_ROOT}/analysis.py",
        "def run_final_analysis(*args, **kwargs):\n"
        "    return {}\n",
    )
    protocol_document = _write_file(
        tmp_path,
        run_spec.R5_PROTOCOL_DOCUMENT_PATH,
        "# Frozen R5 protocol\n",
    )

    prerequisites = {}
    for role, relative in run_spec.R5_PREREQUISITE_PATHS.items():
        payload = {"passed": True, "role": role, "schema_version": 1}
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(run_spec.canonical_json_bytes(payload))
        prerequisites[role] = {
            "path": relative,
            "sha256": run_spec.sha256_file(path),
            "passed": True,
        }

    source_hashes = {
        name: format(index, "x")[-1] * 64
        for index, name in enumerate(sorted(run_spec._V4_SOURCE_KEYS), start=1)
    }
    source_hashes["initial"] = initial["sha256"]
    source_hashes["analysis"] = analysis_tool["sha256"]
    spec = {
        **{
            key: value
            for key, value in pilot.items()
            if key not in {"manifest_sha256", "source_sha256", "schema_version"}
        },
        "schema_version": 4,
        "run_id": run_spec.R5_RUN_ID,
        "generations": 51,
        "proposal_budget": 50,
        "top_k": 5,
        "archive": {
            "archive_size": 32,
            "num_archive_inspirations": 1,
            "num_islands": 2,
            "num_top_k_inspirations": 1,
        },
        "baselines": list(run_spec.R5_BASELINES),
        "manifests": {
            role: _binding(path, str(index) * 64)
            for index, (role, path) in enumerate(
                sorted(run_spec.R5_MANIFEST_PATHS.items()), start=1
            )
        },
        "founder_index": _binding(run_spec.R5_FOUNDER_INDEX_PATH, "5" * 64),
        "candidate_output_width": 6,
        "candidate_contract": {
            "version": run_spec.R4_CONTRACT,
            "argument_names": [
                "parent_genome_summary",
                "parent_stats",
                "population_stats",
                "operator_stats",
                "rng",
            ],
            "input_shapes": [[2], [3], [6], [3, 6]],
            "output_shape": [6],
            "logit_clip": [-8.0, 8.0],
            "rng_readable": False,
            "runtime_budget_ms": 100.0,
        },
        "initial_program": initial,
        "artifact_roots": dict(run_spec.R5_ARTIFACT_ROOTS),
        "source_sha256": source_hashes,
        "protocol_revision": run_spec.R5_PROTOCOL_REVISION,
        "protocol_tools": {
            "world_qualification": world_tool,
            "manifest_generation": shared_tools,
            "disturbance_qualification": shared_tools,
            "opportunity_qualification": shared_tools,
            "final_analysis": analysis_tool,
        },
        "repository_commits": {
            "microcosmos": "a" * 40,
            "shinkaevolve": "b" * 40,
        },
        "prerequisite_artifacts": prerequisites,
        "protocol_document": protocol_document,
    }
    profile = tmp_path / "r5.json"
    _write_spec(profile, spec)
    return spec, profile


def test_r5_profile_builder_is_single_source_and_requires_passing_gates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path
    task = project / "ShinkaEvolve/examples/evo2_ecosystem"
    task.mkdir(parents=True)
    for name in (
        "initial_r4.py",
        "evaluate.py",
        "run_evo.py",
        "freeze_finalist.py",
        "program_lineage.py",
        "r4_selection.py",
    ):
        (task / name).write_text(f"# {name}\n", encoding="utf-8")
    lock = project / "microcosmos/uv.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("frozen\n", encoding="utf-8")
    (project / run_spec.R5_BASELINE_SOURCE_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (project / run_spec.R5_BASELINE_SOURCE_PATH).write_text(
        "# baselines\n",
        encoding="utf-8",
    )
    (project / run_spec.R5_PROTOCOL_DOCUMENT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (project / run_spec.R5_PROTOCOL_DOCUMENT_PATH).write_text(
        "# protocol\n",
        encoding="utf-8",
    )
    tool_sources = {
        "world_qualification": "def validate_world_qualification(path, **kwargs):\n    return {}\n",
        "manifest_generation": "def build_and_publish_manifests(*args, **kwargs):\n    return {}\n",
        "disturbance_qualification": "def run_disturbance_qualification(*args, **kwargs):\n    return {}\n",
        "opportunity_qualification": "def run_opportunity_qualification(*args, **kwargs):\n    return {}\n",
        "final_analysis": "def run_final_analysis(*args, **kwargs):\n    return {}\n",
    }
    by_path: dict[str, list[str]] = {}
    for role, relative in run_spec.R5_PROTOCOL_TOOL_PATHS.items():
        by_path.setdefault(relative, []).append(tool_sources[role])
    for relative, sources in by_path.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(sources), encoding="utf-8")
    for role, relative in run_spec.R5_PREREQUISITE_PATHS.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            run_spec.canonical_json_bytes(
                {"passed": True, "role": role, "schema_version": 1}
            )
        )

    monkeypatch.setattr(run_spec, "PROJECT_ROOT", project)
    monkeypatch.setattr(run_spec, "TASK_DIR", task)
    monkeypatch.setattr(run_spec, "DEPENDENCY_LOCK_PATH", lock)
    hashes = {role: str(index) * 64 for index, role in enumerate(sorted(run_spec.R5_MANIFEST_PATHS), start=1)}
    built = run_spec.build_r5_run_spec(
        manifest_sha256=hashes,
        founder_index_sha256="5" * 64,
        simulator_source_sha256="6" * 64,
        simulator_config_sha256="7" * 64,
        repository_commits={"microcosmos": "a" * 40, "shinkaevolve": "b" * 40},
    )

    assert built["schema_version"] == 4
    assert built["generations"] == 51
    assert built["proposal_budget"] == 50
    assert built["outer_seed"] == 17
    assert built["top_k"] == 5
    assert built["baselines"] == run_spec.R5_BASELINES
    assert all(
        binding["passed"] is True
        for binding in built["prerequisite_artifacts"].values()
    )
    profile = tmp_path / "profile.json"
    digest = run_spec.publish_r5_run_spec(built, profile)
    loaded, loaded_digest, _ = run_spec.load_run_spec(profile)
    assert loaded == built
    assert loaded_digest == digest

    failed = project / run_spec.R5_PREREQUISITE_PATHS["operator_opportunity"]
    failed.write_bytes(
        run_spec.canonical_json_bytes(
            {"passed": False, "role": "operator_opportunity", "schema_version": 1}
        )
    )
    with pytest.raises(ValueError, match="canonical passing evidence"):
        run_spec.build_r5_run_spec(
            manifest_sha256=hashes,
            founder_index_sha256="5" * 64,
            simulator_source_sha256="6" * 64,
            simulator_config_sha256="7" * 64,
            repository_commits={
                "microcosmos": "a" * 40,
                "shinkaevolve": "b" * 40,
            },
        )


def test_schema_v4_authenticates_exact_tools_and_protocol_bindings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, profile = _r5_fixture(tmp_path)
    monkeypatch.setattr(run_spec, "PROJECT_ROOT", tmp_path)

    loaded, _, _ = run_spec.load_run_spec(profile)

    assert loaded == spec
    assert set(run_spec.protocol_tool_paths(loaded)) == set(
        run_spec.R5_PROTOCOL_TOOL_CALLABLES
    )
    assert run_spec.bound_protocol_paths(loaded) == {
        "protocol_document": tmp_path / run_spec.R5_PROTOCOL_DOCUMENT_PATH
    }
    assert run_spec.candidate_contract_version(loaded) == run_spec.R4_CONTRACT


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda spec: spec["protocol_tools"].update(
                {"unexpected": spec["protocol_tools"]["final_analysis"]}
            ),
            "protocol-tool roles",
        ),
        (
            lambda spec: spec["prerequisite_artifacts"].pop(
                "operator_opportunity"
            ),
            "prerequisite roles",
        ),
        (
            lambda spec: spec["repository_commits"].update(
                {"microcosmos": "abc"}
            ),
            "invalid Git commit",
        ),
    ],
)
def test_schema_v4_rejects_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    spec, profile = _r5_fixture(tmp_path)
    monkeypatch.setattr(run_spec, "PROJECT_ROOT", tmp_path)
    mutation(spec)
    _write_spec(profile, spec)

    with pytest.raises(ValueError, match=message):
        run_spec.load_run_spec(profile)


def test_schema_v4_rejects_tool_hash_callable_and_symlink_escape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, profile = _r5_fixture(tmp_path)
    monkeypatch.setattr(run_spec, "PROJECT_ROOT", tmp_path)
    world_path = tmp_path / spec["protocol_tools"]["world_qualification"]["path"]

    world_path.write_text("def different_function():\n    return None\n")
    with pytest.raises(ValueError, match="SHA-256"):
        run_spec.load_run_spec(profile)

    spec["protocol_tools"]["world_qualification"]["sha256"] = (
        run_spec.sha256_file(world_path)
    )
    _write_spec(profile, spec)
    with pytest.raises(ValueError, match="lacks callable"):
        run_spec.load_run_spec(profile)

    outside = tmp_path / "outside.py"
    outside.write_text(
        "def validate_world_qualification(path):\n    return {'passed': True}\n"
    )
    escaped = tmp_path / run_spec.R5_PROTOCOL_ROOT / "escaped.py"
    escaped.symlink_to(outside)
    spec["protocol_tools"]["world_qualification"] = _binding(
        str(escaped.relative_to(tmp_path)),
        run_spec.sha256_file(outside),
    )
    _write_spec(profile, spec)
    with pytest.raises(ValueError, match="outside its allowlisted root"):
        run_spec.load_run_spec(profile)


def test_run_evo_imports_bound_callables_and_revalidates_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, profile = _r5_fixture(tmp_path)
    monkeypatch.setattr(run_spec, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_evo, "MICROCOSMOS_ROOT", tmp_path / "microcosmos")
    loaded, _, _ = run_spec.load_run_spec(profile)

    tools = run_evo._resolve_protocol_tools(loaded)
    evidence = run_evo._verify_prerequisite_artifacts(loaded, tools)

    assert set(tools) == set(run_spec.R5_PROTOCOL_TOOL_CALLABLES)
    assert all(callable(function) for function in tools.values())
    assert set(evidence) == run_spec.R5_PREREQUISITE_ROLES
    assert evidence["world_qualification"]["passed"] is True


def test_run_evo_rejects_noncanonical_or_nonpassing_prerequisite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, profile = _r5_fixture(tmp_path)
    monkeypatch.setattr(run_spec, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_evo, "MICROCOSMOS_ROOT", tmp_path / "microcosmos")
    loaded, _, _ = run_spec.load_run_spec(profile)
    tools = run_evo._resolve_protocol_tools(loaded)
    role = "operator_opportunity"
    path = tmp_path / loaded["prerequisite_artifacts"][role]["path"]
    path.write_text(json.dumps({"passed": True}, indent=2) + "\n")
    with pytest.raises(RuntimeError, match="SHA-256"):
        run_evo._verify_prerequisite_artifacts(loaded, tools)
    loaded["prerequisite_artifacts"][role]["sha256"] = run_spec.sha256_file(path)

    with pytest.raises(RuntimeError, match="not canonical passing evidence"):
        run_evo._verify_prerequisite_artifacts(loaded, tools)

    payload = {"passed": False, "role": role, "schema_version": 1}
    path.write_bytes(run_spec.canonical_json_bytes(payload))
    loaded["prerequisite_artifacts"][role]["sha256"] = run_spec.sha256_file(path)
    with pytest.raises(RuntimeError, match="not canonical passing evidence"):
        run_evo._verify_prerequisite_artifacts(loaded, tools)


def _initialize_repository(path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "r5@example.test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "R5 Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "frozen source"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_r5_git_preflight_allows_only_declared_generated_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, _ = _r5_fixture(tmp_path)
    repositories = {
        "microcosmos": tmp_path / "microcosmos",
        "shinkaevolve": tmp_path / "ShinkaEvolve",
    }
    spec["repository_commits"] = {
        name: _initialize_repository(path) for name, path in repositories.items()
    }
    monkeypatch.setattr(run_spec, "PROJECT_ROOT", tmp_path)

    expected_microcosmos_commit = spec["repository_commits"]["microcosmos"]
    spec["repository_commits"]["microcosmos"] = "0" * 40
    with pytest.raises(RuntimeError, match="HEAD does not match"):
        run_evo._verify_repository_state(spec)
    spec["repository_commits"]["microcosmos"] = expected_microcosmos_commit

    microcosmos_output = (
        repositories["microcosmos"]
        / "experiments/evo2_ecosystem/r5/artifacts/final/result.json"
    )
    microcosmos_output.parent.mkdir(parents=True, exist_ok=True)
    microcosmos_output.write_text("{}\n", encoding="utf-8")
    profile = repositories["shinkaevolve"] / Path(
        run_spec.R5_PROFILE_PATH
    ).relative_to("ShinkaEvolve")
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text("{}\n", encoding="utf-8")
    results = (
        repositories["shinkaevolve"]
        / "examples/evo2_ecosystem/results"
        / run_spec.R5_RUN_ID
        / "launch.json"
    )
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text("{}\n", encoding="utf-8")

    run_evo._verify_repository_state(spec)

    undeclared = (
        repositories["microcosmos"]
        / "experiments/evo2_ecosystem/r5/unplanned_source.py"
    )
    undeclared.write_text("pass\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="undeclared untracked path"):
        run_evo._verify_repository_state(spec)
    undeclared.unlink()

    (repositories["microcosmos"] / ".git/info/exclude").write_text(
        "ignored-scratch.txt\n",
        encoding="utf-8",
    )
    ignored = repositories["microcosmos"] / "ignored-scratch.txt"
    ignored.write_text("hidden from ordinary git status\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="undeclared untracked path"):
        run_evo._verify_repository_state(spec)
    ignored.unlink()

    tracked = repositories["microcosmos"] / "docs/evo2/evo2-r5-world-feasibility-plan.md"
    original = tracked.read_bytes()
    tracked.write_bytes(original + b"changed\n")
    with pytest.raises(RuntimeError, match="tracked modifications"):
        run_evo._verify_repository_state(spec)
    tracked.write_bytes(original)

    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    escaped = microcosmos_output.parent / "escaped.json"
    escaped.symlink_to(outside)
    with pytest.raises(RuntimeError, match="undeclared untracked path"):
        run_evo._verify_repository_state(spec)


def test_legacy_specs_do_not_enter_r5_git_preflight() -> None:
    pilot, _, _ = run_spec.load_run_spec()
    run_evo._verify_repository_state(pilot)


def _write_r4_candidate(tmp_path: Path, expression: str) -> Path:
    initial = Path(evaluate.__file__).with_name("initial_r4.py")
    prefix, rest = initial.read_text(encoding="utf-8").split(
        evaluate.START_MARKER, 1
    )
    _, suffix = rest.split(evaluate.END_MARKER, 1)
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        f"{prefix}{evaluate.START_MARKER}\n"
        "def make_offspring(parent_genome_summary, parent_stats, "
        "population_stats, operator_stats, rng):\n"
        f"    return {expression}\n"
        f"{evaluate.END_MARKER}{suffix}",
        encoding="utf-8",
    )
    return candidate


def test_candidate_boundary_accepts_only_bounded_matrix_multiplication(
    tmp_path: Path,
) -> None:
    candidate = _write_r4_candidate(
        tmp_path,
        "jnp.clip(jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], "
        "[-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0]]) "
        "@ parent_genome_summary, -8.0, 8.0)",
    )
    module = evaluate._load_candidate(
        candidate,
        run_spec.R4_CONTRACT,
        Path(evaluate.__file__).with_name("initial_r4.py"),
    )
    evaluate._smoke_validate_candidate(
        module,
        6,
        contract_version=run_spec.R4_CONTRACT,
    )
    result = module.make_offspring(
        jnp.asarray([2.0, 3.0]),
        jnp.zeros(3),
        jnp.zeros(6),
        jnp.zeros((3, 6)),
        0.0,
    )
    assert np.array_equal(np.asarray(result), [2.0, 3.0, 5.0, -2.0, -3.0, -5.0])

    power = _write_r4_candidate(tmp_path, "parent_genome_summary[0] ** 2")
    with pytest.raises(evaluate.CandidateValidationError, match="Pow"):
        evaluate._load_candidate(
            power,
            run_spec.R4_CONTRACT,
            Path(evaluate.__file__).with_name("initial_r4.py"),
        )
