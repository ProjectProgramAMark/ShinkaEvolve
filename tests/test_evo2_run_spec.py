from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sqlite3

import pytest

from examples.evo2_ecosystem import evaluate as task_evaluator, run_evo, run_spec


def _binding(path: str, digit: str) -> dict[str, str]:
    return {"path": path, "sha256": digit * 64}


def _schema_v2_spec() -> dict:
    pilot, _, _ = run_spec.load_run_spec()
    return {
        **{
            key: value
            for key, value in pilot.items()
            if key not in {"manifest_sha256", "source_sha256", "schema_version"}
        },
        "schema_version": 2,
        "run_id": "evo2-prospective-test",
        "manifests": {
            "training_stable": _binding("fixtures/training_stable.json", "1"),
            "training_punctuated": _binding("fixtures/training_punctuated.json", "2"),
            "development": _binding("fixtures/development.json", "3"),
            "sealed": _binding("fixtures/sealed.json", "4"),
        },
        "founder_index": _binding("fixtures/founders/index.json", "5"),
        "candidate_output_width": 6,
        "preregistration": _binding("plans/prospective.md", "6"),
        "implementation_plan": _binding("plans/implementation.md", "7"),
        "artifact_roots": {
            "results": "scratch/results",
            "frozen": "scratch/frozen",
        },
        "source_sha256": {
            name: f"{index:x}" * 64
            for index, name in enumerate(sorted(run_spec._V2_SOURCE_KEYS), start=1)
        },
    }


def _write_spec(path: Path, spec: dict) -> tuple[str, bytes]:
    raw = run_spec.canonical_json_bytes(spec)
    digest = run_spec.sha256_bytes(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest, raw


def test_checked_in_run_spec_is_canonical_and_pinned() -> None:
    spec, digest, raw = run_spec.load_run_spec()

    assert raw == run_spec.canonical_json_bytes(spec)
    assert digest == run_spec.sha256_bytes(raw)
    assert spec["generations"] == 15
    assert spec["top_k"] == 3
    assert spec["numerical_repeats"] == 3
    assert spec["headless_command"] == "npx -y @roberttlange/headless@0.4.0"
    assert spec["baselines"] == [
        "clone",
        "fixed_parametric",
        "fixed_mixed",
        "stress_responsive",
    ]
    assert spec["source_sha256"]["dependency_lock"] == run_spec.sha256_file(
        run_spec.DEPENDENCY_LOCK_PATH
    )


def test_schema_v2_profile_binds_artifacts_and_abi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _schema_v2_spec()
    profile_dir = tmp_path / "profiles"
    profile_path = profile_dir / "actuator.json"
    digest, raw = _write_spec(profile_path, spec)
    monkeypatch.setattr(run_spec, "PROFILE_DIR", profile_dir)
    monkeypatch.setattr(run_spec, "PROJECT_ROOT", tmp_path)

    loaded, loaded_digest, loaded_raw = run_spec.load_run_spec(profile="actuator")

    assert loaded == spec
    assert (loaded_digest, loaded_raw) == (digest, raw)
    assert run_spec.candidate_output_width(loaded) == 6
    assert run_spec.manifest_path(loaded, "sealed") == (
        tmp_path / "fixtures" / "sealed.json"
    )
    assert run_spec.founder_index_path(loaded) == (
        tmp_path / "fixtures" / "founders" / "index.json"
    )
    assert run_spec.holdout_paths(loaded) == (
        tmp_path / "fixtures" / "development.json",
        tmp_path / "fixtures" / "sealed.json",
    )
    paths = run_spec.paths_for(loaded, "stable")
    assert paths.run_root == tmp_path / "scratch" / "results" / loaded["run_id"]
    assert paths.frozen == (
        tmp_path / "scratch" / "frozen" / loaded["run_id"] / "stable"
    )


def test_schema_v2_rejects_unbound_or_escaping_artifacts(tmp_path: Path) -> None:
    spec = _schema_v2_spec()
    spec["manifests"]["sealed"] = {"path": "../sealed.json", "sha256": "4" * 64}
    path = tmp_path / "bad.json"
    _write_spec(path, spec)
    with pytest.raises(ValueError, match="project-relative"):
        run_spec.load_run_spec(path)


def test_schema_v2_profile_is_propagated_to_evaluator_without_candidate_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _schema_v2_spec()
    digest = run_spec.sha256_bytes(run_spec.canonical_json_bytes(spec))
    paths = run_spec.RunPaths(
        run_root=tmp_path / "run",
        arm_results=tmp_path / "run" / "stable",
        selection=tmp_path / "run" / "selection" / "stable",
        frozen=tmp_path / "frozen",
    )
    paths.run_root.mkdir(parents=True)
    (paths.run_root / "run_spec.json").write_bytes(run_spec.canonical_json_bytes(spec))
    monkeypatch.setattr(run_evo, "TASK_DIR", tmp_path)

    runner = run_evo.build_runner("stable", spec, paths, spec_hash=digest)

    assert runner.job_config.extra_cmd_args == {
        "training_regime": "stable",
        "run_spec_path": str((paths.run_root / "run_spec.json").resolve()),
        "run_spec_sha256": digest,
    }
    assert "exactly 6 finite scores" in runner.evo_config.task_sys_msg


def test_schema_v2_holdout_permissions_use_selected_manifest_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _schema_v2_spec()
    monkeypatch.setattr(run_spec, "PROJECT_ROOT", tmp_path)
    development, sealed = run_spec.holdout_paths(spec)
    for path in (development, sealed):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    founder_path = run_spec.founder_index_path(spec)
    assert founder_path is not None
    founder_path.parent.mkdir(parents=True, exist_ok=True)
    founders = []
    for ordinal, (founder_id, partition) in enumerate(
        (("train-00", "training"), ("dev-00", "development"), ("sealed-00", "sealed")),
        start=1,
    ):
        artifact = f"{founder_id}.npz"
        founders.append(
            {
                "artifact": artifact,
                "artifact_sha256": f"{ordinal:x}" * 64,
                "connection_genes_sha256": f"{ordinal + 3:x}" * 64,
                "controller_layout": "cppn-4x1-15n-30c-v1",
                "founder_id": founder_id,
                "node_genes_sha256": f"{ordinal + 6:x}" * 64,
                "partition": partition,
                "selection_rule": "test",
                "selection_seed": ordinal,
            }
        )
        (founder_path.parent / artifact).write_bytes(b"fixture")
    index_payload = json.dumps(
        {"founders": founders, "schema_version": 1},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    founder_path.write_bytes(index_payload + b"\n")
    spec["founder_index"]["sha256"] = hashlib.sha256(index_payload).hexdigest()
    with pytest.raises(RuntimeError, match="must be unreadable"):
        run_evo._require_holdouts_locked(spec)
    development.chmod(0o000)
    sealed.chmod(0o000)
    founder_holdouts = (
        founder_path.parent / "dev-00.npz",
        founder_path.parent / "sealed-00.npz",
    )
    try:
        with pytest.raises(RuntimeError, match="must be unreadable"):
            run_evo._require_holdouts_locked(spec)
        for path in founder_holdouts:
            path.chmod(0o000)
        run_evo._require_holdouts_locked(spec)
    finally:
        development.chmod(0o600)
        sealed.chmod(0o600)
        for path in founder_holdouts:
            path.chmod(0o600)


def test_schema_v2_training_can_authenticate_index_with_founder_holdouts_locked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import sys

    microcosmos_root = run_evo.MICROCOSMOS_ROOT
    if str(microcosmos_root) not in sys.path:
        sys.path.insert(0, str(microcosmos_root))
    from experiments.evo2_ecosystem.founder_artifacts import (
        FounderIndex,
        founder_index_sha256,
        load_founder_artifact,
        load_founder_index,
        make_founder_record,
        write_founder_artifact,
        write_founder_index,
    )
    from microcosmos.cppn import (
        CONNECTION_WEIGHT,
        CPPNGenome,
        canonical_cppn_genome,
    )

    spec = _schema_v2_spec()
    monkeypatch.setattr(run_spec, "PROJECT_ROOT", tmp_path)
    index_path = run_spec.founder_index_path(spec)
    assert index_path is not None
    index_path.parent.mkdir(parents=True)
    canonical = canonical_cppn_genome()
    records = []
    artifacts: dict[str, Path] = {}
    for offset, (founder_id, partition) in enumerate(
        (("train-00", "training"), ("dev-00", "development"), ("sealed-00", "sealed"))
    ):
        genome = CPPNGenome(
            canonical.node_genes,
            canonical.connection_genes.at[0, CONNECTION_WEIGHT].add(offset * 0.1),
        )
        artifact_name = f"{founder_id}.npz"
        artifact_path = index_path.parent / artifact_name
        artifacts[partition] = artifact_path
        records.append(
            make_founder_record(
                founder_id=founder_id,
                partition=partition,
                artifact=artifact_name,
                digests=write_founder_artifact(artifact_path, genome),
                selection_rule="test",
                selection_seed=offset,
            )
        )
    index = FounderIndex(founders=tuple(records))
    spec["founder_index"]["sha256"] = write_founder_index(index_path, index)

    artifacts["development"].chmod(0o000)
    artifacts["sealed"].chmod(0o000)
    try:
        # Index authentication does not touch any founder body.
        assert (
            task_evaluator._verified_founder_index_path(
                spec,
                {
                    "load_founder_index": load_founder_index,
                    "founder_index_sha256": founder_index_sha256,
                },
            )
            == index_path
        )
        # The launch guard derives only the two founder holdouts from that index.
        assert set(run_evo._founder_holdout_paths(spec)) == {
            artifacts["development"],
            artifacts["sealed"],
        }
        # Training remains available and is still fully authenticated.
        training = index.record("train-00", expected_partition="training")
        loaded = load_founder_artifact(
            index_path.parent,
            training,
            expected_partition="training",
        )
        assert loaded.node_genes.shape == canonical.node_genes.shape
    finally:
        artifacts["development"].chmod(0o600)
        artifacts["sealed"].chmod(0o600)


def test_arm_is_fresh_by_default_and_resume_requires_exact_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, digest, raw = run_spec.load_run_spec()
    paths = run_spec.RunPaths(
        run_root=tmp_path / "run",
        arm_results=tmp_path / "run" / "stable",
        selection=tmp_path / "run" / "selection" / "stable",
        frozen=tmp_path / "frozen",
    )
    monkeypatch.setattr(run_evo.run_spec, "paths_for", lambda *_: paths)
    monkeypatch.setattr(run_evo.sys, "argv", ["run_evo.py", "--regime", "stable"])

    assert run_evo._prepare_arm(spec, digest, raw, "stable", resume=False) == paths
    launch = json.loads((paths.arm_results / "launch.json").read_text())
    assert launch["run_spec_sha256"] == digest
    assert launch["headless_command"].endswith("@0.4.0")
    assert "MemoryMax=24G" in launch["canonical_command"]

    with pytest.raises(RuntimeError, match="nonempty"):
        run_evo._prepare_arm(spec, digest, raw, "stable", resume=False)

    run_evo._prepare_arm(spec, digest, raw, "stable", resume=True)
    assert (paths.arm_results / "resume_001.json").is_file()

    (paths.run_root / "run_spec.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        run_evo._prepare_arm(spec, digest, raw, "stable", resume=True)


def test_headless_command_cannot_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _, _ = run_spec.load_run_spec()
    monkeypatch.setenv(run_evo.HEADLESS_COMMAND_ENV, "npx headless@latest")
    with pytest.raises(RuntimeError, match="conflicts"):
        run_evo._pin_headless(spec)


def test_completion_hashes_checkpointed_database(tmp_path: Path) -> None:
    spec, spec_hash, _ = run_spec.load_run_spec()
    arm = tmp_path / "punctuated"
    arm.mkdir()
    for generation in range(spec["generations"]):
        generation_dir = arm / f"gen_{generation}"
        (generation_dir / "results").mkdir(parents=True)
        (generation_dir / "main.py").write_text("pass\n", encoding="utf-8")
        (generation_dir / "results" / "metrics.json").write_text("{}\n")
        (generation_dir / "results" / "correct.json").write_text("{}\n")

    database = arm / "programs.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE programs (id INTEGER PRIMARY KEY)")
    connection.execute("INSERT INTO programs DEFAULT VALUES")
    connection.commit()

    paths = run_spec.RunPaths(
        run_root=tmp_path,
        arm_results=arm,
        selection=tmp_path / "selection",
        frozen=tmp_path / "frozen",
    )
    try:
        run_evo._write_completion(spec, spec_hash, "punctuated", paths)
    finally:
        connection.close()

    marker = json.loads((tmp_path / "punctuated.complete.json").read_text())
    assert marker["database_sha256"] == run_spec.sha256_file(database)


def test_second_arm_requires_completed_unreadable_sibling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec, digest, _ = run_spec.load_run_spec()
    run_root = tmp_path / "run"
    stable = run_spec.RunPaths(
        run_root=run_root,
        arm_results=run_root / "stable",
        selection=run_root / "selection" / "stable",
        frozen=tmp_path / "frozen" / "stable",
    )
    punctuated = run_spec.RunPaths(
        run_root=run_root,
        arm_results=run_root / "punctuated",
        selection=run_root / "selection" / "punctuated",
        frozen=tmp_path / "frozen" / "punctuated",
    )
    paths = {"stable": stable, "punctuated": punctuated}
    monkeypatch.setattr(run_evo.run_spec, "paths_for", lambda _spec, arm: paths[arm])
    stable.arm_results.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="has not completed"):
        run_evo._require_sibling_isolated(spec, digest, "punctuated")

    marker = {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "regime": "stable",
        "run_spec_sha256": digest,
        "completed_evaluation_count": spec["generations"],
        "database_sha256": "0" * 64,
    }
    (run_root / "stable.complete.json").write_bytes(
        run_spec.canonical_json_bytes(marker)
    )
    with pytest.raises(RuntimeError, match="must be unreadable"):
        run_evo._require_sibling_isolated(spec, digest, "punctuated")

    stable.arm_results.chmod(0o000)
    try:
        run_evo._require_sibling_isolated(spec, digest, "punctuated")
    finally:
        stable.arm_results.chmod(0o700)
