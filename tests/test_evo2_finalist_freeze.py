from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from examples.evo2_ecosystem import freeze_finalist


def _add_generation(
    root: Path,
    generation: int,
    source: bytes,
    score: float,
    *,
    correct: bool = True,
) -> None:
    directory = root / f"gen_{generation}"
    results = directory / "results"
    results.mkdir(parents=True)
    (directory / "main.py").write_bytes(source)
    source_hash = hashlib.sha256(source).hexdigest()
    (results / "correct.json").write_text(
        json.dumps({"correct": correct, "error": None}),
        encoding="utf-8",
    )
    (results / "metrics.json").write_text(
        json.dumps(
            {
                "combined_score": score,
                "public": {"score": score},
                "private": {
                    "candidate_sha256": source_hash,
                    "manifest_sha256": "1" * 64,
                    "evaluator_source_sha256": "e" * 64,
                    "simulator_source_sha256": "s" * 64,
                    "simulator_config_sha256": "c" * 64,
                    "full_evaluation_performed": correct,
                    "numerical_repeats": 3 if correct else 0,
                    "repeat_scores": [score, score, score] if correct else [],
                    "selected_repeat_index": 1 if correct else None,
                    "integrity_valid": correct,
                },
            }
        ),
        encoding="utf-8",
    )


def test_discovery_ranks_correct_unique_sources(tmp_path: Path) -> None:
    _add_generation(tmp_path, 0, b"same", 0.2)
    _add_generation(tmp_path, 1, b"same", 0.8)
    _add_generation(tmp_path, 2, b"different", 0.7)
    _add_generation(tmp_path, 3, b"incorrect", 0.9, correct=False)
    (tmp_path / "gen_4").mkdir()

    candidates = freeze_finalist.discover_candidates(tmp_path)

    assert [candidate.generation for candidate in candidates] == [2, 0]
    assert [candidate.training_score for candidate in candidates] == [0.7, 0.2]


def test_discovery_rejects_a_stale_source_hash(tmp_path: Path) -> None:
    _add_generation(tmp_path, 1, b"candidate", 0.8)
    metrics_path = tmp_path / "gen_1" / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["private"]["candidate_sha256"] = "0" * 64
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        freeze_finalist.discover_candidates(tmp_path)


def test_archive_database_exports_reviewable_parent_lineage(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    database = sqlite3.connect(results / "programs.sqlite")
    database.execute(
        """CREATE TABLE programs (
            id TEXT, code TEXT, parent_id TEXT, archive_inspiration_ids TEXT,
            top_k_inspiration_ids TEXT, generation INTEGER, code_diff TEXT,
            combined_score REAL, public_metrics TEXT, private_metrics TEXT,
            complexity REAL, correct BOOLEAN, metadata TEXT, island_idx INTEGER
        )"""
    )
    parent_code = "parent\n"
    child_code = "child\n"
    rows = (
        (
            "parent",
            parent_code,
            None,
            "[]",
            "[]",
            0,
            "",
            0.1,
            "{}",
            json.dumps(
                {"candidate_sha256": hashlib.sha256(parent_code.encode()).hexdigest()}
            ),
            0.1,
            1,
            "{}",
            0,
        ),
        (
            "child",
            child_code,
            "parent",
            '["parent"]',
            "[]",
            1,
            "diff",
            0.2,
            '{"score": 0.2}',
            json.dumps(
                {"candidate_sha256": hashlib.sha256(child_code.encode()).hexdigest()}
            ),
            0.2,
            1,
            '{"patch_name": "adaptive"}',
            1,
        ),
    )
    database.executemany(
        "INSERT INTO programs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    database.commit()
    database.close()

    output = tmp_path / "lineage.json"
    record = freeze_finalist.export_archive_lineage(results, output)
    assert record["program_count"] == 2
    assert record["correct_program_count"] == 2
    assert record["programs"][1]["parent_id"] == "parent"
    assert record["programs"][1]["patch"]["patch_name"] == "adaptive"
    assert freeze_finalist.export_archive_lineage(results, output) == record


def _episode() -> SimpleNamespace:
    return SimpleNamespace(
        primary_score=np.asarray(0.5),
        post_event_productivity=np.asarray([0.25, 0.75]),
        survived=np.asarray(True),
        final_alive=np.asarray(8),
        minimum_population=np.asarray(3),
        maximum_generation=np.asarray(4),
        birth_count=np.asarray(9),
        natural_death_count=np.asarray(2),
        operator_counts=np.asarray([0, 9, 0, 0]),
        policy_violation_count=np.asarray(0),
        integrity_valid=np.asarray(True),
        event_record=SimpleNamespace(
            catastrophe_death_count=np.asarray(1),
            targeted_founder_lineage=np.asarray(-1),
        ),
    )


def _spec() -> dict:
    return {
        "run_id": "test-run",
        "generations": 3,
        "top_k": 3,
        "numerical_repeats": 3,
        "manifest_sha256": {
            "training_stable": "1" * 64,
            "training_punctuated": "1" * 64,
            "development": "d" * 64,
            "sealed": "f" * 64,
        },
        "source_sha256": {
            "initial": "i" * 64,
            "evaluator": "e" * 64,
            "simulator": "s" * 64,
            "analysis": "a" * 64,
            "baseline": "b" * 64,
            "dependency_lock": "6" * 64,
        },
        "simulator_config_sha256": "c" * 64,
        "archive": {"num_islands": 2},
        "baselines": [
            "clone",
            "fixed_parametric",
            "fixed_mixed",
            "stress_responsive",
        ],
        "bootstrap": {"replicates": 10_000, "confidence": 0.95, "seed": 7},
    }


def test_arm_audit_requires_complete_protocol_matched_budget(tmp_path: Path) -> None:
    spec = _spec()
    results = tmp_path / "stable"
    for generation in range(spec["generations"]):
        _add_generation(
            results,
            generation,
            f"candidate-{generation}".encode(),
            0.2,
            correct=generation < 2,
        )
    (results / "launch.json").write_text("launch", encoding="utf-8")
    database = sqlite3.connect(results / "programs.sqlite")
    database.execute("CREATE TABLE programs (correct BOOLEAN)")
    database.executemany("INSERT INTO programs VALUES (?)", [(1,), (1,), (0,)])
    database.commit()
    database.close()
    database_path = results / "programs.sqlite"
    completion = {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "regime": "stable",
        "run_spec_sha256": freeze_finalist.run_spec.sha256_bytes(
            freeze_finalist.run_spec.canonical_json_bytes(spec)
        ),
        "completed_evaluation_count": spec["generations"],
        "database_sha256": freeze_finalist.run_spec.sha256_file(database_path),
    }
    (tmp_path / "stable.complete.json").write_bytes(
        freeze_finalist.run_spec.canonical_json_bytes(completion)
    )
    paths = freeze_finalist.run_spec.RunPaths(
        run_root=tmp_path,
        arm_results=results,
        selection=tmp_path / "selection",
        frozen=tmp_path / "frozen",
    )

    counts = freeze_finalist._audit_arm(spec, "stable", paths)

    assert counts["completed_evaluation_count"] == 3
    assert counts["correct_evaluation_count"] == 2
    assert counts["full_evaluation_count"] == 2
    assert counts["invalid_proposal_count"] == 1
    assert counts["archive_program_count"] == 3
    assert counts["archive_correct_program_count"] == 2


def test_top_k_is_development_ranked_and_frozen_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec()
    spec_raw = freeze_finalist.run_spec.canonical_json_bytes(spec)
    spec_hash = hashlib.sha256(spec_raw).hexdigest()
    run_root = tmp_path / "results" / spec["run_id"]
    results_dir = run_root / "punctuated"
    selection_dir = run_root / "selection" / "punctuated"
    frozen_dir = tmp_path / "frozen" / spec["run_id"] / "punctuated"
    run_root.mkdir(parents=True)
    (run_root / "run_spec.json").write_bytes(spec_raw)
    (run_root / "run_spec.sha256").write_text(
        f"{spec_hash}  run_spec.json\n", encoding="utf-8"
    )
    sources = {1: b"candidate-one\n", 2: b"candidate-two\n", 3: b"candidate-three\n"}
    for generation, training_score in ((1, 0.9), (2, 0.8), (3, 0.7)):
        _add_generation(
            results_dir,
            generation,
            sources[generation],
            training_score,
        )

    validation_order: list[int] = []
    evaluation_order: list[int] = []

    def load_candidate(path: Path) -> ModuleType:
        module = ModuleType(f"candidate_{path.parent.name}")
        module.generation = int(path.parent.name.removeprefix("gen_"))
        return module

    def smoke_validate(module: ModuleType) -> None:
        validation_order.append(module.generation)

    def build_policy(module: ModuleType, mutate_cppn):
        assert mutate_cppn == "trusted-mutation"
        return module.generation

    manifest = SimpleNamespace(
        partition="development",
        worlds=(
            SimpleNamespace(
                scenario_id="development-1",
                pair_id="pair-1",
                scenario_family="resource_relocation",
                world_seed=21,
                event_kind=SimpleNamespace(value="resource_relocation"),
            ),
        ),
    )
    development_scores = {1: (0.1, 0.2, 0.3), 2: (0.7, 0.8, 0.9), 3: (0.3, 0.4, 0.5)}

    def evaluate_manifest(loaded_manifest, config, policy, *, numerical_repeats):
        assert loaded_manifest is manifest
        assert isinstance(config, Config)
        assert numerical_repeats == 3
        evaluation_order.append(policy)
        return SimpleNamespace(
            episodes=(_episode(),),
            candidate_score=np.asarray(development_scores[policy][1]),
            integrity_valid=np.asarray(True),
            repeat_scores=np.asarray(development_scores[policy]),
            selected_repeat_index=np.asarray(1),
        )

    @dataclass(frozen=True)
    class Config:
        fixed: int = 1

    development_loads: list[str] = []

    def load_development_manifest():
        development_loads.append("development")
        return manifest

    monkeypatch.setattr(
        freeze_finalist.task_evaluator, "_load_candidate", load_candidate
    )
    monkeypatch.setattr(
        freeze_finalist.task_evaluator,
        "_smoke_validate_candidate",
        smoke_validate,
    )
    monkeypatch.setattr(freeze_finalist.task_evaluator, "_build_policy", build_policy)
    monkeypatch.setattr(
        freeze_finalist,
        "_dependencies",
        lambda: {
            "SimulatorConfig": Config,
            "evaluate_manifest": evaluate_manifest,
            "load_development_manifest": load_development_manifest,
            "manifest_sha256": lambda value: "d" * 64,
            "mutate_cppn": "trusted-mutation",
            "simulator_config_sha256": lambda value: "c" * 64,
            "simulator_source_sha256": lambda: "s" * 64,
        },
    )

    paths = freeze_finalist.run_spec.RunPaths(
        run_root=run_root,
        arm_results=results_dir,
        selection=selection_dir,
        frozen=frozen_dir,
    )
    monkeypatch.setattr(freeze_finalist.run_spec, "paths_for", lambda *_: paths)
    monkeypatch.setattr(
        freeze_finalist,
        "audit_matched_run",
        lambda *_: {
            "stable": {"database_sha256": "4" * 64},
            "punctuated": {"database_sha256": "5" * 64},
        },
    )
    monkeypatch.setattr(
        freeze_finalist,
        "_provenance",
        lambda *_: {
            "evaluator": "e" * 64,
            "simulator": "s" * 64,
            "preregistration": "p" * 64,
        },
    )

    def export_archive_lineage(_results, output):
        archive = {
            "schema_version": 1,
            "program_count": 4,
            "correct_program_count": 3,
            "programs": [],
        }
        freeze_finalist._write_atomic(output, freeze_finalist._json_bytes(archive))
        return archive

    monkeypatch.setattr(
        freeze_finalist,
        "export_archive_lineage",
        export_archive_lineage,
    )

    record = freeze_finalist.select_and_freeze(
        "punctuated",
        spec=spec,
        spec_hash=spec_hash,
        spec_raw=spec_raw,
    )

    assert development_loads == ["development"]
    assert validation_order == [1, 2, 3]
    assert evaluation_order == [1, 2, 3]
    assert record["selected_generation"] == 2
    assert record["development_score"] == pytest.approx(0.8)
    assert record["numerical_repeats"] == 3
    assert record["repeat_scores"] == [0.7, 0.8, 0.9]
    assert record["selected_repeat_index"] == 1
    assert (frozen_dir / "main.py").read_bytes() == sources[2]
    assert (frozen_dir / "archive_lineage.json").is_file()
    assert (
        json.loads((frozen_dir / "freeze_record.json").read_text(encoding="utf-8"))
        == record
    )

    evaluation_files = sorted(selection_dir.glob("*.json"))
    assert len(evaluation_files) == 4
    evaluation_path = next(
        path for path in evaluation_files if path.name.startswith("gen_")
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["training_metrics"]["private"]["manifest_sha256"] == "1" * 64
    assert evaluation["development_manifest_sha256"] == "d" * 64
    assert evaluation["simulator_config_sha256"] == "c" * 64
    assert evaluation["numerical_repeats"] == 3
    assert evaluation["episodes"][0]["post_event_productivity"] == [0.25, 0.75]

    assert (
        freeze_finalist.select_and_freeze(
            "punctuated",
            spec=spec,
            spec_hash=spec_hash,
            spec_raw=spec_raw,
        )
        == record
    )
    assert evaluation_order == [1, 2, 3]


def test_wrong_manifest_partition_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec()
    spec_raw = freeze_finalist.run_spec.canonical_json_bytes(spec)
    spec_hash = hashlib.sha256(spec_raw).hexdigest()
    run_root = tmp_path / "run"
    results = run_root / "stable"
    for generation in range(3):
        _add_generation(results, generation, f"candidate-{generation}".encode(), 0.5)
    run_root.mkdir(exist_ok=True)
    (run_root / "run_spec.json").write_bytes(spec_raw)
    (run_root / "run_spec.sha256").write_text(
        f"{spec_hash}  run_spec.json\n", encoding="utf-8"
    )

    @dataclass(frozen=True)
    class Config:
        fixed: int = 1

    paths = freeze_finalist.run_spec.RunPaths(
        run_root=run_root,
        arm_results=results,
        selection=run_root / "selection" / "stable",
        frozen=tmp_path / "frozen",
    )
    monkeypatch.setattr(freeze_finalist.run_spec, "paths_for", lambda *_: paths)
    monkeypatch.setattr(freeze_finalist, "audit_matched_run", lambda *_: {})
    monkeypatch.setattr(freeze_finalist, "_provenance", lambda *_: {})

    monkeypatch.setattr(
        freeze_finalist,
        "_dependencies",
        lambda: {
            "SimulatorConfig": Config,
            "load_development_manifest": lambda: SimpleNamespace(partition="training"),
        },
    )
    monkeypatch.setattr(
        freeze_finalist,
        "export_archive_lineage",
        lambda _results, output: (
            output.parent.mkdir(parents=True, exist_ok=True),
            output.write_text("{}", encoding="utf-8"),
            {"program_count": 1},
        )[-1],
    )
    with pytest.raises(RuntimeError, match="wrong partition"):
        freeze_finalist.select_and_freeze(
            "stable",
            spec=spec,
            spec_hash=spec_hash,
            spec_raw=spec_raw,
        )
