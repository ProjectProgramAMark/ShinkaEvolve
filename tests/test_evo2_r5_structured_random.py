from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.evo2_ecosystem import freeze_finalist, run_evo, run_spec


def _spec() -> dict:
    return {
        "schema_version": 4,
        "run_id": "r5-test",
        "top_k": 5,
        "numerical_repeats": 3,
        "simulator_config_sha256": "c" * 64,
        "source_sha256": {
            "initial": "i" * 64,
            "evaluator": "e" * 64,
            "simulator": "s" * 64,
            "baseline": "b" * 64,
        },
        "manifests": {
            "training_punctuated": {"path": "unused", "sha256": "1" * 64},
            "development": {"path": "unused", "sha256": "d" * 64},
            "sealed": {"path": "unused", "sha256": "f" * 64},
        },
        "founder_index": {"path": "unused", "sha256": "2" * 64},
        "initial_program": {"path": "unused", "sha256": "i" * 64},
        "candidate_output_width": 6,
        "candidate_contract": {"version": run_spec.R4_CONTRACT},
        "protocol_document": {"path": "unused", "sha256": "p" * 64},
        "archive": {"num_islands": 2},
        "bootstrap": {"confidence": 0.95, "replicates": 10_000, "seed": 7},
        "baselines": list(run_spec.R5_BASELINES),
    }


def _paths(tmp_path: Path) -> run_spec.StructuredRandomPaths:
    run_root = tmp_path / "results" / "r5-test"
    root = run_root / "structured_random"
    return run_spec.StructuredRandomPaths(
        run_root=run_root,
        control_root=root,
        roster=root / "roster",
        training=root / "training",
        selection=root / "selection",
        frozen=tmp_path / "frozen" / "r5-test" / "structured_random",
    )


def _roster() -> tuple[SimpleNamespace, ...]:
    records = []
    for generation in range(50):
        source = f"candidate-{generation}\n"
        records.append(
            SimpleNamespace(
                candidate_id=f"structured-{generation:03d}",
                family="threshold" if generation < 25 else "affine",
                source=source,
                sha256=hashlib.sha256(source.encode()).hexdigest(),
            )
        )
    return tuple(records)


def _baseline_module(roster: tuple[SimpleNamespace, ...]):
    def publish(destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        index = {
            "candidate_count": 50,
            "records": [
                {
                    "candidate_id": record.candidate_id,
                    "family": record.family,
                    "path": f"{record.candidate_id}.py",
                    "sha256": record.sha256,
                }
                for record in roster
            ],
        }
        raw = run_spec.canonical_json_bytes(index)
        (destination / "index.json").write_bytes(raw)
        (destination / "index.sha256").write_text(
            f"{hashlib.sha256(raw).hexdigest()}  index.json\n",
            encoding="ascii",
        )
        for record in roster:
            (destination / f"{record.candidate_id}.py").write_text(
                record.source,
                encoding="utf-8",
            )

    return SimpleNamespace(
        publish_structured_random_roster=publish,
        load_structured_random_roster=lambda _path: roster,
    )


def _stored_spec(paths: run_spec.StructuredRandomPaths, spec: dict) -> tuple[bytes, str]:
    raw = run_spec.canonical_json_bytes(spec)
    digest = hashlib.sha256(raw).hexdigest()
    paths.run_root.mkdir(parents=True)
    (paths.run_root / "run_spec.json").write_bytes(raw)
    (paths.run_root / "run_spec.sha256").write_text(
        f"{digest}  run_spec.json\n",
        encoding="utf-8",
    )
    return raw, digest


def test_structured_random_training_and_development_freeze_publish_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec()
    paths = _paths(tmp_path)
    spec_raw, spec_hash = _stored_spec(paths, spec)
    roster = _roster()
    baselines = _baseline_module(roster)
    monkeypatch.setattr(run_spec, "structured_random_paths", lambda _spec: paths)
    monkeypatch.setattr(
        freeze_finalist,
        "_load_r5_baselines",
        lambda _spec: baselines,
    )
    monkeypatch.setattr(
        freeze_finalist,
        "_verify_structured_training_inputs",
        lambda _spec: None,
    )

    training_calls: list[int] = []

    def evaluate_candidate(program_path: Path, **_kwargs):
        generation = int(program_path.stem.rsplit("-", 1)[1])
        training_calls.append(generation)
        expected = freeze_finalist._structured_expected_private(spec, spec_hash)
        return (
            {
                "combined_score": generation / 100.0,
                "public": {"score": generation / 100.0},
                "private": {
                    **expected,
                    "candidate_sha256": roster[generation].sha256,
                    "integrity_valid": True,
                    "numerical_repeats": 3,
                    "repeat_scores": [generation / 100.0] * 3,
                    "selected_repeat_index": 1,
                },
                "text_feedback": "complete",
            },
            True,
            None,
        )

    monkeypatch.setattr(
        freeze_finalist,
        "_evaluate_structured_source",
        evaluate_candidate,
    )

    completion = freeze_finalist.run_structured_random_training(
        spec=spec,
        spec_hash=spec_hash,
        spec_raw=spec_raw,
    )
    assert training_calls == list(range(50))
    assert completion["terminal_evaluation_count"] == 50
    assert completion["unique_valid_candidate_count"] == 50
    assert completion["passed"] is True
    assert (paths.control_root / "complete.json").is_file()

    # Authenticated resume must not grant replacement evaluations.
    assert freeze_finalist.run_structured_random_training(
        spec=spec,
        spec_hash=spec_hash,
        spec_raw=spec_raw,
    ) == completion
    assert training_calls == list(range(50))

    @dataclass(frozen=True)
    class Config:
        fixed: int = 1

    manifest = SimpleNamespace(partition="development", worlds=())
    monkeypatch.setattr(freeze_finalist, "audit_matched_run", lambda _spec: {})
    monkeypatch.setattr(
        freeze_finalist,
        "_dependencies",
        lambda: {
            "SimulatorConfig": Config,
            "manifest_sha256": lambda _manifest: "d" * 64,
            "simulator_config_sha256": lambda _config: "c" * 64,
        },
    )
    monkeypatch.setattr(
        freeze_finalist,
        "_load_development_manifest",
        lambda _spec, _dependencies: manifest,
    )
    monkeypatch.setattr(
        freeze_finalist,
        "_verified_founder_index_path",
        lambda _spec, _dependencies: None,
    )
    monkeypatch.setattr(
        freeze_finalist,
        "_provenance",
        lambda _spec, _dependencies: {"protocol_document": "p" * 64},
    )
    development_calls: list[int] = []

    def reevaluate(regime, candidate, *_args, **_kwargs):
        development_calls.append(candidate.generation)
        score = 1.0 - candidate.generation / 100.0
        return {
            "schema_version": 1,
            "regime": regime,
            "generation": candidate.generation,
            "candidate_source_sha256": candidate.source_sha256,
            "run_spec_sha256": spec_hash,
            "training_score": candidate.training_score,
            "training_metrics": candidate.training_metrics,
            "training_manifest_sha256": "1" * 64,
            "development_score": score,
            "development_integrity_valid": True,
            "numerical_repeats": 3,
            "repeat_scores": [score] * 3,
            "selected_repeat_index": 1,
            "development_manifest_sha256": "d" * 64,
            "simulator_config_sha256": "c" * 64,
            "episodes": [],
        }

    monkeypatch.setattr(freeze_finalist, "reevaluate_candidate", reevaluate)
    development = freeze_finalist.evaluate_structured_random_development(
        spec=spec,
        spec_hash=spec_hash,
        spec_raw=spec_raw,
    )
    assert development["complete"] is True
    assert not paths.frozen.exists()

    arm_selections = {
        regime: tmp_path / regime / "selection" for regime in run_spec.REGIMES
    }
    monkeypatch.setattr(
        run_spec,
        "paths_for",
        lambda _spec, regime: SimpleNamespace(selection=arm_selections[regime]),
    )
    arm_candidates: dict[str, list[freeze_finalist.Candidate]] = {}
    for regime, selection in arm_selections.items():
        selection.mkdir(parents=True)
        (selection / "archive_lineage.json").write_text(
            f"{regime}-archive\n",
            encoding="utf-8",
        )
        arm_candidates[regime] = []
        for generation in range(5):
            source_hash = hashlib.sha256(
                f"{regime}-{generation}".encode()
            ).hexdigest()
            candidate = freeze_finalist.Candidate(
                generation=generation,
                source_path=paths.roster / roster[generation].candidate_id,
                source_sha256=source_hash,
                training_score=float(generation),
                training_metrics={},
            )
            arm_candidates[regime].append(candidate)
            record = selection / f"gen_{generation}_{source_hash[:12]}.json"
            record.write_text("{}\n", encoding="utf-8")
        freeze_finalist._publish_development_completion(
            spec,
            spec_hash,
            regime,
            arm_candidates[regime],
            selection,
        )

    missing_record = arm_selections["punctuated"] / (
        f"gen_0_{arm_candidates['punctuated'][0].source_sha256[:12]}.json"
    )
    missing_payload = missing_record.read_bytes()
    missing_record.unlink()
    with pytest.raises(RuntimeError, match="does not authenticate"):
        freeze_finalist.freeze_structured_random_finalist(
            spec=spec,
            spec_hash=spec_hash,
            spec_raw=spec_raw,
        )
    assert development_calls == [49, 48, 47, 46, 45]
    assert not paths.frozen.exists()
    missing_record.write_bytes(missing_payload)
    frozen = freeze_finalist.freeze_structured_random_finalist(
        spec=spec,
        spec_hash=spec_hash,
        spec_raw=spec_raw,
    )
    assert development_calls == [49, 48, 47, 46, 45]
    assert frozen["selection_rule"]["top_k"] == 5
    assert frozen["selected_generation"] == 45
    assert frozen["protocol_document_sha256"] == "p" * 64
    assert "preregistration_sha256" not in frozen
    assert (paths.frozen / "main.py").read_bytes() == roster[45].source.encode()
    assert json.loads(
        (paths.frozen / "freeze_record.json").read_text(encoding="utf-8")
    ) == frozen


def test_schema_v4_protocol_freeze_fields_replace_legacy_documents() -> None:
    fields = freeze_finalist._protocol_freeze_fields(_spec(), {})
    assert fields == {
        "protocol_document_sha256": "p" * 64,
        "protocol_document_complete": True,
    }


def test_structured_training_reuses_the_r5_search_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        run_evo,
        "_require_holdouts_locked",
        lambda spec: calls.append(("holdouts", spec)),
    )
    monkeypatch.setattr(
        run_evo,
        "_verify_search_inputs",
        lambda spec, regime: calls.append((regime, spec)),
    )
    spec = _spec()

    freeze_finalist._verify_structured_training_inputs(spec)

    assert calls == [("holdouts", spec), ("punctuated", spec)]


def test_each_structured_source_uses_a_fresh_existing_evaluator_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate.py"
    source.write_text("candidate\n", encoding="utf-8")
    stored_spec = tmp_path / "run_spec.json"
    stored_spec.write_text("{}\n", encoding="utf-8")
    spec = {"search": {"evaluation_timeout": "00:10:00"}}
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 600
        results = Path(command[command.index("--results_dir") + 1])
        results.mkdir(parents=True)
        (results / "metrics.json").write_text(
            json.dumps({"combined_score": 0.5, "private": {"integrity_valid": True}}),
            encoding="utf-8",
        )
        (results / "correct.json").write_text(
            json.dumps({"correct": True, "error": None}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(freeze_finalist.subprocess, "run", run)
    for _ in range(2):
        metrics, correct, error = freeze_finalist._evaluate_structured_source(
            source,
            stored_spec=stored_spec,
            spec=spec,
            spec_hash="a" * 64,
        )
        assert metrics["combined_score"] == 0.5
        assert correct is True
        assert error is None

    assert len(commands) == 2
    assert all(command[0] == freeze_finalist.sys.executable for command in commands)
    assert all(
        command[1] == str(freeze_finalist.TASK_DIR / "evaluate.py")
        for command in commands
    )


def test_schema_v4_provenance_uses_bound_r5_analysis_and_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "r5_analysis.py"
    baseline = tmp_path / "r5_baselines.py"
    protocol = tmp_path / "protocol.md"
    sealed = tmp_path / "sealed.json"
    founder = tmp_path / "founders.json"
    for path in (analysis, baseline, protocol, sealed, founder):
        path.write_text(path.name, encoding="utf-8")
    sealed.with_suffix(".sha256").write_text(
        f"{'f' * 64}  sealed.json\n",
        encoding="utf-8",
    )
    spec = {
        "schema_version": 4,
        "source_sha256": {"analysis": "a" * 64, "baseline": "b" * 64},
        "protocol_document": {"path": "unused", "sha256": "p" * 64},
        "founder_index": {"path": "unused", "sha256": "2" * 64},
    }
    seen: list[Path] = []

    def sha256_file(path: Path) -> str:
        path = Path(path).resolve()
        seen.append(path)
        if path == analysis.resolve():
            return "a" * 64
        if path == baseline.resolve():
            return "b" * 64
        if path == protocol.resolve():
            return "p" * 64
        return "0" * 64

    monkeypatch.setattr(run_spec, "sha256_file", sha256_file)
    monkeypatch.setattr(
        run_spec,
        "protocol_tool_paths",
        lambda _spec: {"final_analysis": analysis},
    )
    monkeypatch.setattr(
        run_spec,
        "bound_protocol_paths",
        lambda _spec: {"protocol_document": protocol},
    )
    monkeypatch.setattr(run_spec, "manifest_path", lambda *_args: sealed)
    monkeypatch.setattr(run_spec, "manifest_hash", lambda *_args: "f" * 64)
    monkeypatch.setattr(run_spec, "founder_index_path", lambda _spec: founder)
    monkeypatch.setattr(run_spec, "initial_program_path", lambda _spec: tmp_path / "i")
    monkeypatch.setattr(run_spec, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_spec, "R5_BASELINE_SOURCE_PATH", "r5_baselines.py")
    monkeypatch.setattr(freeze_finalist.os, "access", lambda *_args: False)
    dependencies = {
        "simulator_source_sha256": lambda: "0" * 64,
        "load_founder_index": lambda *_args, **_kwargs: SimpleNamespace(founders=()),
        "founder_index_sha256": lambda _index: "2" * 64,
    }

    provenance = freeze_finalist._provenance(spec, dependencies)

    assert provenance == {
        "analysis": "a" * 64,
        "baseline": "b" * 64,
        "protocol_document": "p" * 64,
    }
    assert analysis.resolve() in seen
    assert baseline.resolve() in seen
    assert freeze_finalist.ANALYSIS_PATH.resolve() not in seen
    assert freeze_finalist.BASELINE_PATH.resolve() not in seen
