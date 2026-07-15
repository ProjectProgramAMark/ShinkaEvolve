from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from examples.evo2_ecosystem import program_lineage, r4_selection, run_evo, run_spec


def _evaluator():
    path = Path(__file__).resolve().parents[1] / "examples/evo2_ecosystem/evaluate.py"
    spec = importlib.util.spec_from_file_location("evo2_r4_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_r4_candidate(tmp_path: Path, body: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    initial = (root / "examples/evo2_ecosystem/initial_r4.py").read_text()
    prefix, rest = initial.split("# EVOLVE-BLOCK-START", 1)
    _, suffix = rest.split("# EVOLVE-BLOCK-END", 1)
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        f"{prefix}# EVOLVE-BLOCK-START\n{body}\n# EVOLVE-BLOCK-END{suffix}"
    )
    return candidate


def _binding(path: str, digit: str) -> dict[str, str]:
    return {"path": path, "sha256": digit * 64}


def _v3_spec() -> dict:
    pilot, _, _ = run_spec.load_run_spec()
    initial_path = (
        Path(__file__).resolve().parents[1] / "examples/evo2_ecosystem/initial_r4.py"
    )
    initial_hash = run_spec.sha256_file(initial_path)
    sources = {
        name: f"{index:x}" * 64
        for index, name in enumerate(sorted(run_spec._V3_SOURCE_KEYS), start=1)
    }
    sources["initial"] = initial_hash
    return {
        **{
            key: value
            for key, value in pilot.items()
            if key not in {"manifest_sha256", "source_sha256", "schema_version"}
        },
        "schema_version": 3,
        "run_id": "evo2-r4-contract-test",
        "generations": 51,
        "proposal_budget": 50,
        "top_k": 5,
        "archive": {
            "archive_size": 32,
            "num_archive_inspirations": 1,
            "num_islands": 2,
            "num_top_k_inspirations": 1,
        },
        "baselines": [
            "clone",
            "parametric_conservative",
            "parametric_standard",
            "parametric_exploratory",
            "structural",
            "mixed",
            "human_stress",
            "human_credit",
            "structured_random",
        ],
        "manifests": {
            "training_stable": _binding("fixtures/train-stable.json", "1"),
            "training_punctuated": _binding("fixtures/train-punctuated.json", "2"),
            "development": _binding("fixtures/development.json", "3"),
            "sealed": _binding("fixtures/sealed.json", "4"),
        },
        "founder_index": _binding("fixtures/founders/index.json", "5"),
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
        "initial_program": {
            "path": "ShinkaEvolve/examples/evo2_ecosystem/initial_r4.py",
            "sha256": initial_hash,
        },
        "preregistration": _binding("microcosmos/docs/evo2/prereg.md", "6"),
        "implementation_plan": _binding("microcosmos/docs/evo2/plan.md", "7"),
        "artifact_roots": {
            "results": "ShinkaEvolve/examples/evo2_ecosystem/results",
            "frozen": "ShinkaEvolve/examples/evo2_ecosystem/frozen",
        },
        "source_sha256": sources,
    }


def test_schema_v3_freezes_r4_abi_and_exact_proposal_budget(tmp_path: Path) -> None:
    spec = _v3_spec()
    path = tmp_path / "r4.json"
    raw = run_spec.canonical_json_bytes(spec)
    path.write_bytes(raw)
    digest = run_spec.sha256_bytes(raw)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
    loaded, _, _ = run_spec.load_run_spec(path)
    assert run_spec.candidate_contract_version(loaded) == run_spec.R4_CONTRACT
    assert loaded["generations"] == loaded["proposal_budget"] + 1 == 51
    assert run_spec.initial_program_path(loaded).name == "initial_r4.py"


def test_r4_candidate_has_five_arrays_six_logits_and_opaque_rng(tmp_path: Path) -> None:
    evaluator = _evaluator()
    initial = (
        Path(__file__).resolve().parents[1] / "examples/evo2_ecosystem/initial_r4.py"
    )
    module = evaluator._load_candidate(initial, run_spec.R4_CONTRACT, initial)
    evaluator._smoke_validate_candidate(
        module, 6, contract_version=run_spec.R4_CONTRACT
    )
    assert np.array_equal(
        np.asarray(
            module.make_offspring(
                jnp.zeros(2), jnp.zeros(3), jnp.zeros(6), jnp.zeros((3, 6)), 0.0
            )
        ),
        np.asarray([-8.0, -8.0, 8.0, -8.0, -8.0, -8.0]),
    )
    bad = _write_r4_candidate(
        tmp_path,
        "def make_offspring(parent_genome_summary, parent_stats, population_stats, "
        "operator_stats, rng):\n    return jnp.array([rng, 0, 0, 0, 0, 0])",
    )
    with pytest.raises(evaluator.CandidateValidationError, match="opaque"):
        evaluator._load_candidate(bad, run_spec.R4_CONTRACT, initial)


def test_r4_policy_delegates_public_abi_to_trusted_microcosmos_adapter(tmp_path: Path) -> None:
    evaluator = _evaluator()
    initial = (
        Path(__file__).resolve().parents[1] / "examples/evo2_ecosystem/initial_r4.py"
    )
    candidate = _write_r4_candidate(
        tmp_path,
        "def make_offspring(parent_genome_summary, parent_stats, population_stats, "
        "operator_stats, rng):\n"
        "    return jnp.array([parent_genome_summary[0], parent_stats[2], "
        "population_stats[2], operator_stats[0, 3], -8.0, 8.0])",
    )
    module = evaluator._load_candidate(candidate, run_spec.R4_CONTRACT, initial)
    captured = {}

    def trusted(logit_policy):
        captured["logit_policy"] = logit_policy
        return "trusted-policy"

    policy = evaluator._build_policy(
        module, trusted, contract_version=run_spec.R4_CONTRACT
    )
    assert policy == "trusted-policy"
    logits = captured["logit_policy"](
        jnp.asarray([0.4, 0.3], dtype=jnp.float32),
        jnp.asarray([0.8, 0.7, 0.6], dtype=jnp.float32),
        jnp.asarray([0.5, 0.4, -1.0, 0.2, 0.3, 0.1], dtype=jnp.float32),
        jnp.arange(18, dtype=jnp.float32).reshape(3, 6) / 10,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert np.allclose(logits, [0.4, 0.6, -1.0, 0.3, -8.0, 8.0])


def test_r4_candidate_allows_fixed_array_slices(tmp_path: Path) -> None:
    evaluator = _evaluator()
    initial = (
        Path(__file__).resolve().parents[1] / "examples/evo2_ecosystem/initial_r4.py"
    )
    candidate = _write_r4_candidate(
        tmp_path,
        "def make_offspring(parent_genome_summary, parent_stats, population_stats, "
        "operator_stats, rng):\n"
        "    success = operator_stats[:, 0]\n"
        "    return jnp.array([success[0], success[1], success[2], "
        "success[0], success[1], success[2]])",
    )
    module = evaluator._load_candidate(candidate, run_spec.R4_CONTRACT, initial)
    evaluator._smoke_validate_candidate(
        module, 6, contract_version=run_spec.R4_CONTRACT
    )


def _r4_episode() -> SimpleNamespace:
    return SimpleNamespace(
        birth_count=6,
        natural_death_count=2,
        survived=True,
        generation_gain=2,
        policy_violation_count=0,
        operator_counts=np.ones(6),
        pre_selection_probability_sum=np.array([0.1, 0.1, 0.5, 0.1, 0.1, 0.1]),
        post_selection_probability_sum=np.array([0.1, 0.2, 0.2, 0.3, 0.1, 0.1]),
        pre_selection_probability_count=1,
        post_selection_probability_count=1,
        operator_success_ema=np.full(6, 0.5),
        operator_usage_ema=np.full(6, 1 / 6),
        operator_evidence_ema=np.full(6, 0.2),
        integrity_valid=True,
    )


def test_stable_feedback_cannot_observe_executed_shock_results() -> None:
    evaluator = _evaluator()
    sham = _r4_episode()
    evaluation = SimpleNamespace(
        episodes=(sham,),
        sham_episodes=(sham,),
        ancestor_episodes=(_r4_episode(),),
        integrity_valid=True,
        candidate_score=0.03,
        repeat_scores=np.array([0.01, 0.03, 0.04]),
        selected_repeat_index=1,
        sham_auc_delta=np.array([0.02]),
        shock_auc_delta=np.array([0.99]),
    )
    stable = evaluator._episode_metrics_r4(evaluation, "stable")
    assert "shock_auc_delta" not in stable["public"]
    assert "shock" not in stable["text_feedback"]
    punctuated = evaluator._episode_metrics_r4(evaluation, "punctuated")
    assert punctuated["public"]["shock_auc_delta"] == pytest.approx(0.99)


def test_hidden_shock_failure_cannot_invalidate_stable_sham_score() -> None:
    evaluator = _evaluator()
    sham = _r4_episode()
    shock = _r4_episode()
    shock.survived = False
    shock.integrity_valid = False
    ancestor_sham = _r4_episode()
    ancestor_shock = _r4_episode()
    evaluation = SimpleNamespace(
        episodes=(sham, shock),
        sham_episodes=(sham,),
        ancestor_episodes=(ancestor_sham, ancestor_shock),
        integrity_valid=False,
        candidate_score=0.03,
        repeat_scores=np.array([0.01, 0.03, 0.04]),
        selected_repeat_index=1,
        sham_auc_delta=np.array([0.02]),
        shock_auc_delta=np.array([-0.99]),
    )

    stable = evaluator._episode_metrics_r4(evaluation, "stable")
    assert stable["combined_score"] == pytest.approx(0.03)
    assert stable["private"]["integrity_valid"] is True
    assert "shock" not in stable["text_feedback"]

    punctuated = evaluator._episode_metrics_r4(evaluation, "punctuated")
    assert punctuated["combined_score"] == -2.0
    assert punctuated["private"]["integrity_valid"] is False

    ancestor_sham.survived = False
    stable = evaluator._episode_metrics_r4(evaluation, "stable")
    assert stable["combined_score"] == -2.0
    assert stable["private"]["integrity_valid"] is False


def test_r4_task_message_names_credit_and_six_trusted_actions(tmp_path: Path) -> None:
    spec = _v3_spec()
    paths = run_spec.RunPaths(
        run_root=tmp_path / "run",
        arm_results=tmp_path / "run" / "stable",
        selection=tmp_path / "run" / "selection" / "stable",
        frozen=tmp_path / "frozen",
    )
    paths.run_root.mkdir(parents=True)
    (paths.run_root / "run_spec.json").write_bytes(run_spec.canonical_json_bytes(spec))
    runner = run_evo.build_runner("stable", spec, paths)
    assert "operator_stats, shape (3, 6)" in runner.evo_config.task_sys_msg
    assert "parametric-exploratory" in runner.evo_config.task_sys_msg
    assert Path(runner.evo_config.init_program_path).name == "initial_r4.py"


def test_adaptive_classification_uses_expected_probabilities_not_counts() -> None:
    observations = []
    for founder in ("f0", "f1", "f2", "f3"):
        observations.append(
            {
                "founder_id": founder,
                "pre_probability": [0.0, 0.45, 0.45, 0.05, 0.05, 0.0],
                "post_probability": [0.0, 0.05, 0.10, 0.75, 0.05, 0.05],
                "realized_counts": [100, 0, 0, 0, 0, 0],
            }
        )
    result = r4_selection.classify_adaptive(
        observations, bootstrap_replicates=200, bootstrap_seed=7
    )
    assert result.eligible
    assert result.total_variation >= 0.20
    finalists = r4_selection.select_finalists(
        [
            {
                "generation": 4,
                "development_score": 0.2,
                "development_integrity_valid": True,
                "adaptive_eligibility": {"eligible": False},
            },
            {
                "generation": 8,
                "development_score": 0.1,
                "development_integrity_valid": True,
                "adaptive_eligibility": {"eligible": True},
            },
        ]
    )
    assert finalists["unrestricted"]["generation"] == 4
    assert finalists["adaptive"]["generation"] == 8


def test_r4_lineage_paths_distinguish_unrestricted_and_adaptive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _v3_spec()
    spec["artifact_roots"] = {"results": "results", "frozen": "frozen"}
    profile = tmp_path / "profile.json"
    raw = run_spec.canonical_json_bytes(spec)
    digest = run_spec.sha256_bytes(raw)
    profile.write_bytes(raw)
    profile.with_suffix(".sha256").write_text(f"{digest}  {profile.name}\n")
    monkeypatch.setattr(run_spec, "PROJECT_ROOT", tmp_path)
    paths = run_spec.paths_for(spec, "punctuated")
    paths.run_root.mkdir(parents=True)
    (paths.run_root / "run_spec.json").write_bytes(raw)
    for finalist_type, generation in (("unrestricted", 4), ("adaptive", 8)):
        root = paths.frozen / finalist_type
        root.mkdir(parents=True)
        (root / "freeze_record.json").write_text(
            '{"selected_generation":' + str(generation) + "}\n"
        )
    captured = []

    def export(**kwargs):
        captured.append(kwargs)
        return {"selected_representatives": []}

    monkeypatch.setattr(program_lineage, "export_champion_lineage", export)
    program_lineage.export_for_regime(
        "punctuated",
        selected_run_spec_path=profile,
        finalist_type="adaptive",
    )
    assert captured[0]["frozen_source_path"] == paths.frozen / "adaptive" / "main.py"
    assert captured[0]["output_path"].name.endswith("_adaptive.json")
