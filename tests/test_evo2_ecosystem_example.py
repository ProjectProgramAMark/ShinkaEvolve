from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


def _load_evaluator():
    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "evo2_ecosystem" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evo2_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_launcher():
    from examples.evo2_ecosystem import run_evo

    return run_evo


def _write_candidate(tmp_path: Path, body: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    initial = (root / "examples" / "evo2_ecosystem" / "initial.py").read_text(
        encoding="utf-8"
    )
    prefix, remainder = initial.split("# EVOLVE-BLOCK-START", maxsplit=1)
    _, suffix = remainder.split("# EVOLVE-BLOCK-END", maxsplit=1)
    path = tmp_path / "candidate.py"
    path.write_text(
        prefix + "# EVOLVE-BLOCK-START" + "\n" + body + "\n# EVOLVE-BLOCK-END" + suffix,
        encoding="utf-8",
    )
    return path


def test_valid_candidate_is_jax_compatible(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    path = _write_candidate(
        tmp_path,
        """def make_offspring(parent_genome, parent_stats, population_stats, rng):
    stress = jnp.clip(1.0 - population_stats[0], 0.0, 1.0)
    return jnp.array([0.0, 1.0 - stress, stress, parent_genome[0]])
""",
    )
    module = evaluator._load_candidate(path)
    evaluator._smoke_validate_candidate(module)


def test_candidate_output_width_is_selected_by_trusted_protocol(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    path = _write_candidate(
        tmp_path,
        "def make_offspring(parent_genome, parent_stats, population_stats, rng):\n"
        "    return jnp.array([0.0, 1.0, 0.0, 0.0, 0.5, -0.5])\n",
    )
    module = evaluator._load_candidate(path)
    evaluator._smoke_validate_candidate(module, output_width=6)
    with pytest.raises(evaluator.CandidateValidationError, match="exactly 4"):
        evaluator._smoke_validate_candidate(module)


def test_generated_candidate_may_omit_only_the_final_newline(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    path = _write_candidate(
        tmp_path,
        "def make_offspring(parent_genome, parent_stats, population_stats, rng):\n"
        "    return jnp.array([0.0, 1.0, 0.0, 0.0])\n",
    )
    path.write_text(path.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    evaluator._smoke_validate_candidate(evaluator._load_candidate(path))


def test_candidate_executes_the_already_validated_bytes(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    path = _write_candidate(
        tmp_path,
        "def make_offspring(parent_genome, parent_stats, population_stats, rng):\n"
        "    return jnp.array([0.0, 1.0, 0.0, 0.0])\n",
    )
    module = evaluator._load_candidate(path)
    path.write_text("invalid after validation", encoding="utf-8")
    result = module.make_offspring(
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.zeros(4),
        jnp.zeros(()),
    )
    assert np.array_equal(np.asarray(result), np.asarray([0.0, 1.0, 0.0, 0.0]))


def test_sensitive_environment_is_removed_before_candidate_execution(
    monkeypatch,
) -> None:
    evaluator = _load_evaluator()
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent")
    monkeypatch.setenv("XLA_FLAGS", "--safe")
    evaluator._scrub_sensitive_environment()
    assert "OPENAI_API_KEY" not in evaluator.os.environ
    assert "SSH_AUTH_SOCK" not in evaluator.os.environ
    assert evaluator.os.environ["XLA_FLAGS"] == "--safe"


def test_launcher_fails_closed_until_holdouts_are_unreadable(
    monkeypatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    development = tmp_path / "development.json"
    sealed = tmp_path / "sealed.json"
    development.write_text("{}", encoding="utf-8")
    sealed.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(launcher, "HOLDOUT_PATHS", (development, sealed))
    with pytest.raises(RuntimeError, match="must be unreadable"):
        launcher._require_holdouts_locked()
    development.chmod(0o000)
    sealed.chmod(0o000)
    launcher._require_holdouts_locked()


@pytest.mark.parametrize(
    "body",
    [
        "import os\ndef make_offspring(parent_genome, parent_stats, population_stats, rng):\n    return [1, 0, 0, 0]\n",
        "def make_offspring(parent_genome, parent_stats, population_stats, rng):\n    return open('/tmp/x')\n",
        "def make_offspring(parent_genome, parent_stats, population_stats, rng):\n    return parent_stats.shape\n",
        "def make_offspring(parent_genome, parent_stats, population_stats, rng):\n    while True:\n        pass\n",
        "def make_offspring(parent_genome, parent_stats, population_stats, rng):\n    return jnp.array([getattr(jnp, 'array'), 0, 0, 0])\n",
        "def make_offspring(parent_genome, parent_stats, population_stats, rng):\n    return jnp.array([rng, 0, 0, 0])\n",
    ],
)
def test_ast_rejects_ambient_capabilities(tmp_path: Path, body: str) -> None:
    evaluator = _load_evaluator()
    path = _write_candidate(tmp_path, body)
    with pytest.raises(evaluator.CandidateValidationError):
        evaluator._load_candidate(path)


def test_immutable_task_code_cannot_change(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    path = _write_candidate(
        tmp_path,
        "def make_offspring(parent_genome, parent_stats, population_stats, rng):\n"
        "    return jnp.array([0.0, 0.0, 0.0, 1.0])\n",
    )
    source = path.read_text(encoding="utf-8").replace(
        "import jax.numpy as jnp", "import jax.numpy as jnp  # changed"
    )
    path.write_text(source, encoding="utf-8")
    with pytest.raises(
        evaluator.CandidateValidationError,
        match="immutable task code",
    ):
        evaluator._load_candidate(path)


@pytest.mark.parametrize(
    "expression",
    [
        "jnp.array([1.0, 2.0, 3.0])",
        "jnp.array([1.0, jnp.exp(1000.0), 0.0, 0.0])",
    ],
)
def test_invalid_outputs_are_rejected(tmp_path: Path, expression: str) -> None:
    evaluator = _load_evaluator()
    path = _write_candidate(
        tmp_path,
        "def make_offspring(parent_genome, parent_stats, population_stats, rng):\n"
        f"    return {expression}\n",
    )
    module = evaluator._load_candidate(path)
    with pytest.raises(evaluator.CandidateValidationError):
        evaluator._smoke_validate_candidate(module)


def test_errors_are_sanitized() -> None:
    evaluator = _load_evaluator()
    assert evaluator._sanitize_error(TimeoutError("secret /path")) == (
        "timeout",
        "candidate evaluation timed out",
    )
    code, message = evaluator._sanitize_error(
        RuntimeError("token=secret /home/private/file")
    )
    assert code == "evaluation_failed"
    assert "secret" not in message
    assert "/home" not in message


@dataclass
class _Episode:
    primary_score: float = 0.75
    birth_count: int = 4
    final_alive: int = 3
    survived: bool = True
    policy_violation_count: int = 0
    operator_counts: np.ndarray = None

    def __post_init__(self):
        if self.operator_counts is None:
            self.operator_counts = np.array([1, 1, 1, 1])


def test_evaluator_routes_only_through_training_loader(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator = _load_evaluator()
    candidate = _write_candidate(
        tmp_path,
        "def make_offspring(parent_genome, parent_stats, population_stats, rng):\n"
        "    return jnp.array([0.0, 0.0, 0.0, 1.0])\n",
    )
    calls = []

    class Config:
        pass

    def load_training_manifest(regime):
        calls.append(("load", regime))
        return SimpleNamespace(partition="training")

    def evaluate_manifest(manifest, config, policy):
        calls.append(("evaluate", manifest.partition, type(config).__name__))
        return SimpleNamespace(
            episodes=(_Episode(),),
            candidate_score=jnp.array(0.75),
            integrity_valid=jnp.array(True),
        )

    monkeypatch.setattr(
        evaluator,
        "_imports",
        lambda: {
            "SimulatorConfig": Config,
            "evaluate_manifest": evaluate_manifest,
            "load_training_manifest": load_training_manifest,
            "manifest_sha256": lambda manifest: "a" * 64,
            "mutate_cppn": lambda *args: args,
            "simulator_config_sha256": lambda config: "c" * 64,
            "simulator_source_sha256": lambda: "b" * 64,
        },
    )
    metrics = evaluator._evaluate_candidate(candidate, "punctuated")
    assert calls == [
        ("load", "punctuated"),
        ("evaluate", "training", "Config"),
    ]
    assert metrics["combined_score"] == 0.75
    assert metrics["private"]["manifest_sha256"] == "a" * 64
    assert metrics["private"]["simulator_config_sha256"] == "c" * 64
    assert "world_scores" not in metrics["private"]


def test_main_writes_shinka_contract_and_sanitizes_failure(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator = _load_evaluator()
    monkeypatch.setattr(
        evaluator,
        "_evaluate_candidate",
        lambda *_: (_ for _ in ()).throw(RuntimeError("secret path")),
    )
    evaluator.main("candidate.py", str(tmp_path), "stable")
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    correct = json.loads((tmp_path / "correct.json").read_text(encoding="utf-8"))
    assert metrics["combined_score"] == 0.0
    assert metrics["private"]["error_code"] == "evaluation_failed"
    assert metrics["private"]["full_evaluation_performed"] is False
    assert metrics["private"]["numerical_repeats"] == 0
    assert metrics["private"]["repeat_scores"] == []
    assert metrics["private"]["selected_repeat_index"] is None
    assert correct == {
        "correct": False,
        "error": "candidate evaluation failed (RuntimeError)",
    }
