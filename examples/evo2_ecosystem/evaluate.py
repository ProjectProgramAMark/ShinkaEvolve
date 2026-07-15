"""Bounded ShinkaEvolve evaluator for Evo² heredity policies."""

from __future__ import annotations

import argparse
import ast
import hmac
import hashlib
from importlib import metadata
import math
import os
import platform
from pathlib import Path
import time
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from shinka.core.wrap_eval import save_json_results

try:
    from examples.evo2_ecosystem import run_spec
except ImportError:  # direct execution from the task directory
    import run_spec


TASK_DIR = Path(__file__).resolve().parent
MICROCOSMOS_ROOT = TASK_DIR.parents[2] / "microcosmos"

MAX_SOURCE_BYTES = 12_000
MAX_AST_NODES = 1_024
MAX_NONBLANK_LINES = 60
REGIMES = ("stable", "punctuated")
START_MARKER = "# EVOLVE-BLOCK-START"
END_MARKER = "# EVOLVE-BLOCK-END"

_SENSITIVE_ENV_FRAGMENTS = (
    "API_KEY",
    "AUTH_TOKEN",
    "ACCESS_TOKEN",
    "SECRET_KEY",
    "PASSWORD",
    "CREDENTIAL",
)
_SENSITIVE_ENV_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "SSH_AUTH_SOCK",
    }
)

_JNP_ATTRIBUTES = frozenset(
    {
        "abs",
        "array",
        "asarray",
        "clip",
        "exp",
        "float32",
        "log",
        "maximum",
        "max",
        "mean",
        "minimum",
        "min",
        "ones_like",
        "sqrt",
        "stack",
        "sum",
        "tanh",
        "where",
        "zeros_like",
    }
)
_JNP_CALLS = _JNP_ATTRIBUTES - {"float32"}
_ARRAY_ATTRIBUTES = frozenset(
    {"add", "at", "astype", "max", "min", "multiply", "set"}
)
_ARRAY_METHODS = _ARRAY_ATTRIBUTES - {"at"}
_ALLOWED_NODES = (
    ast.Module,
    ast.Import,
    ast.alias,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Assign,
    ast.Return,
    ast.Expr,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Subscript,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Call,
    ast.keyword,
    ast.Attribute,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.MatMult,
    ast.Div,
    ast.Mod,
    ast.BitAnd,
    ast.BitOr,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


class CandidateValidationError(ValueError):
    """Candidate source or output violates the bounded policy contract."""


def _initial_source_path(
    contract_version: str,
    initial_source_path: str | Path | None = None,
) -> Path:
    if initial_source_path is not None:
        return Path(initial_source_path)
    if contract_version == run_spec.R4_CONTRACT:
        return TASK_DIR / "initial_r4.py"
    if contract_version == run_spec.LEGACY_CONTRACT:
        return TASK_DIR / "initial.py"
    raise CandidateValidationError("unknown candidate contract")


def _candidate_source(
    program_path: str | Path,
    contract_version: str = run_spec.LEGACY_CONTRACT,
    initial_source_path: str | Path | None = None,
) -> str:
    path = Path(program_path)
    source_bytes = path.read_bytes()
    if len(source_bytes) > MAX_SOURCE_BYTES:
        raise CandidateValidationError("source exceeds the size limit")
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CandidateValidationError("source must be UTF-8") from error
    if sum(bool(line.strip()) for line in source.splitlines()) > MAX_NONBLANK_LINES:
        raise CandidateValidationError("source exceeds the line limit")
    _validate_immutable_regions(
        source,
        _initial_source_path(contract_version, initial_source_path),
    )
    return source


def _split_evolve_block(source: str) -> tuple[str, str, str]:
    if source.count(START_MARKER) != 1 or source.count(END_MARKER) != 1:
        raise CandidateValidationError("source must contain exactly one marker pair")
    start = source.index(START_MARKER) + len(START_MARKER)
    end = source.index(END_MARKER)
    if start >= end:
        raise CandidateValidationError("evolve markers are out of order")
    return source[:start], source[start:end], source[end:]


def _validate_immutable_regions(source: str, initial_source_path: Path) -> None:
    expected = initial_source_path.read_text(encoding="utf-8")
    expected_prefix, _, expected_suffix = _split_evolve_block(expected)
    prefix, _, suffix = _split_evolve_block(source)
    normalized_suffix = suffix.rstrip("\r\n")
    normalized_expected_suffix = expected_suffix.rstrip("\r\n")
    if not hmac.compare_digest(
        hashlib.sha256(prefix.encode()).digest(),
        hashlib.sha256(expected_prefix.encode()).digest(),
    ) or not hmac.compare_digest(
        hashlib.sha256(normalized_suffix.encode()).digest(),
        hashlib.sha256(normalized_expected_suffix.encode()).digest(),
    ):
        raise CandidateValidationError("immutable task code was modified")


def _validate_candidate_source(
    source: str,
    contract_version: str = run_spec.LEGACY_CONTRACT,
) -> ast.Module:
    """Accept one pure array expression function and no ambient capabilities."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise CandidateValidationError("source is not valid Python") from error

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise CandidateValidationError("source exceeds the syntax-tree limit")
    unsupported = next(
        (node for node in nodes if not isinstance(node, _ALLOWED_NODES)),
        None,
    )
    if unsupported is not None:
        raise CandidateValidationError(
            f"unsupported syntax: {type(unsupported).__name__}"
        )

    imports = [node for node in nodes if isinstance(node, ast.Import)]
    functions = [node for node in nodes if isinstance(node, ast.FunctionDef)]
    other = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.FunctionDef))
    ]
    if other or len(imports) != 1 or len(functions) != 1:
        raise CandidateValidationError(
            "module must contain only the approved import and make_offspring"
        )
    imported = imports[0].names
    if (
        len(imported) != 1
        or imported[0].name != "jax.numpy"
        or imported[0].asname != "jnp"
    ):
        raise CandidateValidationError("only 'import jax.numpy as jnp' is allowed")

    function = functions[0]
    arguments = function.args
    argument_names = [argument.arg for argument in arguments.args]
    expected_arguments = {
        run_spec.LEGACY_CONTRACT: [
            "parent_genome",
            "parent_stats",
            "population_stats",
            "rng",
        ],
        run_spec.R4_CONTRACT: [
            "parent_genome_summary",
            "parent_stats",
            "population_stats",
            "operator_stats",
            "rng",
        ],
    }.get(contract_version)
    if expected_arguments is None:
        raise CandidateValidationError("unknown candidate contract")
    if (
        function.name != "make_offspring"
        or argument_names != expected_arguments
        or arguments.posonlyargs
        or arguments.kwonlyargs
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or arguments.defaults
        or arguments.kw_defaults
        or function.decorator_list
        or function.returns is not None
    ):
        raise CandidateValidationError("make_offspring has the wrong signature")
    if any(argument.annotation is not None for argument in arguments.args):
        raise CandidateValidationError("function annotations are not allowed")

    local_names = set(argument_names)
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise CandidateValidationError("assignments must target one local name")
            if node.targets[0].id in set(argument_names) | {"jnp"}:
                raise CandidateValidationError("task inputs cannot be reassigned")
            local_names.add(node.targets[0].id)
        if isinstance(node, ast.Attribute):
            jnp_attribute = (
                isinstance(node.value, ast.Name)
                and node.value.id == "jnp"
                and node.attr in _JNP_ATTRIBUTES
            )
            array_method = node.attr in _ARRAY_ATTRIBUTES
            if not (jnp_attribute or array_method):
                raise CandidateValidationError(
                    "attribute access is restricted to jnp and approved array methods"
                )
        if isinstance(node, ast.Call):
            jnp_call = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "jnp"
                and node.func.attr in _JNP_CALLS
            )
            array_method_call = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _ARRAY_METHODS
            )
            if not (jnp_call or array_method_call):
                raise CandidateValidationError(
                    "only approved jnp and array-method calls are allowed"
                )
            if any(keyword.arg is None for keyword in node.keywords):
                raise CandidateValidationError(
                    "expanded keyword arguments are not allowed"
                )
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id == "rng":
                raise CandidateValidationError("rng is opaque and cannot be read")
            if node.id not in local_names | {"jnp"}:
                raise CandidateValidationError(f"unknown name: {node.id}")
        if isinstance(node, ast.Constant):
            value = node.value
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise CandidateValidationError("unsupported literal")
            if isinstance(value, float) and not math.isfinite(value):
                raise CandidateValidationError("non-finite literals are not allowed")
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and abs(value) > 1_000_000
            ):
                raise CandidateValidationError("numeric literal exceeds the limit")

    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if not returns or any(node.value is None for node in returns):
        raise CandidateValidationError("make_offspring must return operator scores")
    return tree


def _load_candidate(
    program_path: str | Path,
    contract_version: str = run_spec.LEGACY_CONTRACT,
    initial_source_path: str | Path | None = None,
) -> ModuleType:
    source = _candidate_source(program_path, contract_version, initial_source_path)
    tree = _validate_candidate_source(source, contract_version)
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    executable = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    code = compile(
        executable,
        filename=f"<evo2-candidate-{source_digest[:16]}>",
        mode="exec",
        dont_inherit=True,
        optimize=2,
    )
    module = ModuleType(f"evo2_candidate_{source_digest[:16]}")
    module.__dict__.update({"__builtins__": {}, "jnp": jnp})
    exec(code, module.__dict__, module.__dict__)  # noqa: S102 - validated AST only
    function = getattr(module, "make_offspring", None)
    if not callable(function):
        raise CandidateValidationError("make_offspring is not callable")
    module.__dict__["_evo2_source_sha256"] = source_digest
    module.__dict__["_evo2_contract_version"] = contract_version
    return module


def _scrub_sensitive_environment() -> None:
    """Remove credentials before any candidate source is compiled or executed."""
    for name in tuple(os.environ):
        upper = name.upper()
        if upper in _SENSITIVE_ENV_NAMES or any(
            fragment in upper for fragment in _SENSITIVE_ENV_FRAGMENTS
        ):
            os.environ.pop(name, None)


def _smoke_validate_candidate(
    module: ModuleType,
    output_width: int = 4,
    *,
    contract_version: str | None = None,
    runtime_budget_ms: float | None = None,
) -> None:
    function = module.make_offspring
    contract_version = contract_version or getattr(
        module, "_evo2_contract_version", run_spec.LEGACY_CONTRACT
    )
    if contract_version == run_spec.R4_CONTRACT:
        inputs = (
            jnp.array([[0.0, 0.0], [1.0, 1.0], [0.4, 0.3]], dtype=jnp.float32),
            jnp.array(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.7, 0.6, 0.5]],
                dtype=jnp.float32,
            ),
            jnp.array(
                [
                    [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.5, 0.4, -0.1, 0.2, 0.3, 0.8],
                ],
                dtype=jnp.float32,
            ),
            jnp.array(
                [
                    jnp.zeros((3, 6), dtype=jnp.float32),
                    jnp.ones((3, 6), dtype=jnp.float32),
                    jnp.full((3, 6), 0.5, dtype=jnp.float32),
                ]
            ),
            jnp.zeros((3,), dtype=jnp.float32),
        )
    elif contract_version == run_spec.LEGACY_CONTRACT:
        inputs = (
            jnp.array([[0.0, 0.0], [1.0, 1.0], [0.4, 0.3]], dtype=jnp.float32),
            jnp.array([[0.0, 0.0], [1.0, 1.0], [0.7, 0.6]], dtype=jnp.float32),
            jnp.array(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 1.0],
                    [0.5, -0.1, 0.2, 0.8],
                ],
                dtype=jnp.float32,
            ),
            jnp.zeros((3,), dtype=jnp.float32),
        )
    else:
        raise CandidateValidationError("unknown candidate contract")
    batched = jax.vmap(function)
    try:
        eager_scores = np.asarray(batched(*inputs))
        abstract = jax.eval_shape(batched, *inputs)
        compiled = jax.jit(batched)
        scores = np.asarray(compiled(*inputs))
        replay = np.asarray(compiled(*inputs))
        if runtime_budget_ms is not None:
            start = time.perf_counter()
            for _ in range(16):
                timed = compiled(*inputs)
            jax.block_until_ready(timed)
            elapsed_ms = (time.perf_counter() - start) * 1000.0 / 16.0
            if elapsed_ms > runtime_budget_ms:
                raise CandidateValidationError("candidate exceeds runtime budget")
    except Exception as error:
        raise CandidateValidationError("candidate is not JAX-compatible") from error
    if (
        abstract.shape != (3, output_width)
        or eager_scores.shape != (3, output_width)
        or scores.shape != (3, output_width)
    ):
        raise CandidateValidationError(
            f"candidate must return exactly {output_width} scores"
        )
    if not np.issubdtype(scores.dtype, np.number) or not np.all(np.isfinite(scores)):
        raise CandidateValidationError("candidate scores must be finite numbers")
    if not np.array_equal(scores, replay):
        raise CandidateValidationError("candidate must replay deterministically")


def _imports() -> dict[str, Any]:
    """Import Microcosmos only after the trusted repository boundary is checked."""
    if not (MICROCOSMOS_ROOT / "experiments" / "evo2_ecosystem").is_dir():
        raise RuntimeError("Microcosmos Evo² evaluator is unavailable")
    import sys

    root = str(MICROCOSMOS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from experiments.evo2_ecosystem.episode import (  # noqa: PLC0415
        SimulatorConfig,
        evaluate_manifest,
        simulator_config_sha256,
        simulator_source_sha256,
    )
    from experiments.evo2_ecosystem.manifests import (  # noqa: PLC0415
        load_bound_manifest,
        load_training_manifest,
    )
    from experiments.evo2_ecosystem.protocol import manifest_sha256  # noqa: PLC0415
    from experiments.evo2_ecosystem.founder_artifacts import (  # noqa: PLC0415
        founder_index_sha256,
        load_founder_index,
    )
    from microcosmos.heredity import mutate_cppn  # noqa: PLC0415

    try:
        from microcosmos.heredity import make_r4_offspring_policy  # noqa: PLC0415
    except ImportError:
        make_r4_offspring_policy = None
    try:
        from experiments.evo2_ecosystem.episode import (  # noqa: PLC0415
            evaluate_manifest_paired_delta,
        )
    except ImportError:
        evaluate_manifest_paired_delta = None

    return {
        "SimulatorConfig": SimulatorConfig,
        "evaluate_manifest": evaluate_manifest,
        "load_bound_manifest": load_bound_manifest,
        "load_training_manifest": load_training_manifest,
        "manifest_sha256": manifest_sha256,
        "founder_index_sha256": founder_index_sha256,
        "load_founder_index": load_founder_index,
        "mutate_cppn": mutate_cppn,
        "make_r4_offspring_policy": make_r4_offspring_policy,
        "evaluate_manifest_paired_delta": evaluate_manifest_paired_delta,
        "simulator_config_sha256": simulator_config_sha256,
        "simulator_source_sha256": simulator_source_sha256,
    }


def _load_bound_manifest(
    spec: dict[str, Any],
    role: str,
    dependencies: dict[str, Any],
):
    """Load exactly the manifest named by the selected immutable protocol."""
    if run_spec.schema_version(spec) == 1:
        if not role.startswith("training_"):
            raise ValueError("the pilot evaluator may load only training manifests")
        return dependencies["load_training_manifest"](role.removeprefix("training_"))
    return dependencies["load_bound_manifest"](
        run_spec.manifest_path(spec, role),
        expected_sha256=run_spec.manifest_hash(spec, role),
        expected_partition="training",
    )


def _verified_founder_index_path(
    spec: dict[str, Any], dependencies: dict[str, Any]
) -> Path | None:
    path = run_spec.founder_index_path(spec)
    if path is None:
        return None
    # Training must not open development or sealed founder bytes.  The manifest
    # evaluator resolves and verifies only the training founders it actually uses.
    index = dependencies["load_founder_index"](path, verify_artifacts=False)
    if dependencies["founder_index_sha256"](index) != spec["founder_index"]["sha256"]:
        raise RuntimeError("founder index does not match the selected run spec")
    return path


def _build_policy(
    candidate: ModuleType,
    mutate_cppn,
    *,
    contract_version: str = run_spec.LEGACY_CONTRACT,
):
    """Expose summaries to the candidate and retain mutations in trusted code."""
    function = candidate.make_offspring

    if contract_version == run_spec.LEGACY_CONTRACT:

        def policy(parent_genome, parent_stats, population_stats, mutation_context):
            safe_genome = jnp.stack(
                [parent_stats.node_fraction, parent_stats.connection_fraction]
            )
            safe_parent = jnp.stack(
                [parent_stats.energy_fraction, parent_stats.intake_ema]
            )
            safe_population = jnp.stack(
                [
                    population_stats.alive_fraction,
                    population_stats.population_change_ema,
                    population_stats.action_diversity,
                    population_stats.lineage_entropy,
                ]
            )
            scores = jnp.asarray(
                function(
                    safe_genome,
                    safe_parent,
                    safe_population,
                    jnp.array(0.0, dtype=jnp.float32),
                ),
                dtype=jnp.float32,
            )
            return mutate_cppn(parent_genome, scores, mutation_context)

        return policy

    if contract_version != run_spec.R4_CONTRACT:
        raise CandidateValidationError("unknown candidate contract")

    if mutate_cppn is None:
        raise RuntimeError("Microcosmos does not provide the trusted r4 adapter")
    return mutate_cppn(function)


def _episode_metrics_legacy(evaluation) -> dict[str, Any]:
    episodes = evaluation.episodes
    integrity = bool(np.asarray(evaluation.integrity_valid))
    raw_score = float(np.asarray(evaluation.candidate_score))
    score = raw_score if integrity and math.isfinite(raw_score) else 0.0
    births = np.asarray([episode.birth_count for episode in episodes], dtype=float)
    final_alive = np.asarray([episode.final_alive for episode in episodes], dtype=float)
    survived = np.asarray([episode.survived for episode in episodes], dtype=float)
    violations = int(
        np.sum([int(np.asarray(ep.policy_violation_count)) for ep in episodes])
    )
    operator_counts = np.sum(
        np.stack([np.asarray(ep.operator_counts) for ep in episodes]), axis=0
    )
    operator_total = int(np.sum(operator_counts))
    fractions = (
        operator_counts / operator_total if operator_total else np.zeros(4, dtype=float)
    )
    repeat_scores = np.asarray(
        getattr(evaluation, "repeat_scores", np.asarray([raw_score])),
        dtype=float,
    )
    selected_repeat_index = int(
        np.asarray(getattr(evaluation, "selected_repeat_index", 0))
    )
    feedback = (
        f"training score={score:.4f}, "
        f"survival={float(np.mean(survived)):.2f}, "
        f"mean births={float(np.mean(births)):.1f}; "
        "operator fractions "
        f"clone={fractions[0]:.2f}, parametric={fractions[1]:.2f}, "
        f"structural={fractions[2]:.2f}, mixed={fractions[3]:.2f}."
        f" numerical repeats={repeat_scores.size}, "
        f"range={float(np.ptp(repeat_scores)):.4f}."
    )
    return {
        "combined_score": score,
        "public": {
            "score": score,
            "survival_rate": float(np.mean(survived)),
            "mean_final_alive": float(np.mean(final_alive)),
            "mean_births": float(np.mean(births)),
            "clone_fraction": float(fractions[0]),
            "parametric_fraction": float(fractions[1]),
            "structural_fraction": float(fractions[2]),
            "mixed_fraction": float(fractions[3]),
        },
        "private": {
            "full_evaluation_performed": True,
            "integrity_valid": integrity,
            "policy_violation_count": violations,
            "numerical_repeats": int(repeat_scores.size),
            "repeat_scores": [float(value) for value in repeat_scores],
            "selected_repeat_index": selected_repeat_index,
        },
        "text_feedback": feedback,
    }


_R4_OPERATOR_NAMES = (
    "clone",
    "parametric_conservative",
    "parametric_standard",
    "parametric_exploratory",
    "structural",
    "mixed",
)


def _mean_optional(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    array = np.asarray(value, dtype=float)
    return float(np.mean(array)) if array.size else default


def _r4_episode_vector(episodes: Any, name: str) -> np.ndarray:
    values = [np.asarray(getattr(episode, name), dtype=float) for episode in episodes]
    if not values:
        return np.zeros(6, dtype=float)
    result = np.sum(np.stack(values), axis=0)
    if result.shape != (6,):
        raise ValueError(f"r4 telemetry {name} must have shape (6,)")
    return result


def _episode_metrics_r4(evaluation: Any, regime: str) -> dict[str, Any]:
    """Return only training-visible r4 diagnostics for one outer arm."""
    if regime == "stable":
        episodes = getattr(evaluation, "sham_episodes", None)
        if episodes is None:
            raise ValueError("stable r4 feedback requires explicit sham-only episodes")
        all_episodes = tuple(evaluation.episodes)
        ancestor_episodes = tuple(getattr(evaluation, "ancestor_episodes", ()))
        sham_ids = {id(episode) for episode in episodes}
        sham_indices = [
            index for index, episode in enumerate(all_episodes) if id(episode) in sham_ids
        ]
        if len(sham_indices) != len(episodes) or len(ancestor_episodes) != len(all_episodes):
            raise ValueError("stable r4 feedback requires paired ancestor sham episodes")
        integrity = all(
            bool(np.asarray(getattr(all_episodes[index], "integrity_valid", True)))
            and bool(np.asarray(all_episodes[index].survived))
            and bool(
                np.asarray(getattr(ancestor_episodes[index], "integrity_valid", True))
            )
            and bool(np.asarray(ancestor_episodes[index].survived))
            for index in sham_indices
        )
    else:
        episodes = evaluation.episodes
        integrity = bool(np.asarray(evaluation.integrity_valid))
    raw_score = float(np.asarray(evaluation.candidate_score))
    score = raw_score if integrity and math.isfinite(raw_score) else -2.0
    births = np.asarray([episode.birth_count for episode in episodes], dtype=float)
    deaths = np.asarray(
        [getattr(episode, "natural_death_count", 0) for episode in episodes],
        dtype=float,
    )
    survived = np.asarray([episode.survived for episode in episodes], dtype=float)
    generation_gain = np.asarray(
        [
            getattr(
                episode,
                "generation_gain",
                getattr(episode, "maximum_generation", 0),
            )
            for episode in episodes
        ],
        dtype=float,
    )
    violations = int(
        np.sum([int(np.asarray(ep.policy_violation_count)) for ep in episodes])
    )
    counts = _r4_episode_vector(episodes, "operator_counts")
    total = float(np.sum(counts))
    fractions = counts / total if total else np.zeros(6, dtype=float)

    pre_sum = _r4_episode_vector(episodes, "pre_selection_probability_sum")
    post_sum = _r4_episode_vector(episodes, "post_selection_probability_sum")
    pre_count = float(
        np.sum([getattr(ep, "pre_selection_probability_count", 0) for ep in episodes])
    )
    post_count = float(
        np.sum([getattr(ep, "post_selection_probability_count", 0) for ep in episodes])
    )
    pre_prob = pre_sum / pre_count if pre_count else np.zeros(6, dtype=float)
    post_prob = post_sum / post_count if post_count else np.zeros(6, dtype=float)

    success = np.mean(
        np.stack([np.asarray(ep.operator_success_ema, dtype=float) for ep in episodes]),
        axis=0,
    )
    usage = np.mean(
        np.stack([np.asarray(ep.operator_usage_ema, dtype=float) for ep in episodes]),
        axis=0,
    )
    evidence = np.mean(
        np.stack(
            [np.asarray(ep.operator_evidence_ema, dtype=float) for ep in episodes]
        ),
        axis=0,
    )
    for name, value in (
        ("success", success),
        ("usage", usage),
        ("evidence", evidence),
    ):
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError(f"r4 operator {name} telemetry is invalid")

    repeat_scores = np.asarray(evaluation.repeat_scores, dtype=float)
    selected_repeat_index = int(np.asarray(evaluation.selected_repeat_index))
    if not integrity:
        repeat_scores = np.full_like(repeat_scores, -2.0, dtype=float)
    if (
        repeat_scores.ndim != 1
        or repeat_scores.size == 0
        or not 0 <= selected_repeat_index < repeat_scores.size
        or float(repeat_scores[selected_repeat_index]) != score
    ):
        raise ValueError("r4 coherent numerical-repeat selection is invalid")

    public: dict[str, Any] = {
        "score": score,
        "sham_auc_delta": _mean_optional(getattr(evaluation, "sham_auc_delta", None)),
        "survival_rate": float(np.mean(survived)),
        "mean_births": float(np.mean(births)),
        "mean_deaths": float(np.mean(deaths)),
        "mean_generation_gain": float(np.mean(generation_gain)),
        "operator_fraction": {
            name: float(fractions[index])
            for index, name in enumerate(_R4_OPERATOR_NAMES)
        },
        "pre_selection_probability": [float(value) for value in pre_prob],
        "post_selection_probability": [float(value) for value in post_prob],
        "operator_success_ema": [float(value) for value in success],
        "operator_usage_ema": [float(value) for value in usage],
        "operator_evidence_ema": [float(value) for value in evidence],
    }
    if regime == "punctuated":
        public["shock_auc_delta"] = _mean_optional(
            getattr(evaluation, "shock_auc_delta", None)
        )

    dominant = _R4_OPERATOR_NAMES[int(np.argmax(fractions))]
    weakness = (
        "no births resolved heredity choices"
        if total == 0
        else f"dominant action was {dominant} ({float(np.max(fractions)):.2f})"
    )
    feedback_parts = [
        f"paired ancestor delta={score:.4f}",
        f"sham delta={public['sham_auc_delta']:.4f}",
    ]
    if regime == "punctuated":
        feedback_parts.append(f"shock delta={public['shock_auc_delta']:.4f}")
    feedback_parts.extend(
        [
            f"survival={public['survival_rate']:.2f}",
            f"mean births={public['mean_births']:.1f}",
            weakness,
        ]
    )
    return {
        "combined_score": score,
        "public": public,
        "private": {
            "full_evaluation_performed": True,
            "integrity_valid": integrity,
            "policy_violation_count": violations,
            "numerical_repeats": int(repeat_scores.size),
            "repeat_scores": [float(value) for value in repeat_scores],
            "selected_repeat_index": selected_repeat_index,
        },
        "text_feedback": "; ".join(feedback_parts) + ".",
    }


def _episode_metrics(
    evaluation: Any,
    regime: str = "punctuated",
    contract_version: str = run_spec.LEGACY_CONTRACT,
) -> dict[str, Any]:
    if contract_version == run_spec.R4_CONTRACT:
        return _episode_metrics_r4(evaluation, regime)
    return _episode_metrics_legacy(evaluation)


def _evaluate_candidate(
    program_path: str | Path,
    regime: str,
    *,
    run_spec_path: str | Path | None = None,
    expected_run_spec_sha256: str | None = None,
) -> dict[str, Any]:
    if regime not in REGIMES:
        raise ValueError("unknown training regime")
    spec, spec_hash, _ = run_spec.load_run_spec(run_spec_path)
    if expected_run_spec_sha256 is not None and not hmac.compare_digest(
        spec_hash, expected_run_spec_sha256
    ):
        raise RuntimeError("selected run specification hash does not match launcher")
    contract_version = run_spec.candidate_contract_version(spec)
    initial_path = run_spec.initial_program_path(spec)
    if run_spec.sha256_file(initial_path) != spec["source_sha256"]["initial"]:
        raise RuntimeError("initial program does not match the selected run spec")
    candidate = _load_candidate(program_path, contract_version, initial_path)
    candidate_source_hash = candidate._evo2_source_sha256
    evaluator_source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output_width = run_spec.candidate_output_width(spec)
    runtime_budget_ms = (
        float(spec["candidate_contract"]["runtime_budget_ms"])
        if contract_version == run_spec.R4_CONTRACT
        else None
    )
    _smoke_validate_candidate(
        candidate,
        output_width,
        contract_version=contract_version,
        runtime_budget_ms=runtime_budget_ms,
    )
    dependencies = _imports()
    simulator_source_hash = dependencies["simulator_source_sha256"]()
    manifest = _load_bound_manifest(spec, f"training_{regime}", dependencies)
    if manifest.partition != "training":
        raise RuntimeError("training loader returned a non-training manifest")
    config = dependencies["SimulatorConfig"]()
    founder_index_path = _verified_founder_index_path(spec, dependencies)
    if contract_version == run_spec.R4_CONTRACT:
        mutate = dependencies["make_r4_offspring_policy"]
        paired_evaluator = dependencies["evaluate_manifest_paired_delta"]
        if mutate is None or paired_evaluator is None:
            raise RuntimeError("Microcosmos does not provide the trusted r4 contract")
    else:
        mutate = dependencies["mutate_cppn"]
        paired_evaluator = None
    policy = _build_policy(candidate, mutate, contract_version=contract_version)
    evaluation_kwargs = {}
    if founder_index_path is not None:
        evaluation_kwargs["founder_index_path"] = founder_index_path
    if contract_version == run_spec.R4_CONTRACT:
        ancestor = _load_candidate(initial_path, contract_version, initial_path)
        _smoke_validate_candidate(
            ancestor,
            output_width,
            contract_version=contract_version,
            runtime_budget_ms=runtime_budget_ms,
        )
        ancestor_policy = _build_policy(
            ancestor,
            mutate,
            contract_version=contract_version,
        )
        evaluation = paired_evaluator(
            manifest,
            config,
            policy,
            ancestor_policy,
            regime=regime,
            **evaluation_kwargs,
        )
    else:
        evaluation = dependencies["evaluate_manifest"](
            manifest,
            config,
            policy,
            **evaluation_kwargs,
        )
    metrics = _episode_metrics(evaluation, regime, contract_version)
    metrics["private"]["candidate_sha256"] = candidate_source_hash
    metrics["private"]["manifest_sha256"] = dependencies["manifest_sha256"](manifest)
    metrics["private"]["simulator_config_sha256"] = dependencies[
        "simulator_config_sha256"
    ](config)
    metrics["private"]["evaluator_source_sha256"] = evaluator_source_hash
    metrics["private"]["simulator_source_sha256"] = simulator_source_hash
    metrics["private"]["run_spec_sha256"] = spec_hash
    metrics["private"]["run_spec_schema_version"] = run_spec.schema_version(spec)
    metrics["private"]["candidate_output_width"] = output_width
    metrics["private"]["candidate_contract_version"] = contract_version
    metrics["private"]["initial_program_sha256"] = spec["source_sha256"]["initial"]
    if founder_index_path is not None:
        metrics["private"]["founder_index_sha256"] = spec["founder_index"]["sha256"]
    metrics["private"]["backend"] = jax.default_backend()
    metrics["private"]["jax_version"] = jax.__version__
    metrics["private"]["jaxlib_version"] = metadata.version("jaxlib")
    metrics["private"]["python_version"] = platform.python_version()
    metrics["private"]["xla_flags"] = os.environ.get("XLA_FLAGS", "")
    final_candidate_hash = hashlib.sha256(
        _candidate_source(program_path, contract_version, initial_path).encode("utf-8")
    ).hexdigest()
    final_evaluator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    final_simulator_hash = dependencies["simulator_source_sha256"]()
    _, final_spec_hash, _ = run_spec.load_run_spec(run_spec_path)
    if not (
        hmac.compare_digest(candidate_source_hash, final_candidate_hash)
        and hmac.compare_digest(evaluator_source_hash, final_evaluator_hash)
        and hmac.compare_digest(simulator_source_hash, final_simulator_hash)
        and hmac.compare_digest(spec_hash, final_spec_hash)
    ):
        raise RuntimeError("trusted source changed during candidate evaluation")
    return metrics


def _sanitize_error(error: BaseException) -> tuple[str, str]:
    if isinstance(error, CandidateValidationError):
        return "candidate_rejected", f"candidate rejected: {error}"
    if isinstance(error, TimeoutError):
        return "timeout", "candidate evaluation timed out"
    return "evaluation_failed", f"candidate evaluation failed ({type(error).__name__})"


def _failure_metrics(
    error_code: str,
    program_path: str | Path,
    regime: str,
    *,
    run_spec_path: str | Path | None = None,
    expected_run_spec_sha256: str | None = None,
) -> dict[str, Any]:
    """Return an auditable invalid proposal without pretending it was evaluated."""
    try:
        candidate_hash = hashlib.sha256(Path(program_path).read_bytes()).hexdigest()
    except OSError:
        candidate_hash = ""
    evaluator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_hash = ""
    simulator_config_hash = ""
    simulator_source_hash = ""
    spec_hash = ""
    spec_schema_version = None
    output_width = 4
    contract_version = run_spec.LEGACY_CONTRACT
    initial_program_hash = ""
    founder_hash = ""
    try:
        spec, spec_hash, _ = run_spec.load_run_spec(run_spec_path)
        if expected_run_spec_sha256 is not None and not hmac.compare_digest(
            spec_hash, expected_run_spec_sha256
        ):
            raise RuntimeError(
                "selected run specification hash does not match launcher"
            )
        spec_schema_version = run_spec.schema_version(spec)
        output_width = run_spec.candidate_output_width(spec)
        contract_version = run_spec.candidate_contract_version(spec)
        initial_program_hash = spec["source_sha256"]["initial"]
        dependencies = _imports()
        manifest = _load_bound_manifest(spec, f"training_{regime}", dependencies)
        config = dependencies["SimulatorConfig"]()
        manifest_hash = dependencies["manifest_sha256"](manifest)
        simulator_config_hash = dependencies["simulator_config_sha256"](config)
        simulator_source_hash = dependencies["simulator_source_sha256"]()
        if spec_schema_version >= 2:
            founder_hash = spec["founder_index"]["sha256"]
    except Exception:
        # The record remains invalid and the freezer will fail closed if trusted
        # infrastructure provenance could not be established.
        pass
    invalid_score = -2.0 if contract_version == run_spec.R4_CONTRACT else 0.0
    return {
        "combined_score": invalid_score,
        "public": {"score": invalid_score},
        "private": {
            "error_code": error_code,
            "full_evaluation_performed": False,
            "integrity_valid": False,
            "candidate_sha256": candidate_hash,
            "manifest_sha256": manifest_hash,
            "simulator_config_sha256": simulator_config_hash,
            "evaluator_source_sha256": evaluator_hash,
            "simulator_source_sha256": simulator_source_hash,
            "run_spec_sha256": spec_hash,
            "run_spec_schema_version": spec_schema_version,
            "candidate_output_width": output_width,
            "candidate_contract_version": contract_version,
            "initial_program_sha256": initial_program_hash,
            "founder_index_sha256": founder_hash,
            "numerical_repeats": 0,
            "repeat_scores": [],
            "selected_repeat_index": None,
        },
        "text_feedback": "Candidate was invalid; inspect the bounded policy contract.",
    }


def main(
    program_path: str,
    results_dir: str,
    training_regime: str,
    run_spec_path: str | Path | None = None,
    run_spec_sha256: str | None = None,
) -> None:
    _scrub_sensitive_environment()
    try:
        if run_spec_path is None and run_spec_sha256 is None:
            metrics = _evaluate_candidate(program_path, training_regime)
        else:
            metrics = _evaluate_candidate(
                program_path,
                training_regime,
                run_spec_path=run_spec_path,
                expected_run_spec_sha256=run_spec_sha256,
            )
        correct = bool(metrics["private"]["integrity_valid"])
        error = None if correct else "ecosystem integrity check failed"
    except Exception as caught:  # evaluator must always emit Shinka artifacts
        error_code, error = _sanitize_error(caught)
        metrics = _failure_metrics(
            error_code,
            program_path,
            training_regime,
            run_spec_path=run_spec_path,
            expected_run_spec_sha256=run_spec_sha256,
        )
        correct = False
    save_json_results(
        results_dir,
        metrics,
        correct,
        error,
        verbose=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--training_regime", choices=REGIMES, required=True)
    parser.add_argument("--run_spec_path")
    parser.add_argument("--run_spec_sha256")
    arguments = parser.parse_args()
    main(
        arguments.program_path,
        arguments.results_dir,
        arguments.training_regime,
        arguments.run_spec_path,
        arguments.run_spec_sha256,
    )
