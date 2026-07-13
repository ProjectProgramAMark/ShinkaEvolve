"""Canonical, shared specification for the matched Evo² outer searches."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[2]
PREREGISTRATION_PATH = TASK_DIR.parents[2] / "plans" / "experiment-analysis-plan.md"
DEPENDENCY_LOCK_PATH = TASK_DIR.parents[2] / "microcosmos" / "uv.lock"
SPEC_PATH = TASK_DIR / "run_spec.json"
SPEC_HASH_PATH = TASK_DIR / "run_spec.sha256"
PROFILE_DIR = TASK_DIR / "run_specs"
REGIMES = ("stable", "punctuated")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROFILE = re.compile(r"[a-z0-9][a-z0-9_-]{1,62}")

_COMMON_KEYS = {
    "schema_version",
    "run_id",
    "generations",
    "outer_seed",
    "model",
    "headless_command",
    "top_k",
    "numerical_repeats",
    "simulator_config_sha256",
    "source_sha256",
    "archive",
    "bootstrap",
    "baselines",
    "search",
}
_MANIFEST_ROLES = {
    "training_stable",
    "training_punctuated",
    "development",
    "sealed",
}
_V1_SOURCE_KEYS = {
    "initial",
    "evaluator",
    "simulator",
    "analysis",
    "baseline",
    "dependency_lock",
    "preregistration",
}
_V2_SOURCE_KEYS = {
    "initial",
    "evaluator",
    "simulator",
    "analysis",
    "baseline",
    "dependency_lock",
    "launcher",
    "finalist_selector",
    "lineage_selector",
    "run_spec_module",
}
_V3_SOURCE_KEYS = _V2_SOURCE_KEYS | {
    "adaptive_selector",
    "r4_final_analysis",
    "r4_manifest_generator",
    "r4_qualification",
    "r4_opportunity",
}
_V4_SOURCE_KEYS = _V2_SOURCE_KEYS | {"adaptive_selector"}

R5_PROTOCOL_REVISION = "evo2-r5-world-feasibility-v1"
R5_RUN_ID = "evo2-r5-world-feasibility-20260713"
R5_PROTOCOL_TOOL_CALLABLES = {
    "world_qualification": "validate_world_qualification",
    "manifest_generation": "build_and_publish_manifests",
    "disturbance_qualification": "run_disturbance_qualification",
    "opportunity_qualification": "run_opportunity_qualification",
    "final_analysis": "run_final_analysis",
}
R5_PREREQUISITE_ROLES = {
    "world_qualification",
    "disturbance_qualification",
    "operator_opportunity",
}
R5_PROTOCOL_ROOT = "microcosmos/experiments/evo2_ecosystem/r5"
R5_PROTOCOL_TOOL_PATHS = {
    "world_qualification": f"{R5_PROTOCOL_ROOT}/qualify_worlds.py",
    "manifest_generation": f"{R5_PROTOCOL_ROOT}/tools.py",
    "disturbance_qualification": f"{R5_PROTOCOL_ROOT}/tools.py",
    "opportunity_qualification": f"{R5_PROTOCOL_ROOT}/tools.py",
    "final_analysis": f"{R5_PROTOCOL_ROOT}/analysis.py",
}
R5_ARTIFACT_ROOT = f"{R5_PROTOCOL_ROOT}/artifacts"
R5_PROTOCOL_DOCUMENT_PATH = (
    "microcosmos/docs/evo2/evo2-r5-world-feasibility-plan.md"
)
R5_STOP_REPORT_PATH = (
    "microcosmos/docs/evo2/evo2-r5-world-feasibility-stop-report.md"
)
R5_PROFILE_PATH = (
    "ShinkaEvolve/examples/evo2_ecosystem/run_specs/"
    "evo2-r5-world-feasibility.json"
)
R5_PROFILE_HASH_PATH = (
    "ShinkaEvolve/examples/evo2_ecosystem/run_specs/"
    "evo2-r5-world-feasibility.sha256"
)
R5_PREREQUISITE_PATHS = {
    "world_qualification": f"{R5_ARTIFACT_ROOT}/world_qualification.json",
    "disturbance_qualification": (
        f"{R5_ARTIFACT_ROOT}/disturbance_qualification.json"
    ),
    "operator_opportunity": f"{R5_ARTIFACT_ROOT}/operator_opportunity.json",
}
R5_MANIFEST_PATHS = {
    role: f"{R5_ARTIFACT_ROOT}/manifests/{role}.json" for role in _MANIFEST_ROLES
}
R5_FOUNDER_INDEX_PATH = f"{R5_ARTIFACT_ROOT}/founders/index.json"
R5_BASELINE_SOURCE_PATH = f"{R5_PROTOCOL_ROOT}/baselines.py"
R5_ARTIFACT_ROOTS = {
    "results": "ShinkaEvolve/examples/evo2_ecosystem/results",
    "frozen": "ShinkaEvolve/examples/evo2_ecosystem/frozen",
}
R5_BASELINES = [
    "clone",
    "conservative_parametric",
    "standard_parametric",
    "exploratory_parametric",
    "structural",
    "mixed",
    "human_stress",
    "human_credit",
    "structured_random",
]
R5_OUTER_SEED = 17
R5_MODEL = "headless/codex@gpt-5.5?effort=high"
R5_HEADLESS_COMMAND = "npx -y @roberttlange/headless@0.4.0"
R5_BOOTSTRAP = {
    "confidence": 0.95,
    "replicates": 10_000,
    "seed": 20_260_712,
}
R5_SEARCH = {
    "evaluation_timeout": "00:10:00",
    "max_db_workers": 1,
    "max_evaluation_jobs": 1,
    "max_novelty_attempts": 1,
    "max_patch_attempts": 2,
    "max_patch_resamples": 2,
    "max_proposal_jobs": 1,
    "max_tokens": 8192,
    "patch_type_probs": [0.7, 0.3],
    "patch_types": ["diff", "full"],
    "reasoning_effort": "high",
    "temperature": 0.0,
}

LEGACY_CONTRACT = "evo2-heredity-r3-v1"
R4_CONTRACT = "evo2-heredity-r4-v1"


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    arm_results: Path
    selection: Path
    frozen: Path


@dataclass(frozen=True)
class StructuredRandomPaths:
    """Fixed r5 artifact locations for the matched structured-random control."""

    run_root: Path
    control_root: Path
    roster: Path
    training: Path
    selection: Path
    frozen: Path


def canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_relative_path(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"{label} path must be project-relative without '..'")


def _validate_binding(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must bind exactly one path and SHA-256")
    _validate_relative_path(value["path"], label)
    if not _is_sha256(value["sha256"]):
        raise ValueError(f"{label} contains an invalid SHA-256")


def _validate_passed_binding(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "passed"}:
        raise ValueError(f"{label} must bind one path, SHA-256, and passed flag")
    _validate_relative_path(value["path"], label)
    if not _is_sha256(value["sha256"]):
        raise ValueError(f"{label} contains an invalid SHA-256")
    if value["passed"] is not True:
        raise ValueError(f"{label} must record passed=true")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_shared(spec: dict[str, Any]) -> None:
    integer_keys = ("generations", "outer_seed", "top_k", "numerical_repeats")
    integers_valid = all(
        isinstance(spec[key], int) and not isinstance(spec[key], bool)
        for key in integer_keys
    )
    if not integers_valid or (
        not isinstance(spec["run_id"], str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", spec["run_id"])
        or spec["generations"] < 1
        or spec["outer_seed"] < 0
        or spec["top_k"] not in {3, 5}
        or spec["numerical_repeats"] != 3
        or not isinstance(spec["model"], str)
        or not spec["model"].startswith("headless/")
    ):
        raise ValueError("run specification contains an invalid fixed setting")
    if spec["headless_command"] != "npx -y @roberttlange/headless@0.4.0":
        raise ValueError("headless package must be pinned to 0.4.0")
    if not isinstance(spec["baselines"], list) or not all(
        isinstance(item, str) and item for item in spec["baselines"]
    ):
        raise ValueError("run specification has an invalid baseline set")
    if not _is_sha256(spec["simulator_config_sha256"]):
        raise ValueError("run specification contains an invalid simulator SHA-256")


def _validate_v1(spec: dict[str, Any]) -> None:
    required = _COMMON_KEYS | {"manifest_sha256"}
    if set(spec) != required:
        raise ValueError("run specification has the wrong schema")
    if spec["baselines"] != [
        "clone",
        "fixed_parametric",
        "fixed_mixed",
        "stress_responsive",
    ]:
        raise ValueError("run specification has the wrong baseline set")
    if spec["top_k"] != 3:
        raise ValueError("legacy run specification must freeze top_k=3")
    if set(spec["manifest_sha256"]) != _MANIFEST_ROLES:
        raise ValueError("run specification has incomplete manifest hashes")
    if set(spec["source_sha256"]) != _V1_SOURCE_KEYS:
        raise ValueError("run specification has incomplete source hashes")
    hashes = [
        spec["simulator_config_sha256"],
        *spec["manifest_sha256"].values(),
        *spec["source_sha256"].values(),
    ]
    if any(not _is_sha256(value) for value in hashes):
        raise ValueError("run specification contains an invalid SHA-256")


def _validate_v2(spec: dict[str, Any]) -> None:
    required = _COMMON_KEYS | {
        "manifests",
        "founder_index",
        "candidate_output_width",
        "preregistration",
        "implementation_plan",
        "artifact_roots",
    }
    if set(spec) != required:
        raise ValueError("run specification has the wrong schema")
    if spec["top_k"] != 3:
        raise ValueError("schema-v2 run specification must freeze top_k=3")
    if set(spec["manifests"]) != _MANIFEST_ROLES:
        raise ValueError("run specification has incomplete manifest bindings")
    for role, binding in spec["manifests"].items():
        _validate_binding(binding, f"{role} manifest")
    _validate_binding(spec["founder_index"], "founder index")
    _validate_binding(spec["preregistration"], "preregistration")
    _validate_binding(spec["implementation_plan"], "implementation plan")
    width = spec["candidate_output_width"]
    if not isinstance(width, int) or isinstance(width, bool) or not 4 <= width <= 16:
        raise ValueError("candidate_output_width must be an integer in [4, 16]")
    if set(spec["source_sha256"]) != _V2_SOURCE_KEYS or any(
        not _is_sha256(value) for value in spec["source_sha256"].values()
    ):
        raise ValueError("run specification has incomplete source hashes")
    roots = spec["artifact_roots"]
    if not isinstance(roots, dict) or set(roots) != {"results", "frozen"}:
        raise ValueError("artifact_roots must bind results and frozen roots")
    for name, value in roots.items():
        _validate_relative_path(value, f"{name} artifact root")


def _validate_v3(spec: dict[str, Any]) -> None:
    required = _COMMON_KEYS | {
        "manifests",
        "founder_index",
        "candidate_output_width",
        "candidate_contract",
        "initial_program",
        "proposal_budget",
        "preregistration",
        "implementation_plan",
        "artifact_roots",
    }
    if set(spec) != required:
        raise ValueError("run specification has the wrong schema")
    # Reuse the immutable artifact validation from schema 2.
    v2_view = {
        key: value
        for key, value in spec.items()
        if key not in {"candidate_contract", "initial_program", "proposal_budget"}
    }
    v2_view["schema_version"] = 2
    v2_view["top_k"] = 3
    v2_view["source_sha256"] = {
        name: spec["source_sha256"][name] for name in _V2_SOURCE_KEYS
    }
    _validate_v2(v2_view)
    if spec["candidate_output_width"] != 6:
        raise ValueError("the r4 candidate output width must be six")
    _validate_binding(spec["initial_program"], "initial program")
    contract = spec["candidate_contract"]
    expected_contract = {
        "version": R4_CONTRACT,
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
    }
    if contract != expected_contract:
        raise ValueError("run specification has the wrong r4 candidate contract")
    if (
        not isinstance(spec["proposal_budget"], int)
        or isinstance(spec["proposal_budget"], bool)
        or spec["proposal_budget"] != 50
        or spec["generations"] != spec["proposal_budget"] + 1
        or spec["top_k"] != 5
        or spec["archive"].get("num_islands") != 2
        or spec["archive"].get("archive_size") != 32
    ):
        raise ValueError("run specification has the wrong fixed r4 search budget")
    if set(spec["source_sha256"]) != _V3_SOURCE_KEYS or any(
        not _is_sha256(value) for value in spec["source_sha256"].values()
    ):
        raise ValueError("run specification has incomplete r4 source hashes")
    if spec["source_sha256"]["initial"] != spec["initial_program"]["sha256"]:
        raise ValueError("initial source bindings disagree")


def _validate_v4(spec: dict[str, Any]) -> None:
    required = _COMMON_KEYS | {
        "manifests",
        "founder_index",
        "candidate_output_width",
        "candidate_contract",
        "initial_program",
        "proposal_budget",
        "artifact_roots",
        "protocol_revision",
        "protocol_tools",
        "repository_commits",
        "prerequisite_artifacts",
        "protocol_document",
    }
    if set(spec) != required:
        raise ValueError("run specification has the wrong schema")
    if not isinstance(spec["source_sha256"], dict) or set(
        spec["source_sha256"]
    ) != _V4_SOURCE_KEYS or any(
        not _is_sha256(value) for value in spec["source_sha256"].values()
    ):
        raise ValueError("run specification has incomplete r5 source hashes")

    # Preserve schema 2's artifact and ABI validation without changing its
    # permanently frozen top-k and source-key contracts.
    v2_view = {
        key: value
        for key, value in spec.items()
        if key
        not in {
            "candidate_contract",
            "initial_program",
            "proposal_budget",
            "protocol_revision",
            "protocol_tools",
            "repository_commits",
            "prerequisite_artifacts",
            "protocol_document",
        }
    }
    v2_view.update(
        {
            "schema_version": 2,
            "top_k": 3,
            "source_sha256": {
                name: spec["source_sha256"][name] for name in _V2_SOURCE_KEYS
            },
            "preregistration": spec["protocol_document"],
            "implementation_plan": spec["protocol_document"],
        }
    )
    _validate_v2(v2_view)

    if spec["protocol_revision"] != R5_PROTOCOL_REVISION:
        raise ValueError("run specification has the wrong r5 protocol revision")
    if spec["run_id"] != R5_RUN_ID:
        raise ValueError("run specification has the wrong frozen r5 run ID")
    if spec["candidate_output_width"] != 6:
        raise ValueError("the r5 candidate output width must be six")
    _validate_binding(spec["initial_program"], "initial program")
    expected_contract = {
        "version": R4_CONTRACT,
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
    }
    if spec["candidate_contract"] != expected_contract:
        raise ValueError("run specification has the wrong r5 candidate contract")
    archive = spec["archive"]
    if (
        not isinstance(spec["proposal_budget"], int)
        or isinstance(spec["proposal_budget"], bool)
        or spec["proposal_budget"] != 50
        or spec["generations"] != spec["proposal_budget"] + 1
        or spec["top_k"] != 5
        or not isinstance(archive, dict)
        or archive.get("num_islands") != 2
        or archive.get("archive_size") != 32
    ):
        raise ValueError("run specification has the wrong fixed r5 search budget")
    if spec["baselines"] != R5_BASELINES:
        raise ValueError("run specification has the wrong r5 baseline set")
    if spec["source_sha256"]["initial"] != spec["initial_program"]["sha256"]:
        raise ValueError("initial source bindings disagree")

    tools = spec["protocol_tools"]
    if not isinstance(tools, dict) or set(tools) != set(
        R5_PROTOCOL_TOOL_CALLABLES
    ):
        raise ValueError("run specification has the wrong r5 protocol-tool roles")
    protocol_root = Path(R5_PROTOCOL_ROOT)
    for role, binding in tools.items():
        _validate_binding(binding, f"{role} protocol tool")
        if not _is_relative_to(Path(binding["path"]), protocol_root):
            raise ValueError("r5 protocol-tool path escapes the allowlisted root")
    if spec["source_sha256"]["analysis"] != tools["final_analysis"]["sha256"]:
        raise ValueError("final-analysis source bindings disagree")

    commits = spec["repository_commits"]
    if not isinstance(commits, dict) or set(commits) != {
        "microcosmos",
        "shinkaevolve",
    }:
        raise ValueError("repository_commits must bind both source repositories")
    if any(
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{40}", value) is None
        for value in commits.values()
    ):
        raise ValueError("repository_commits contains an invalid Git commit")

    prerequisites = spec["prerequisite_artifacts"]
    if not isinstance(prerequisites, dict) or set(prerequisites) != (
        R5_PREREQUISITE_ROLES
    ):
        raise ValueError("run specification has the wrong prerequisite roles")
    for role, binding in prerequisites.items():
        _validate_passed_binding(binding, f"{role} prerequisite")
        if binding["path"] != R5_PREREQUISITE_PATHS[role]:
            raise ValueError(f"{role} prerequisite has the wrong frozen path")

    _validate_binding(spec["protocol_document"], "protocol document")
    if spec["protocol_document"]["path"] != R5_PROTOCOL_DOCUMENT_PATH:
        raise ValueError("protocol document has the wrong frozen path")
    if {
        role: binding["path"] for role, binding in spec["manifests"].items()
    } != R5_MANIFEST_PATHS:
        raise ValueError("r5 manifests have the wrong frozen paths")
    if spec["founder_index"]["path"] != R5_FOUNDER_INDEX_PATH:
        raise ValueError("r5 founder index has the wrong frozen path")
    if spec["artifact_roots"] != R5_ARTIFACT_ROOTS:
        raise ValueError("r5 artifact roots have the wrong frozen paths")


def _validate(spec: dict[str, Any]) -> None:
    if not isinstance(spec.get("schema_version"), int):
        raise ValueError("run specification has the wrong schema")
    if spec["schema_version"] == 1:
        _validate_v1(spec)
    elif spec["schema_version"] == 2:
        _validate_v2(spec)
    elif spec["schema_version"] == 3:
        _validate_v3(spec)
    elif spec["schema_version"] == 4:
        _validate_v4(spec)
    else:
        raise ValueError("run specification has the wrong schema")
    _validate_shared(spec)


def _resolve_confined_file(binding: dict[str, str], root: str, label: str) -> Path:
    path = artifact_path(binding)
    allowed_root = (PROJECT_ROOT / root).resolve()
    if not _is_relative_to(path, allowed_root):
        raise ValueError(f"{label} resolves outside its allowlisted root")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return path


def protocol_tool_paths(spec: dict[str, Any]) -> dict[str, Path]:
    """Authenticate and resolve the five schema-v4 protocol tool sources."""
    if schema_version(spec) != 4:
        return {}
    paths: dict[str, Path] = {}
    for role, callable_name in R5_PROTOCOL_TOOL_CALLABLES.items():
        binding = spec["protocol_tools"][role]
        path = _resolve_confined_file(binding, R5_PROTOCOL_ROOT, f"{role} tool")
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"{role} protocol-tool SHA-256 does not match")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as error:
            raise ValueError(f"{role} protocol tool is not valid Python") from error
        definitions = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        if callable_name not in definitions:
            raise ValueError(
                f"{role} protocol tool lacks callable {callable_name}"
            )
        paths[role] = path
    return paths


def prerequisite_artifact_paths(spec: dict[str, Any]) -> dict[str, Path]:
    """Resolve schema-v4 prerequisite evidence within the frozen artifact root."""
    if schema_version(spec) != 4:
        return {}
    return {
        role: _resolve_confined_file(
            binding,
            R5_ARTIFACT_ROOT,
            f"{role} prerequisite",
        )
        for role, binding in spec["prerequisite_artifacts"].items()
    }


def _require_r5_hashes(
    values: Mapping[str, str],
    expected: set[str],
    label: str,
) -> dict[str, str]:
    if not isinstance(values, Mapping) or set(values) != expected:
        raise ValueError(f"{label} must bind exactly {sorted(expected)}")
    normalized = dict(values)
    if any(not _is_sha256(value) for value in normalized.values()):
        raise ValueError(f"{label} contains an invalid SHA-256")
    return normalized


def _r5_passing_prerequisites() -> dict[str, dict[str, Any]]:
    bindings = {}
    for role in sorted(R5_PREREQUISITE_ROLES):
        relative = R5_PREREQUISITE_PATHS[role]
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_file():
            raise ValueError(f"{role} prerequisite is missing: {path}")
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{role} prerequisite is not valid JSON") from error
        if (
            not isinstance(payload, dict)
            or raw != canonical_json_bytes(payload)
            or payload.get("passed") is not True
        ):
            raise ValueError(f"{role} prerequisite is not canonical passing evidence")
        bindings[role] = {
            "path": relative,
            "sha256": sha256_bytes(raw),
            "passed": True,
        }
    return bindings


def build_r5_run_spec(
    *,
    manifest_sha256: Mapping[str, str],
    founder_index_sha256: str,
    simulator_source_sha256: str,
    simulator_config_sha256: str,
    repository_commits: Mapping[str, str],
) -> dict[str, Any]:
    """Build the sole schema-v4 profile after every prerequisite has passed."""
    manifests = _require_r5_hashes(
        manifest_sha256,
        set(_MANIFEST_ROLES),
        "r5 manifest hashes",
    )
    if not _is_sha256(founder_index_sha256):
        raise ValueError("founder_index_sha256 is invalid")
    if not _is_sha256(simulator_source_sha256):
        raise ValueError("simulator_source_sha256 is invalid")
    if not _is_sha256(simulator_config_sha256):
        raise ValueError("simulator_config_sha256 is invalid")
    commits = dict(repository_commits)
    if set(commits) != {"microcosmos", "shinkaevolve"} or any(
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{40}", value) is None
        for value in commits.values()
    ):
        raise ValueError("repository_commits must bind two lowercase Git commits")

    tools = {}
    for role, relative in R5_PROTOCOL_TOOL_PATHS.items():
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_file():
            raise ValueError(f"{role} protocol tool is missing: {path}")
        tools[role] = {"path": relative, "sha256": sha256_file(path)}

    protocol_path = (PROJECT_ROOT / R5_PROTOCOL_DOCUMENT_PATH).resolve()
    if not protocol_path.is_file():
        raise ValueError("r5 protocol document is missing")
    initial_path = TASK_DIR / "initial_r4.py"
    baseline_path = (PROJECT_ROOT / R5_BASELINE_SOURCE_PATH).resolve()
    source_paths = {
        "initial": initial_path,
        "evaluator": TASK_DIR / "evaluate.py",
        "analysis": (PROJECT_ROOT / R5_PROTOCOL_TOOL_PATHS["final_analysis"]).resolve(),
        "baseline": baseline_path,
        "dependency_lock": DEPENDENCY_LOCK_PATH,
        "launcher": TASK_DIR / "run_evo.py",
        "finalist_selector": TASK_DIR / "freeze_finalist.py",
        "lineage_selector": TASK_DIR / "program_lineage.py",
        "run_spec_module": Path(__file__).resolve(),
        "adaptive_selector": TASK_DIR / "r4_selection.py",
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"r5 trusted source is missing: {missing}")
    sources = {name: sha256_file(path) for name, path in source_paths.items()}
    sources["simulator"] = simulator_source_sha256

    spec: dict[str, Any] = {
        "schema_version": 4,
        "run_id": R5_RUN_ID,
        "generations": 51,
        "proposal_budget": 50,
        "outer_seed": R5_OUTER_SEED,
        "model": R5_MODEL,
        "headless_command": R5_HEADLESS_COMMAND,
        "top_k": 5,
        "numerical_repeats": 3,
        "simulator_config_sha256": simulator_config_sha256,
        "source_sha256": sources,
        "archive": {
            "archive_size": 32,
            "num_archive_inspirations": 1,
            "num_islands": 2,
            "num_top_k_inspirations": 1,
        },
        "bootstrap": dict(R5_BOOTSTRAP),
        "baselines": list(R5_BASELINES),
        "search": dict(R5_SEARCH),
        "manifests": {
            role: {"path": R5_MANIFEST_PATHS[role], "sha256": manifests[role]}
            for role in sorted(_MANIFEST_ROLES)
        },
        "founder_index": {
            "path": R5_FOUNDER_INDEX_PATH,
            "sha256": founder_index_sha256,
        },
        "candidate_output_width": 6,
        "candidate_contract": {
            "version": R4_CONTRACT,
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
            "path": initial_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sources["initial"],
        },
        "artifact_roots": dict(R5_ARTIFACT_ROOTS),
        "protocol_revision": R5_PROTOCOL_REVISION,
        "protocol_tools": tools,
        "repository_commits": commits,
        "prerequisite_artifacts": _r5_passing_prerequisites(),
        "protocol_document": {
            "path": R5_PROTOCOL_DOCUMENT_PATH,
            "sha256": sha256_file(protocol_path),
        },
    }
    _validate(spec)
    protocol_tool_paths(spec)
    return spec


def publish_r5_run_spec(
    spec: dict[str, Any],
    path: str | Path | None = None,
) -> str:
    """Publish one canonical schema-v4 profile and matching sidecar."""
    _validate(spec)
    protocol_tool_paths(spec)
    target = (
        (PROJECT_ROOT / R5_PROFILE_PATH).resolve()
        if path is None
        else Path(path).resolve()
    )
    raw = canonical_json_bytes(spec)
    digest = sha256_bytes(raw)
    write_once(target, raw)
    write_once(
        target.with_suffix(".sha256"),
        f"{digest}  {target.name}\n".encode("ascii"),
    )
    return digest


def resolve_run_spec_path(
    path: str | Path | None = None,
    *,
    profile: str | None = None,
) -> Path:
    """Resolve one explicit protocol profile, defaulting to the immutable pilot."""
    if path is not None and profile is not None:
        raise ValueError("run-spec path and profile are mutually exclusive")
    if profile is not None:
        if not _PROFILE.fullmatch(profile):
            raise ValueError("profile has an invalid portable name")
        selected = PROFILE_DIR / f"{profile}.json"
    elif path is not None:
        selected = Path(path).expanduser()
    else:
        selected = SPEC_PATH
    return selected.resolve()


def load_run_spec(
    path: str | Path | None = None,
    *,
    profile: str | None = None,
) -> tuple[dict[str, Any], str, bytes]:
    selected = resolve_run_spec_path(path, profile=profile)
    raw = selected.read_bytes()
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("run specification is not valid JSON") from error
    if not isinstance(spec, dict) or raw != canonical_json_bytes(spec):
        raise ValueError("run specification is not canonical JSON")
    _validate(spec)
    if schema_version(spec) == 4:
        protocol_tool_paths(spec)
    digest = sha256_bytes(raw)
    sidecar = selected.with_suffix(".sha256")
    if sidecar.read_text(encoding="utf-8") != f"{digest}  {selected.name}\n":
        raise ValueError("run specification SHA-256 sidecar does not match")
    return spec, digest, raw


def schema_version(spec: dict[str, Any]) -> int:
    # Several focused unit fixtures predate the explicit field. Loaded specs are
    # always strictly validated, while these in-memory pilot fixtures retain
    # their original schema-1 meaning.
    return int(spec.get("schema_version", 1))


def candidate_output_width(spec: dict[str, Any]) -> int:
    """Return the trusted candidate ABI width; the pilot is permanently four."""
    return int(spec.get("candidate_output_width", 4))


def candidate_contract_version(spec: dict[str, Any]) -> str:
    """Return the statically selected candidate ABI."""
    if schema_version(spec) < 3:
        return LEGACY_CONTRACT
    return str(spec["candidate_contract"]["version"])


def artifact_path(binding: dict[str, str]) -> Path:
    """Resolve a schema-v2 project-relative immutable artifact binding."""
    return (PROJECT_ROOT / binding["path"]).resolve()


def initial_program_path(spec: dict[str, Any]) -> Path:
    if schema_version(spec) < 3:
        return TASK_DIR / "initial.py"
    return artifact_path(spec["initial_program"])


def manifest_path(spec: dict[str, Any], role: str) -> Path:
    if role not in _MANIFEST_ROLES:
        raise ValueError(f"unknown manifest role: {role}")
    if schema_version(spec) == 1:
        visible = PROJECT_ROOT / "microcosmos" / "experiments" / "evo2_ecosystem"
        if role == "sealed":
            return (
                PROJECT_ROOT
                / "microcosmos"
                / "experiments"
                / "evo2_sealed"
                / "final.json"
            )
        return visible / "manifests" / f"{role}.json"
    return artifact_path(spec["manifests"][role])


def manifest_hash(spec: dict[str, Any], role: str) -> str:
    if role not in _MANIFEST_ROLES:
        raise ValueError(f"unknown manifest role: {role}")
    if schema_version(spec) == 1:
        return str(spec["manifest_sha256"][role])
    return str(spec["manifests"][role]["sha256"])


def founder_index_path(spec: dict[str, Any]) -> Path | None:
    if schema_version(spec) == 1:
        return None
    return artifact_path(spec["founder_index"])


def holdout_paths(spec: dict[str, Any]) -> tuple[Path, Path]:
    return manifest_path(spec, "development"), manifest_path(spec, "sealed")


def bound_protocol_paths(spec: dict[str, Any]) -> dict[str, Path]:
    if schema_version(spec) == 1:
        return {"preregistration": PREREGISTRATION_PATH}
    if schema_version(spec) == 4:
        return {"protocol_document": artifact_path(spec["protocol_document"])}
    return {
        "preregistration": artifact_path(spec["preregistration"]),
        "implementation_plan": artifact_path(spec["implementation_plan"]),
    }


def paths_for(spec: dict[str, Any], regime: str) -> RunPaths:
    if regime not in REGIMES:
        raise ValueError("regime must be stable or punctuated")
    if schema_version(spec) == 1:
        results_root = TASK_DIR / "results"
        frozen_root = TASK_DIR / "frozen"
    else:
        results_root = artifact_path({"path": spec["artifact_roots"]["results"]})
        frozen_root = artifact_path({"path": spec["artifact_roots"]["frozen"]})
    run_root = results_root / spec["run_id"]
    return RunPaths(
        run_root=run_root,
        arm_results=run_root / regime,
        selection=run_root / "selection" / regime,
        frozen=frozen_root / spec["run_id"] / regime,
    )


def structured_random_paths(spec: dict[str, Any]) -> StructuredRandomPaths:
    """Derive the one schema-v4 structured-random control layout."""
    if schema_version(spec) != 4:
        raise ValueError("structured-random control requires an r5 run specification")
    results_root = artifact_path({"path": spec["artifact_roots"]["results"]})
    frozen_root = artifact_path({"path": spec["artifact_roots"]["frozen"]})
    run_root = results_root / spec["run_id"]
    control_root = run_root / "structured_random"
    return StructuredRandomPaths(
        run_root=run_root,
        control_root=control_root,
        roster=control_root / "roster",
        training=control_root / "training",
        selection=control_root / "selection",
        frozen=frozen_root / spec["run_id"] / "structured_random",
    )


def write_once(path: Path, payload: bytes) -> None:
    """Atomically publish immutable run metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to replace immutable file at {path}")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_run_spec(run_root: Path, raw: bytes, digest: str) -> None:
    if (
        run_root.is_dir()
        and any(run_root.iterdir())
        and not (
            (run_root / "run_spec.json").is_file()
            and (run_root / "run_spec.sha256").is_file()
        )
    ):
        raise RuntimeError("nonempty run directory lacks its canonical specification")
    run_root.mkdir(parents=True, exist_ok=True)
    write_once(run_root / "run_spec.json", raw)
    write_once(run_root / "run_spec.sha256", f"{digest}  run_spec.json\n".encode())
