"""Canonical, shared specification for the matched Evo² outer searches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


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

LEGACY_CONTRACT = "evo2-heredity-r3-v1"
R4_CONTRACT = "evo2-heredity-r4-v1"


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    arm_results: Path
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


def _validate(spec: dict[str, Any]) -> None:
    if not isinstance(spec.get("schema_version"), int):
        raise ValueError("run specification has the wrong schema")
    if spec["schema_version"] == 1:
        _validate_v1(spec)
    elif spec["schema_version"] == 2:
        _validate_v2(spec)
    elif spec["schema_version"] == 3:
        _validate_v3(spec)
    else:
        raise ValueError("run specification has the wrong schema")
    _validate_shared(spec)


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
