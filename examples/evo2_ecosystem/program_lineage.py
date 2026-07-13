"""Authenticate and freeze a selected Evo² champion's actual parent chain.

This tool is intentionally separate from the historical finalist freeze.  It
never mutates the Shinka archive or an existing frozen finalist directory.  It
publishes one canonical, write-once selection record (and its SHA-256 sidecar)
while the sealed manifest is still inaccessible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from shinka.database import DatabaseConfig, Program, ProgramDatabase

if __package__:
    from examples.evo2_ecosystem import run_spec
else:  # direct ``python examples/.../program_lineage.py``
    import run_spec


TASK_DIR = Path(__file__).resolve().parent
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _source_sha256(program: Program) -> str:
    return hashlib.sha256(program.code.encode("utf-8")).hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON at {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_run_spec(path: Path) -> tuple[dict[str, Any], bytes, str]:
    raw = path.read_bytes()
    spec = _load_json_object(path, "run specification")
    if raw != run_spec.canonical_json_bytes(spec):
        raise ValueError("run specification is not canonical JSON")
    digest = run_spec.sha256_bytes(raw)
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != (
        f"{digest}  {path.name}\n"
    ):
        raise ValueError("run specification SHA-256 sidecar does not match")
    return spec, raw, digest


def _assert_sealed_locked(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"sealed manifest is missing at {path}")
    if os.access(path, os.R_OK):
        raise RuntimeError(
            "sealed manifest must remain unreadable while program lineage is frozen"
        )


def _champion_program(
    database: ProgramDatabase,
    *,
    generation: int,
    source_sha256: str,
) -> Program:
    if database.cursor is None:
        raise ConnectionError("archive database is not connected")
    rows = database.cursor.execute(
        "SELECT id FROM programs WHERE generation = ? ORDER BY id",
        (generation,),
    ).fetchall()
    matches: list[Program] = []
    for row in rows:
        program = database.get(str(row["id"]))
        if program is not None and _source_sha256(program) == source_sha256:
            matches.append(program)
    if len(matches) != 1:
        raise ValueError(
            "declared champion does not identify exactly one archive program"
        )
    champion = matches[0]
    if not champion.correct:
        raise ValueError("declared champion is not marked correct in the archive")
    if champion.private_metrics.get("candidate_sha256") != source_sha256:
        raise ValueError("declared champion evaluator hash does not match its source")
    return champion


def _actual_parent_chain(
    database: ProgramDatabase,
    champion: Program,
) -> list[Program]:
    if database.cursor is None:
        raise ConnectionError("archive database is not connected")
    program_count = int(
        database.cursor.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
    )
    ancestors = database.get_ancestry(
        champion.id,
        max_ancestors=program_count + 1,
    )
    chain = [*ancestors, champion]
    ids = [program.id for program in chain]
    if len(ids) != len(set(ids)):
        raise ValueError("champion parent chain contains a cycle")
    if not chain or chain[0].parent_id is not None:
        raise ValueError("champion parent chain does not terminate at a root")
    for parent, child in zip(chain, chain[1:]):
        if child.parent_id != parent.id:
            raise ValueError("champion parent chain is missing a declared parent")
    return chain


def _program_record(program: Program, ordinal: int) -> dict[str, Any]:
    source_hash = _source_sha256(program)
    if program.correct and program.private_metrics.get("candidate_sha256") not in (
        None,
        source_hash,
    ):
        raise ValueError("ancestor evaluator hash does not match its source")
    score = float(program.combined_score)
    if not math.isfinite(score):
        raise ValueError("ancestor training score is not finite")
    metadata = program.metadata if isinstance(program.metadata, dict) else {}
    return {
        "ordinal": ordinal,
        "program_id": program.id,
        "parent_id": program.parent_id,
        "generation": int(program.generation),
        "candidate_source_sha256": source_hash,
        "correct": bool(program.correct),
        "training_score": score,
        "code_diff": program.code_diff or "",
        "patch": {
            key: metadata.get(key)
            for key in (
                "patch_type",
                "patch_name",
                "patch_description",
                "model_name",
                "diff_summary",
            )
        },
    }


def _deduplicate_sources(
    chain: list[dict[str, Any]],
    champion_id: str,
) -> list[dict[str, Any]]:
    """Collapse equal sources while retaining every actual parent occurrence."""
    groups: dict[str, dict[str, Any]] = {}
    for item in chain:
        source_hash = item["candidate_source_sha256"]
        if source_hash not in groups:
            groups[source_hash] = {
                "source_ordinal": len(groups),
                "candidate_source_sha256": source_hash,
                "representative_program_id": item["program_id"],
                "occurrences": [],
            }
        groups[source_hash]["occurrences"].append(
            {
                key: item[key]
                for key in ("ordinal", "program_id", "parent_id", "generation")
            }
        )
        if item["program_id"] == champion_id:
            groups[source_hash]["representative_program_id"] = champion_id
    return list(groups.values())


def _nearest_inner_index(size: int, fraction: float) -> int:
    target = (size - 1) * fraction
    return min(range(1, size - 1), key=lambda index: (abs(index - target), index))


def _select_representatives(
    distinct_sources: list[dict[str, Any]],
    chain: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not distinct_sources:
        raise ValueError("champion lineage has no distinct sources")
    source_order = [item["candidate_source_sha256"] for item in distinct_sources]
    roles: dict[str, set[str]] = {source_hash: set() for source_hash in source_order}
    root_hash = chain[0]["candidate_source_sha256"]
    final_hash = chain[-1]["candidate_source_sha256"]
    roles[root_hash].add("root")
    roles[final_hash].add("final")
    preferred_program: dict[str, str] = {
        final_hash: chain[-1]["program_id"],
    }
    if len(chain) > 1:
        parent = chain[-2]
        parent_hash = parent["candidate_source_sha256"]
        roles[parent_hash].add("direct_parent")
        preferred_program.setdefault(parent_hash, parent["program_id"])

    if len(distinct_sources) <= 5:
        selected_hashes = set(source_order)
        for source_hash in source_order:
            roles[source_hash].add("all_distinct_short_path")
    else:
        one_third = source_order[_nearest_inner_index(len(source_order), 1.0 / 3.0)]
        two_thirds = source_order[_nearest_inner_index(len(source_order), 2.0 / 3.0)]
        roles[one_third].add("one_third")
        roles[two_thirds].add("two_thirds")
        selected_hashes = {
            root_hash,
            one_third,
            two_thirds,
            chain[-2]["candidate_source_sha256"] if len(chain) > 1 else root_hash,
            final_hash,
        }

    by_hash = {item["candidate_source_sha256"]: item for item in distinct_sources}
    selected: list[dict[str, Any]] = []
    for source_hash in source_order:
        if source_hash not in selected_hashes:
            continue
        group = by_hash[source_hash]
        representative_id = preferred_program.get(
            source_hash,
            group["representative_program_id"],
        )
        occurrence = next(
            item
            for item in group["occurrences"]
            if item["program_id"] == representative_id
        )
        selected.append(
            {
                "selection_ordinal": len(selected),
                "source_ordinal": group["source_ordinal"],
                "candidate_source_sha256": source_hash,
                "representative_program_id": representative_id,
                "representative_parent_id": occurrence["parent_id"],
                "representative_generation": occurrence["generation"],
                "roles": sorted(roles[source_hash]),
            }
        )
    return selected


def export_champion_lineage(
    *,
    database_path: Path,
    freeze_record_path: Path,
    frozen_source_path: Path,
    generation_source_path: Path,
    run_spec_path: Path,
    sealed_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Verify and publish the declared champion's actual ancestry record."""
    _assert_sealed_locked(sealed_manifest_path)
    freeze_record = _load_json_object(freeze_record_path, "freeze record")
    spec, _, spec_hash = _load_run_spec(run_spec_path)
    run_id = freeze_record.get("run_id")
    regime = freeze_record.get("regime")
    generation = freeze_record.get("selected_generation")
    champion_hash = freeze_record.get("candidate_source_sha256")
    if (
        not isinstance(run_id, str)
        or spec.get("run_id") != run_id
        or regime not in run_spec.REGIMES
        or not isinstance(generation, int)
        or isinstance(generation, bool)
        or not isinstance(champion_hash, str)
        or _SHA256.fullmatch(champion_hash) is None
        or freeze_record.get("run_spec_sha256") != spec_hash
    ):
        raise ValueError("freeze record does not declare a valid run champion")
    if freeze_record.get("sealed_manifest_locked") is not True:
        raise ValueError("freeze record does not attest sealed-manifest isolation")
    sealed_hash = freeze_record.get("sealed_manifest_sha256")
    sealed_sidecar = sealed_manifest_path.with_suffix(".sha256")
    if (
        not isinstance(sealed_hash, str)
        or _SHA256.fullmatch(sealed_hash) is None
        or not sealed_sidecar.is_file()
        or sealed_sidecar.read_text(encoding="utf-8")
        != f"{sealed_hash}  {sealed_manifest_path.name}\n"
    ):
        raise ValueError("sealed manifest sidecar does not match the freeze record")

    database_hash = run_spec.sha256_file(database_path)
    if freeze_record.get("database_sha256") != database_hash:
        raise ValueError("archive database hash does not match the freeze record")
    wal_path = Path(f"{database_path}-wal")
    if wal_path.exists() and wal_path.stat().st_size:
        raise ValueError("archive database has an active write-ahead log")
    for path, label in (
        (frozen_source_path, "frozen champion"),
        (generation_source_path, "generation champion"),
    ):
        if not path.is_file() or run_spec.sha256_file(path) != champion_hash:
            raise ValueError(f"{label} source does not match the declared champion")

    database = ProgramDatabase(
        DatabaseConfig(db_path=str(database_path)),
        embedding_model="",
        read_only=True,
    )
    try:
        champion = _champion_program(
            database,
            generation=generation,
            source_sha256=champion_hash,
        )
        programs = _actual_parent_chain(database, champion)
    finally:
        database.close()

    chain = [
        _program_record(program, ordinal) for ordinal, program in enumerate(programs)
    ]
    distinct_sources = _deduplicate_sources(chain, champion.id)
    selected = _select_representatives(distinct_sources, chain)
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "regime": regime,
        "run_spec_sha256": spec_hash,
        "database_sha256": database_hash,
        "freeze_record_sha256": run_spec.sha256_file(freeze_record_path),
        "lineage_selector_source_sha256": run_spec.sha256_file(Path(__file__)),
        "trusted_source_sha256": spec.get("source_sha256", {}),
        "sealed_manifest_sha256": sealed_hash,
        "sealed_manifest_locked": True,
        "champion": {
            "program_id": champion.id,
            "generation": generation,
            "candidate_source_sha256": champion_hash,
        },
        "actual_parent_chain_length": len(chain),
        "distinct_source_count": len(distinct_sources),
        "actual_parent_chain": chain,
        "distinct_sources": distinct_sources,
        "selection_rule": {
            "short_path_max_distinct_sources": 5,
            "short_path": "all distinct sources",
            "long_path": "root, nearest 1/3, nearest 2/3, direct parent, final",
            "deduplication": "candidate source SHA-256 with all occurrences retained",
        },
        "selected_representatives": selected,
    }
    # Close the read-only archive first, then re-check holdout isolation at the
    # publication boundary to narrow the unlock race around this audit record.
    _assert_sealed_locked(sealed_manifest_path)
    payload = run_spec.canonical_json_bytes(record)
    run_spec.write_once(output_path, payload)
    digest = run_spec.sha256_bytes(payload)
    sidecar = output_path.with_suffix(".sha256")
    run_spec.write_once(sidecar, f"{digest}  {output_path.name}\n".encode())
    return record


def export_for_regime(
    regime: str,
    *,
    selected_run_spec_path: str | Path | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Export one canonical lineage using only run-spec-derived paths."""
    spec, _, _ = run_spec.load_run_spec(
        selected_run_spec_path,
        profile=profile,
    )
    paths = run_spec.paths_for(spec, regime)
    generation = _load_json_object(
        paths.frozen / "freeze_record.json",
        "freeze record",
    ).get("selected_generation")
    if not isinstance(generation, int) or isinstance(generation, bool):
        raise ValueError("freeze record has an invalid selected generation")
    return export_champion_lineage(
        database_path=paths.arm_results / "programs.sqlite",
        freeze_record_path=paths.frozen / "freeze_record.json",
        frozen_source_path=paths.frozen / "main.py",
        generation_source_path=paths.arm_results / f"gen_{generation}" / "main.py",
        run_spec_path=paths.run_root / "run_spec.json",
        sealed_manifest_path=run_spec.manifest_path(spec, "sealed"),
        output_path=paths.selection / "program_lineage_selection.json",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=run_spec.REGIMES, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run-spec", type=Path)
    selection.add_argument("--profile")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    record = export_for_regime(
        arguments.regime,
        selected_run_spec_path=arguments.run_spec,
        profile=arguments.profile,
    )
    print(
        f"Frozen {len(record['selected_representatives'])} distinct "
        f"{arguments.regime} lineage representatives"
    )


if __name__ == "__main__":
    main()
