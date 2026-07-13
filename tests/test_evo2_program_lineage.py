from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from examples.evo2_ecosystem import program_lineage, run_spec
from shinka.database import DatabaseConfig, Program, ProgramDatabase


def _hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _program(
    program_id: str,
    source: str,
    generation: int,
    parent_id: str | None,
) -> Program:
    source_hash = _hash(source)
    return Program(
        id=program_id,
        code=source,
        parent_id=parent_id,
        generation=generation,
        combined_score=float(generation) / 10.0,
        correct=True,
        private_metrics={"candidate_sha256": source_hash},
        code_diff=f"diff-{program_id}",
        metadata={"patch_description": f"patch-{program_id}"},
        island_idx=0,
    )


def _archive(path: Path) -> tuple[str, list[str]]:
    sources = [
        "def policy():\n    return 0\n",
        "def policy():\n    return 1\n",
        "def policy():\n    return 2\n",
        "def policy():\n    return 3\n",
        "def policy():\n    return 4\n",
        "def policy():\n    return 5\n",
    ]
    programs = [
        _program("root", sources[0], 0, None),
        _program("root-copy", sources[0], 1, "root"),
        _program("b", sources[1], 2, "root-copy"),
        _program("c", sources[2], 3, "b"),
        _program("d", sources[3], 4, "c"),
        _program("direct-parent", sources[4], 5, "d"),
        _program("champion", sources[5], 6, "direct-parent"),
    ]
    database = ProgramDatabase(
        DatabaseConfig(db_path=str(path), num_islands=1),
        embedding_model="",
    )
    try:
        for item in programs:
            database.add(item, defer_maintenance=True)
    finally:
        database.close()
    return sources[-1], sources


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    database_path = tmp_path / "programs.sqlite"
    champion_source, _ = _archive(database_path)
    champion_hash = _hash(champion_source)
    run_spec_path = tmp_path / "run_spec.json"
    spec = {
        "run_id": "lineage-test",
        "source_sha256": {"initial": "1" * 64, "evaluator": "2" * 64},
    }
    spec_bytes = run_spec.canonical_json_bytes(spec)
    spec_hash = run_spec.sha256_bytes(spec_bytes)
    run_spec_path.write_bytes(spec_bytes)
    run_spec_path.with_suffix(".sha256").write_text(
        f"{spec_hash}  run_spec.json\n",
        encoding="utf-8",
    )
    freeze_record_path = tmp_path / "freeze_record.json"
    freeze_record_path.write_text(
        json.dumps(
            {
                "run_id": "lineage-test",
                "regime": "punctuated",
                "selected_generation": 6,
                "candidate_source_sha256": champion_hash,
                "run_spec_sha256": spec_hash,
                "database_sha256": run_spec.sha256_file(database_path),
                "sealed_manifest_sha256": "3" * 64,
                "sealed_manifest_locked": True,
            }
        ),
        encoding="utf-8",
    )
    frozen_source_path = tmp_path / "frozen.py"
    generation_source_path = tmp_path / "generation.py"
    frozen_source_path.write_text(champion_source, encoding="utf-8")
    generation_source_path.write_text(champion_source, encoding="utf-8")
    sealed_manifest_path = tmp_path / "sealed.json"
    sealed_manifest_path.write_text("{}\n", encoding="utf-8")
    sealed_manifest_path.with_suffix(".sha256").write_text(
        f"{'3' * 64}  sealed.json\n",
        encoding="utf-8",
    )
    return {
        "database_path": database_path,
        "freeze_record_path": freeze_record_path,
        "frozen_source_path": frozen_source_path,
        "generation_source_path": generation_source_path,
        "run_spec_path": run_spec_path,
        "sealed_manifest_path": sealed_manifest_path,
        "output_path": tmp_path / "program_lineage_selection.json",
    }


def test_export_authenticates_chain_deduplicates_and_selects_representatives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    monkeypatch.setattr(program_lineage, "_assert_sealed_locked", lambda path: None)

    record = program_lineage.export_champion_lineage(**paths)

    assert record["champion"]["program_id"] == "champion"
    assert [item["program_id"] for item in record["actual_parent_chain"]] == [
        "root",
        "root-copy",
        "b",
        "c",
        "d",
        "direct-parent",
        "champion",
    ]
    assert record["actual_parent_chain_length"] == 7
    assert record["distinct_source_count"] == 6
    assert [
        item["program_id"] for item in record["distinct_sources"][0]["occurrences"]
    ] == ["root", "root-copy"]
    selected = record["selected_representatives"]
    assert [item["representative_program_id"] for item in selected] == [
        "root",
        "c",
        "d",
        "direct-parent",
        "champion",
    ]
    assert selected[0]["roles"] == ["root"]
    assert selected[-2]["roles"] == ["direct_parent"]
    assert selected[-1]["roles"] == ["final"]
    output = paths["output_path"]
    assert output.read_bytes() == run_spec.canonical_json_bytes(record)
    digest = run_spec.sha256_file(output)
    assert output.with_suffix(".sha256").read_text(encoding="utf-8") == (
        f"{digest}  {output.name}\n"
    )
    assert program_lineage.export_champion_lineage(**paths) == record


def test_short_distinct_path_selects_every_source() -> None:
    chain = [
        {
            "ordinal": index,
            "program_id": f"p{index}",
            "parent_id": None if index == 0 else f"p{index - 1}",
            "generation": index,
            "candidate_source_sha256": str(index) * 64,
        }
        for index in range(4)
    ]
    groups = program_lineage._deduplicate_sources(chain, "p3")

    selected = program_lineage._select_representatives(groups, chain)

    assert len(selected) == 4
    assert all("all_distinct_short_path" in item["roles"] for item in selected)
    assert "root" in selected[0]["roles"]
    assert "direct_parent" in selected[-2]["roles"]
    assert "final" in selected[-1]["roles"]


def test_database_hash_mismatch_is_rejected_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    monkeypatch.setattr(program_lineage, "_assert_sealed_locked", lambda path: None)
    freeze = json.loads(paths["freeze_record_path"].read_text(encoding="utf-8"))
    freeze["database_sha256"] = "0" * 64
    paths["freeze_record_path"].write_text(json.dumps(freeze), encoding="utf-8")

    with pytest.raises(ValueError, match="database hash"):
        program_lineage.export_champion_lineage(**paths)

    assert not paths["output_path"].exists()


def test_readable_sealed_manifest_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "sealed.json"
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must remain unreadable"):
        program_lineage._assert_sealed_locked(manifest)


def test_missing_parent_is_not_accepted_as_a_root(tmp_path: Path) -> None:
    database_path = tmp_path / "broken.sqlite"
    database = ProgramDatabase(
        DatabaseConfig(db_path=str(database_path), num_islands=1),
        embedding_model="",
    )
    try:
        champion = _program(
            "orphan",
            "def orphan():\n    return 1\n",
            1,
            "missing-parent",
        )
        database.add(champion, defer_maintenance=True)
    finally:
        database.close()
    readonly = ProgramDatabase(
        DatabaseConfig(db_path=str(database_path), num_islands=1),
        embedding_model="",
        read_only=True,
    )
    try:
        orphan = readonly.get("orphan")
        assert orphan is not None
        with pytest.raises(ValueError, match="terminate at a root"):
            program_lineage._actual_parent_chain(readonly, orphan)
    finally:
        readonly.close()
