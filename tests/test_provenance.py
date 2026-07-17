"""Tests for run provenance capture."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deltx.common.config import DeltxConfig
from deltx.common.exceptions import ProvenanceError
from deltx.common.provenance import (
    DatasetFingerprint,
    Evaluation,
    Provenance,
    RunManifest,
    _porcelain_paths,
    capture_git_state,
    capture_package_versions,
    fingerprint_dataset,
    index_row,
    load_index,
    make_run_id,
    sha256_file,
    snapshot_config,
    utc_now_iso,
    write_run,
)

MOMENT = datetime(2026, 7, 17, 18, 42, 11, tzinfo=UTC)


def _git(args: Sequence[str], cwd: Path) -> None:
    """Run a git command in a test repository, failing loudly."""
    subprocess.run(
        ["git", *args],  # noqa: S603, S607
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """A git repository with one committed file."""
    _git(["init"], tmp_path)
    _git(["config", "user.email", "test@example.com"], tmp_path)
    _git(["config", "user.name", "Test"], tmp_path)
    (tmp_path / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _git(["add", "."], tmp_path)
    _git(["commit", "-m", "init"], tmp_path)
    return tmp_path


def _manifest(run_id: str = "2026-07-17T18-42-11Z_test") -> RunManifest:
    """A minimal but valid manifest."""
    return RunManifest(
        run_id=run_id,
        provenance=Provenance(
            timestamp_utc=utc_now_iso(),
            python_version="3.12.0",
            platform="test-platform",
        ),
        dataset=DatasetFingerprint(
            features_path="features.parquet",
            sha256="deadbeef",
            rows_available=10,
            rows_used=8,
        ),
        headline=Evaluation(
            metrics={"accuracy": 0.8, "f1_score": 0.75, "auroc": 0.9}
        ),
    )


# -- run ids ---------------------------------------------------------------


def test_make_run_id_without_tag() -> None:
    assert make_run_id(moment=MOMENT) == "2026-07-17T18-42-11Z"


def test_make_run_id_slugifies_tag() -> None:
    assert make_run_id("Gemini LOMO!", moment=MOMENT) == (
        "2026-07-17T18-42-11Z_gemini-lomo"
    )


def test_make_run_id_drops_tag_with_no_alphanumerics() -> None:
    assert make_run_id("!!!", moment=MOMENT) == "2026-07-17T18-42-11Z"


# -- hashing ---------------------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"deltx")
    assert sha256_file(path) == hashlib.sha256(b"deltx").hexdigest()


def test_sha256_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="Cannot hash"):
        sha256_file(tmp_path / "absent.bin")


# -- git state -------------------------------------------------------------


def test_porcelain_paths_keeps_first_char_of_unstaged_path() -> None:
    """Regression: ' M scripts/x.py' must not lose its leading 's'.

    git status --porcelain encodes state in two columns, so an unstaged
    modification begins with a space. Stripping the output before slicing shifts
    every such path by one character.
    """
    status = " M scripts/train_detector.py\n?? src/new.py\nM  staged.py\n"
    assert _porcelain_paths(status) == [
        "scripts/train_detector.py",
        "src/new.py",
        "staged.py",
    ]


def test_capture_git_state_outside_repo(tmp_path: Path) -> None:
    state, diff = capture_git_state(tmp_path)
    assert state.commit is None
    assert state.dirty is False
    assert diff is None


def test_capture_git_state_clean_repo(temp_repo: Path) -> None:
    state, diff = capture_git_state(temp_repo)
    assert state.commit is not None
    assert state.dirty is False
    assert state.dirty_files == []
    assert diff is None


def test_capture_git_state_records_unstaged_modification(temp_repo: Path) -> None:
    (temp_repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    state, diff = capture_git_state(temp_repo)
    assert state.dirty is True
    assert state.dirty_files == ["tracked.py"]
    assert diff is not None
    assert "x = 2" in diff
    assert state.diff_sha256 == hashlib.sha256(diff.encode("utf-8")).hexdigest()
    assert state.diff_path == "uncommitted.diff"


def test_capture_git_state_untracked_file_yields_no_diff(temp_repo: Path) -> None:
    """Untracked files are dirty but invisible to `git diff HEAD`.

    diff_path must stay None rather than point at a file write_run won't create.
    """
    (temp_repo / "new.py").write_text("y = 1\n", encoding="utf-8")
    state, diff = capture_git_state(temp_repo)
    assert state.dirty is True
    assert state.dirty_files == ["new.py"]
    assert diff is None
    assert state.diff_path is None
    assert state.diff_sha256 is None


# -- environment -----------------------------------------------------------


def test_capture_package_versions_omits_absent_packages() -> None:
    versions = capture_package_versions(["pandas", "definitely-not-a-package-xyz"])
    assert "pandas" in versions
    assert "definitely-not-a-package-xyz" not in versions


def test_snapshot_config_keeps_only_result_affecting_fields() -> None:
    snapshot = snapshot_config(DeltxConfig())
    assert snapshot["random_seed"] == 42
    assert "confidence_threshold" in snapshot
    # Locations and dead knobs are not levers and must not imply reproducibility.
    assert "classifier_path" not in snapshot
    assert "model_cache_dir" not in snapshot
    assert "batch_size" not in snapshot


# -- dataset fingerprint ---------------------------------------------------


def test_fingerprint_dataset_counts_and_hashes(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "source_dataset": ["codenet", "codenet", "droidcollection", "aigcodeset"],
            "ai_model": [None, None, "gemini", "codestral"],
        }
    )
    path = tmp_path / "features.parquet"
    frame.to_parquet(path)

    fingerprint = fingerprint_dataset(path, frame, frame.iloc[:3])

    assert fingerprint.sha256 == sha256_file(path)
    assert fingerprint.rows_available == 4
    assert fingerprint.rows_used == 3
    assert fingerprint.class_balance == {"0": 2, "1": 1}
    assert fingerprint.source_counts == {"codenet": 2, "droidcollection": 1}
    # Human rows carry ai_model=None and must never reach the generator counts.
    assert fingerprint.generator_counts == {"gemini": 1}


def test_fingerprint_dataset_tolerates_missing_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame({"label": [0, 1]})
    path = tmp_path / "bare.parquet"
    frame.to_parquet(path)

    fingerprint = fingerprint_dataset(path, frame, frame)

    assert fingerprint.source_counts == {}
    assert fingerprint.generator_counts == {}


# -- persistence -----------------------------------------------------------


def test_index_row_is_flat() -> None:
    row = index_row(_manifest(), Path("data/runs/x"))
    assert all(not isinstance(value, dict | list) for value in row.values())
    assert row["headline_f1"] == 0.75


def test_write_run_persists_manifest_report_and_diff(tmp_path: Path) -> None:
    manifest = _manifest()
    # Box glyphs guard the explicit UTF-8 encoding: Windows' default codepage
    # cannot represent them.
    report = "Results ━━━ ✓ →"

    run_dir = write_run(tmp_path, manifest, report, "diff body")

    assert (run_dir / "report.txt").read_text(encoding="utf-8") == report
    assert (run_dir / "uncommitted.diff").read_text(encoding="utf-8") == "diff body"
    stored = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stored["run_id"] == manifest.run_id


def test_write_run_omits_diff_when_clean(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, _manifest(), "report", None)
    assert not (run_dir / "uncommitted.diff").exists()


def test_write_run_appends_to_index(tmp_path: Path) -> None:
    write_run(tmp_path, _manifest("run-a"), "a")
    write_run(tmp_path, _manifest("run-b"), "b")

    lines = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == ["run-a", "run-b"]


def test_load_index_returns_rows(tmp_path: Path) -> None:
    write_run(tmp_path, _manifest("run-a"), "a")
    frame = load_index(tmp_path)
    assert len(frame) == 1
    assert frame.loc[0, "run_id"] == "run-a"


def test_load_index_without_runs_is_empty(tmp_path: Path) -> None:
    assert load_index(tmp_path).empty
