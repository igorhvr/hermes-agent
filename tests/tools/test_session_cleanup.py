"""Tests for tools/session_cleanup.py — session artifact disk cleanup."""

import os
import json
import time
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Import the module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tools.session_cleanup import (
    _extract_session_id_from_filename,
    _get_active_session_ids,
    _human_size,
    format_prune_summary,
    prune_all_artifacts,
    prune_checkpoints,
    prune_session_files,
    SESSION_FILE_RETENTION_DAYS,
    CHECKPOINT_RETENTION_DAYS,
)


# ─── Filename extraction ────────────────────────────────────────────────────


class TestExtractSessionId:
    def test_session_json(self):
        assert _extract_session_id_from_filename("session_20260412_171123_466c.json") == "20260412_171123_466c"

    def test_session_uuid(self):
        assert _extract_session_id_from_filename("session_13bac027-2f82-4267-a0c6-1c37a032945a.json") == \
               "13bac027-2f82-4267-a0c6-1c37a032945a"

    def test_request_dump(self):
        result = _extract_session_id_from_filename(
            "request_dump_20260412_171123_466c_20260412_171125_789abc.json"
        )
        assert result == "20260412_171123_466c"

    def test_jsonl_transcript(self):
        assert _extract_session_id_from_filename("20260412_012806_2013cd04.jsonl") == "20260412_012806_2013cd04"

    def test_unknown_format(self):
        assert _extract_session_id_from_filename("random_file.txt") is None

    def test_sessions_json_state_file(self):
        """sessions.json should return None — it's not a session artifact."""
        assert _extract_session_id_from_filename("sessions.json") is None


# ─── Active session detection ────────────────────────────────────────────────


class TestGetActiveSessionIds:
    def test_returns_active_ids(self):
        db = MagicMock()
        db.list_sessions_rich.return_value = [
            {"id": "active1", "ended_at": None},
            {"id": "ended1", "ended_at": 1234567890.0},
            {"id": "active2", "ended_at": None},
        ]
        result = _get_active_session_ids(db)
        assert result == {"active1", "active2"}

    def test_returns_empty_on_error(self):
        db = MagicMock()
        db.list_sessions_rich.side_effect = Exception("DB error")
        result = _get_active_session_ids(db)
        assert result == set()


# ─── Session file pruning ────────────────────────────────────────────────────


class TestPruneSessionFiles:
    @pytest.fixture
    def sessions_dir(self, tmp_path):
        d = tmp_path / "sessions"
        d.mkdir()
        return d

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.list_sessions_rich.return_value = [
            {"id": "active_session_1", "ended_at": None},
        ]
        return db

    def _create_old_file(self, directory, filename, age_days=60, content="{}"):
        """Create a file with a modification time in the past."""
        f = directory / filename
        f.write_text(content)
        old_time = time.time() - (age_days * 86400)
        os.utime(f, (old_time, old_time))
        return f

    def test_deletes_old_session_files(self, sessions_dir, mock_db):
        self._create_old_file(sessions_dir, "session_old_ended_1.json", age_days=60)
        self._create_old_file(sessions_dir, "session_old_ended_2.json", age_days=45)

        files_del, bytes_freed = prune_session_files(
            sessions_dir, mock_db, retention_days=30
        )
        assert files_del == 2
        assert bytes_freed > 0
        assert not (sessions_dir / "session_old_ended_1.json").exists()
        assert not (sessions_dir / "session_old_ended_2.json").exists()

    def test_preserves_recent_files(self, sessions_dir, mock_db):
        # Recent file (5 days old) — should NOT be deleted
        recent = sessions_dir / "session_recent_1.json"
        recent.write_text("{}")
        old_time = time.time() - (5 * 86400)
        os.utime(recent, (old_time, old_time))

        files_del, _ = prune_session_files(sessions_dir, mock_db, retention_days=30)
        assert files_del == 0
        assert recent.exists()

    def test_preserves_active_session_files(self, sessions_dir, mock_db):
        """Files belonging to active sessions should never be deleted."""
        self._create_old_file(
            sessions_dir, "session_active_session_1.json", age_days=60
        )
        files_del, _ = prune_session_files(sessions_dir, mock_db, retention_days=30)
        assert files_del == 0
        assert (sessions_dir / "session_active_session_1.json").exists()

    def test_preserves_sessions_json_state_file(self, sessions_dir, mock_db):
        """sessions.json must never be deleted."""
        self._create_old_file(sessions_dir, "sessions.json", age_days=365)
        files_del, _ = prune_session_files(sessions_dir, mock_db, retention_days=1)
        assert files_del == 0
        assert (sessions_dir / "sessions.json").exists()

    def test_deletes_old_request_dumps(self, sessions_dir, mock_db):
        self._create_old_file(
            sessions_dir,
            "request_dump_old_ended_1_20260412_171125_789abc.json",
            age_days=60,
        )
        files_del, _ = prune_session_files(sessions_dir, mock_db, retention_days=30)
        assert files_del == 1

    def test_deletes_old_jsonl_transcripts(self, sessions_dir, mock_db):
        self._create_old_file(sessions_dir, "20260312_012806_2013cd04.jsonl", age_days=60)
        files_del, _ = prune_session_files(sessions_dir, mock_db, retention_days=30)
        assert files_del == 1

    def test_dry_run_does_not_delete(self, sessions_dir, mock_db):
        f = self._create_old_file(sessions_dir, "session_dry_run_test.json", age_days=60)
        files_del, _ = prune_session_files(
            sessions_dir, mock_db, retention_days=30, dry_run=True
        )
        # Dry run returns 0 because nothing was actually deleted
        assert files_del == 0
        assert f.exists()

    def test_nonexistent_directory(self, tmp_path, mock_db):
        fake_dir = tmp_path / "nonexistent"
        files_del, bytes_freed = prune_session_files(fake_dir, mock_db)
        assert files_del == 0
        assert bytes_freed == 0


# ─── Checkpoint pruning ──────────────────────────────────────────────────────


class TestPruneCheckpoints:
    @pytest.fixture
    def checkpoints_dir(self, tmp_path):
        d = tmp_path / "checkpoints"
        d.mkdir()
        return d

    def _create_old_checkpoint(self, directory, name, age_days=30, size_kb=10):
        """Create a checkpoint directory with some content."""
        cp = directory / name
        cp.mkdir()
        # Create some files to simulate a git repo
        (cp / "HEAD").write_text("ref: refs/heads/master\n")
        (cp / "config").write_text("[core]\n\trepositoryformatversion = 0\n")
        (cp / "HERMES_WORKDIR").write_text("/tmp/test\n")
        # Create filler data
        (cp / "objects").mkdir()
        (cp / "objects" / "data").write_bytes(b"x" * (size_kb * 1024))

        old_time = time.time() - (age_days * 86400)
        os.utime(cp, (old_time, old_time))
        return cp

    def test_deletes_old_checkpoints(self, checkpoints_dir):
        self._create_old_checkpoint(checkpoints_dir, "abc123", age_days=30)
        self._create_old_checkpoint(checkpoints_dir, "def456", age_days=20)

        dirs_del, bytes_freed = prune_checkpoints(
            checkpoints_dir, retention_days=14
        )
        assert dirs_del == 2
        assert bytes_freed > 0
        assert not (checkpoints_dir / "abc123").exists()
        assert not (checkpoints_dir / "def456").exists()

    def test_preserves_recent_checkpoints(self, checkpoints_dir):
        self._create_old_checkpoint(checkpoints_dir, "recent1", age_days=5)

        dirs_del, _ = prune_checkpoints(checkpoints_dir, retention_days=14)
        assert dirs_del == 0
        assert (checkpoints_dir / "recent1").exists()

    def test_dry_run_does_not_delete(self, checkpoints_dir):
        cp = self._create_old_checkpoint(checkpoints_dir, "dryrun1", age_days=30)
        dirs_del, _ = prune_checkpoints(
            checkpoints_dir, retention_days=14, dry_run=True
        )
        assert dirs_del == 0
        assert cp.exists()

    def test_nonexistent_directory(self, tmp_path):
        fake_dir = tmp_path / "nonexistent"
        dirs_del, bytes_freed = prune_checkpoints(fake_dir)
        assert dirs_del == 0
        assert bytes_freed == 0


# ─── prune_all_artifacts integration ─────────────────────────────────────────


class TestPruneAllArtifacts:
    @pytest.fixture
    def hermes_home(self, tmp_path):
        h = tmp_path / "hermes"
        (h / "sessions").mkdir(parents=True)
        (h / "checkpoints").mkdir(parents=True)
        return h

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.list_sessions_rich.return_value = []
        return db

    def test_returns_results_for_all_types(self, hermes_home, mock_db):
        results = prune_all_artifacts(hermes_home, mock_db)
        assert "session_files" in results
        assert "checkpoints" in results

    def test_combined_cleanup(self, hermes_home, mock_db):
        # Create old session file
        sf = hermes_home / "sessions" / "session_old_test.json"
        sf.write_text("{}")
        old_time = time.time() - (60 * 86400)
        os.utime(sf, (old_time, old_time))

        # Create old checkpoint
        cp = hermes_home / "checkpoints" / "oldcp"
        cp.mkdir()
        (cp / "HEAD").write_text("ref: refs/heads/master\n")
        os.utime(cp, (old_time, old_time))

        results = prune_all_artifacts(
            hermes_home, mock_db,
            session_retention_days=30,
            checkpoint_retention_days=14,
        )
        assert results["session_files"][0] == 1
        assert results["checkpoints"][0] == 1


# ─── Formatting ──────────────────────────────────────────────────────────────


class TestFormatPruneSummary:
    def test_no_artifacts(self):
        results = {"session_files": (0, 0), "checkpoints": (0, 0)}
        assert format_prune_summary(results) == "No stale artifacts found."

    def test_with_files(self):
        results = {"session_files": (5, 1024 * 1024 * 100), "checkpoints": (3, 1024 * 1024 * 500)}
        summary = format_prune_summary(results)
        assert "5 removed" in summary
        assert "3 removed" in summary
        assert "Total freed" in summary


class TestHumanSize:
    def test_bytes(self):
        assert _human_size(500) == "500 B"

    def test_kb(self):
        assert _human_size(2048) == "2.0 KB"

    def test_mb(self):
        assert _human_size(1024 * 1024 * 5) == "5.0 MB"

    def test_gb(self):
        assert _human_size(1024 * 1024 * 1024 * 2) == "2.0 GB"
