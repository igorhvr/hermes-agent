"""
Session Artifact Cleanup — Prune stale disk files that accumulate over time.

Hermes creates several on-disk artifacts per session that are never automatically
cleaned up:

  ~/.hermes/sessions/session_<id>.json      — CLI session transcript logs
  ~/.hermes/sessions/request_dump_<id>.json — API debug request dumps
  ~/.hermes/sessions/<id>.jsonl             — Gateway legacy transcript files
  ~/.hermes/checkpoints/<hash>/             — Filesystem checkpoint shadow repos

The SessionDB.prune_sessions() method only deletes DB rows. This module handles
the disk side: identifying stale files, protecting active sessions, and reclaiming
disk space.

Inspired by qwibitai/nanoclaw#1632 (auto-prune stale session artifacts).
"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Default retention periods (days)
SESSION_FILE_RETENTION_DAYS = 30
REQUEST_DUMP_RETENTION_DAYS = 7
CHECKPOINT_RETENTION_DAYS = 14
JSONL_TRANSCRIPT_RETENTION_DAYS = 30


def _get_active_session_ids(db) -> Set[str]:
    """Get session IDs that are still active (not ended) from the DB.

    Returns a set of session ID strings. On error, returns an empty set
    (fail-safe: if we can't determine active sessions, don't delete anything).
    """
    try:
        rows = db.list_sessions_rich(limit=100000, include_children=True)
        return {r["id"] for r in rows if r.get("ended_at") is None}
    except Exception as e:
        logger.warning("Could not fetch active sessions: %s", e)
        return set()


def _extract_session_id_from_filename(filename: str) -> Optional[str]:
    """Extract the session ID from a session artifact filename.

    Examples:
        session_20260412_171123_466c.json → 20260412_171123_466c
        request_dump_20260412_171123_466c_20260412_171125_789abc.json → 20260412_171123_466c
        20260412_012806_2013cd04.jsonl → 20260412_012806_2013cd04
    """
    name = filename
    if name.startswith("session_") and name.endswith(".json"):
        return name[len("session_"):-len(".json")]
    elif name.startswith("request_dump_") and name.endswith(".json"):
        # request_dump_{session_id}_{timestamp}.json
        # Session ID is the part between "request_dump_" and the last timestamp
        stem = name[len("request_dump_"):-len(".json")]
        # Session IDs look like: 20260412_171123_466c or UUID format
        # The dump adds another _YYYYMMDD_HHMMSS_ffffff suffix
        # Try splitting off the last 3 underscore-separated components (date_time_microseconds)
        parts = stem.rsplit("_", 3)
        if len(parts) >= 4:
            return "_".join(parts[:-3])
        return stem
    elif name.endswith(".jsonl"):
        return name[:-len(".jsonl")]
    return None


def prune_session_files(
    sessions_dir: Path,
    db,
    retention_days: int = SESSION_FILE_RETENTION_DAYS,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Delete stale session_*.json files from the sessions directory.

    Only deletes files for sessions that are:
    1. Older than retention_days (by file modification time)
    2. NOT currently active in the DB

    Args:
        sessions_dir: Path to ~/.hermes/sessions/
        db: SessionDB instance for checking active sessions
        retention_days: Only delete files older than this many days
        dry_run: If True, report what would be deleted without deleting

    Returns:
        (files_deleted, bytes_freed) tuple
    """
    if not sessions_dir.exists():
        return 0, 0

    active_ids = _get_active_session_ids(db)
    cutoff = time.time() - (retention_days * 86400)
    files_deleted = 0
    bytes_freed = 0

    for f in sessions_dir.iterdir():
        if not f.is_file():
            continue

        # Skip the sessions.json state file
        if f.name == "sessions.json":
            continue

        # Only process session files and request dumps
        if not (f.name.startswith("session_") or f.name.startswith("request_dump_")):
            if not f.name.endswith(".jsonl"):
                continue

        try:
            stat = f.stat()
        except OSError:
            continue

        # Skip files newer than retention period
        if stat.st_mtime > cutoff:
            continue

        # Extract session ID and check if it's active
        session_id = _extract_session_id_from_filename(f.name)
        if session_id and session_id in active_ids:
            continue

        size = stat.st_size
        if dry_run:
            logger.info("Would remove: %s (%d KB)", f.name, size // 1024)
        else:
            try:
                f.unlink()
                files_deleted += 1
                bytes_freed += size
            except OSError as e:
                logger.debug("Failed to remove %s: %s", f.name, e)

    return files_deleted, bytes_freed


def prune_checkpoints(
    checkpoints_dir: Path,
    retention_days: int = CHECKPOINT_RETENTION_DAYS,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Delete stale checkpoint shadow repos.

    Checkpoints are keyed by sha256(working_dir)[:16], not by session ID.
    We use modification time as the sole criterion for staleness.

    Args:
        checkpoints_dir: Path to ~/.hermes/checkpoints/
        retention_days: Only delete checkpoints older than this many days
        dry_run: If True, report what would be deleted without deleting

    Returns:
        (dirs_deleted, bytes_freed) tuple
    """
    if not checkpoints_dir.exists():
        return 0, 0

    cutoff = time.time() - (retention_days * 86400)
    dirs_deleted = 0
    bytes_freed = 0

    for entry in checkpoints_dir.iterdir():
        if not entry.is_dir():
            continue

        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue

        if mtime > cutoff:
            continue

        # Calculate size before deletion
        try:
            size = sum(
                f.stat().st_size
                for f in entry.rglob("*")
                if f.is_file()
            )
        except OSError:
            size = 0

        if dry_run:
            logger.info("Would remove checkpoint: %s (%d KB)", entry.name, size // 1024)
        else:
            try:
                shutil.rmtree(entry)
                dirs_deleted += 1
                bytes_freed += size
            except OSError as e:
                logger.debug("Failed to remove checkpoint %s: %s", entry.name, e)

    return dirs_deleted, bytes_freed


def prune_all_artifacts(
    hermes_home: Path,
    db,
    session_retention_days: int = SESSION_FILE_RETENTION_DAYS,
    checkpoint_retention_days: int = CHECKPOINT_RETENTION_DAYS,
    dry_run: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """Prune all stale session artifacts from disk.

    This is the main entry point for both the CLI command and automated cleanup.

    Args:
        hermes_home: Path to ~/.hermes/
        db: SessionDB instance
        session_retention_days: Retention for session files and request dumps
        checkpoint_retention_days: Retention for checkpoint directories
        dry_run: If True, report what would be deleted without deleting

    Returns:
        Dict mapping artifact type to (count_deleted, bytes_freed):
        {
            "session_files": (N, bytes),
            "checkpoints": (N, bytes),
        }
    """
    results = {}

    sessions_dir = hermes_home / "sessions"
    files_del, files_bytes = prune_session_files(
        sessions_dir, db,
        retention_days=session_retention_days,
        dry_run=dry_run,
    )
    results["session_files"] = (files_del, files_bytes)

    checkpoints_dir = hermes_home / "checkpoints"
    cp_del, cp_bytes = prune_checkpoints(
        checkpoints_dir,
        retention_days=checkpoint_retention_days,
        dry_run=dry_run,
    )
    results["checkpoints"] = (cp_del, cp_bytes)

    return results


def format_prune_summary(results: Dict[str, Tuple[int, int]]) -> str:
    """Format prune results as a human-readable summary."""
    lines = []
    total_freed = 0

    files_del, files_bytes = results.get("session_files", (0, 0))
    if files_del:
        lines.append(f"  Session files: {files_del} removed ({_human_size(files_bytes)})")
        total_freed += files_bytes

    cp_del, cp_bytes = results.get("checkpoints", (0, 0))
    if cp_del:
        lines.append(f"  Checkpoints:   {cp_del} removed ({_human_size(cp_bytes)})")
        total_freed += cp_bytes

    if not lines:
        return "No stale artifacts found."

    lines.append(f"  Total freed:   {_human_size(total_freed)}")
    return "\n".join(lines)


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
