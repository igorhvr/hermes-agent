"""ACP historical catalog without storage initialization or an execution owner."""
import json
from pathlib import Path

from hermes_state import SessionDB
from acp_adapter.session import (
    _normalize_cwd_for_compare, _parse_model_config, _session_info, _updated_at_sort_key,
)


def _unset_acp_receipt_rows(db):
    """Rows whose private local receipt proves ACP creation but whose ``created_source`` an older
    owner left unset (its first routing write relabelled the row to ``local``). The receipt's own
    policy is the provenance, so the picker still lists them before the owner's boot repair runs."""
    from hermes_state_compression import _UNSET_CREATED_SQL
    from hermes_state_local import POLICY_PREFIX
    with db._read_ctx() as conn:
        # A read-only open does not migrate: a store from before the column has nothing to repair.
        if not any(row[1] == 'created_source' for row in conn.execute('PRAGMA table_info(sessions)')):
            return {}
        receipts = conn.execute('SELECT value FROM state_meta WHERE key GLOB ?', (POLICY_PREFIX + '*',)).fetchall()
    ids = []
    for (raw,) in receipts:
        try:
            receipt = json.loads(raw)
        except (TypeError, ValueError):
            continue
        policy = receipt.get('policy') if isinstance(receipt, dict) else None
        if not isinstance(policy, dict) or policy.get('source') != 'acp':
            continue
        for sid in (receipt.get('session_id'), *(receipt.get('lineage') or ())):
            if isinstance(sid, str) and sid not in ids:
                ids.append(sid)
    unset = []
    with db._read_ctx() as conn:
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            unset.extend(row[0] for row in conn.execute(
                f"SELECT id FROM sessions WHERE id IN ({','.join('?' * len(chunk))}) AND source='local' AND "
                + _UNSET_CREATED_SQL.format(col='created_source'), chunk))
    rows = {}
    for sid in unset:
        row = db.get_session(sid)
        if row is not None and not row.get('hidden') and not row.get('archived'):
            rows[sid] = row
    return rows


def read_catalog_rows(path):
    path = Path(path)
    if not path.exists():
        return {}
    from hermes_cli.gateway_session_store import SurfaceLabelReader
    # The owner stores an editor thread as ``local`` with ``created_source='acp'``; the surface
    # reader matches both labels in SQL, before the LIMIT.
    with SessionDB(db_path=path, read_only=True) as db:
        rows = SurfaceLabelReader(db).list_sessions_rich(source="acp", limit=1000)
        catalog = {str(row["id"]): dict(row) for row in rows}
        catalog.update(_unset_acp_receipt_rows(db))
        return catalog


def _local_owners(db):
    """Physical id -> retained logical owner, from each private local creation receipt's lineage.
    A canonical reset or compression advances the receipt's physical target but keeps policy/FIFO
    on the creation id. Legacy reset histories have no receipt and stay independent chats."""
    from hermes_state_local import POLICY_PREFIX
    owners = {}
    with db._read_ctx() as conn:
        rows = conn.execute('SELECT value FROM state_meta WHERE key GLOB ?', (POLICY_PREFIX + '*',)).fetchall()
    for (raw,) in rows:
        try:
            receipt = json.loads(raw)
        except (TypeError, ValueError):
            continue
        owner = receipt.get('session_id') if isinstance(receipt, dict) else None
        lineage = receipt.get('lineage') if isinstance(owner, str) else None
        for physical in lineage if isinstance(lineage, list) else ():
            if isinstance(physical, str):
                owners[physical] = owner
    return owners


def read_local_owners(path):
    path = Path(path)
    if not path.exists():
        return {}
    with SessionDB(db_path=path, read_only=True) as db:
        return _local_owners(db)


def logical_session_id(path, session_id):
    """The authority's logical id for an id an editor already holds: compression and a canonical
    reset advance the physical transcript, never the session identity (the local receipt's owner,
    else ``SessionAuthority.logical_owner``'s compression root)."""
    path = Path(path)
    if not path.exists():
        return session_id
    with SessionDB(db_path=path, read_only=True) as db:
        lineage = db.get_compression_lineage(session_id)
        root = lineage[0] if lineage else session_id
        return _local_owners(db).get(root, root)


def catalog_sessions(path, cwd=None):
    normalized = _normalize_cwd_for_compare(cwd) if cwd else None
    owners = read_local_owners(path)
    results = {}
    for sid, row in read_catalog_rows(path).items():
        # The listing projects a compressed chat's tip; the editor must hold the logical id.
        sid = row.get("_lineage_root_id") or sid
        sid = owners.get(sid, sid)
        count = int(row.get("message_count") or 0)
        session_cwd = row.get('cwd') or _parse_model_config(row.get("model_config")).get("cwd", ".")
        if count <= 0 or (normalized and _normalize_cwd_for_compare(session_cwd) != normalized):
            continue
        info = _session_info(sid, session_cwd, row.get("model") or "", count, row.get("title"),
                             row.get("preview"), row.get("last_active") or row.get("started_at"))
        # One row per logical conversation: its most recently active segment represents it.
        held = results.get(sid)
        if held is None or _updated_at_sort_key(info.get("updated_at")) > _updated_at_sort_key(held.get("updated_at")):
            results[sid] = info
    return sorted(results.values(), key=lambda row: _updated_at_sort_key(row.get("updated_at")), reverse=True)
