"""An ACP row an older owner relabelled ``local`` with no ``created_source`` is repaired on load:
the receipt's policy is its creation provenance, so the editor thread stays in the picker."""
import pytest

from tests.acp_adapter.test_gateway_sessions import editor


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_relabelled_acp_thread_is_restamped_and_stays_listed(daemon, tmp_path):
    from hermes_state import SessionDB

    home = daemon[0]
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        sid = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']['sessionId']
    # What the older owner left behind after the first prompt: a routing label, no provenance.
    with SessionDB(db_path=home / 'state.db') as db:
        db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET source='local', created_source='unknown' WHERE id=?", (sid,)))
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        assert sid not in [row['sessionId'] for row in (await acp.rpc('session/list', cwd=str(tmp_path)))['result']['sessions']]
        loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
        assert 'result' in loaded, loaded
        turn = await acp.rpc('session/prompt', sessionId=sid,
                             prompt=[{'type': 'text', 'text': 'after the upgrade'}])
        assert turn.get('result', {}).get('stopReason') == 'end_turn', turn
        with SessionDB(db_path=home / 'state.db') as db:
            row = db.get_session(sid)
            assert (row['source'], row['created_source']) == ('local', 'acp')
        listed = await acp.rpc('session/list', cwd=str(tmp_path))
        assert sid in [row['sessionId'] for row in listed['result']['sessions']], listed


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_new_acp_thread_keeps_its_creation_provenance_after_the_first_prompt(daemon, tmp_path):
    """The reporter's fresh-session case: the routing write may relabel ``source``, but the creation
    stamp keeps the thread in the editor picker."""
    from hermes_state import SessionDB

    home = daemon[0]
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        sid = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']['sessionId']
        turn = await acp.rpc('session/prompt', sessionId=sid,
                             prompt=[{'type': 'text', 'text': 'inbox-zero'}])
        assert turn.get('result', {}).get('stopReason') == 'end_turn', turn
        with SessionDB(db_path=home / 'state.db') as db:
            row = db.get_session(sid)
            assert (row['source'], row['created_source']) == ('local', 'acp')
        listed = await acp.rpc('session/list', cwd=str(tmp_path))
        assert sid in [row['sessionId'] for row in listed['result']['sessions']], listed
