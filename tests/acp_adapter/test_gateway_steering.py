"""Session steering on the shipped gateway ACP path (ACP steering tracker).

A gateway-backed session serves turns exactly like the direct ``HermesACPAgent``,
so it must advertise ``_meta.steering.supported`` and answer ``_session/steering``
with ``injected`` while a turn runs. Otherwise an editor sees steering as
unsupported and silently queues a mid-turn prompt instead of steering it.

The end-to-end test runs an actual ``hermes acp`` stdio process against an
ordinary isolated gateway whose provider streams slowly enough to hold the turn
open while the editor steers.
"""
import asyncio
from types import SimpleNamespace

import pytest

from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.fakes.fake_llm_provider import FakeLLMServer, Text
from tests.waiting import async_wait_until

LONG_ANSWER = "STEADY " * 200


@pytest.fixture
def model_peer():
    """The daemon's provider: every turn streams slowly, so a steer lands mid-turn."""
    with FakeLLMServer([Text(LONG_ANSWER, delay_per_chunk=0.02) for _ in range(4)]) as provider:
        yield SimpleNamespace(server_port=provider.port)


def _steering(acp, sid, text):
    return acp.rpc("_session/steering", sessionId=sid, prompt=[{"type": "text", "text": text}])


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_initialize_advertises_steering(daemon, tmp_path):
    async with editor(daemon, tmp_path) as acp:
        initialized = await acp.rpc("initialize", protocolVersion=1, clientCapabilities={})
        meta = initialized["result"].get("_meta") or {}
        assert meta.get("steering", {}).get("supported") is True, initialized


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_steering_injects_into_the_running_turn(daemon, tmp_path):
    async with viewer(daemon) as ws, editor(daemon, tmp_path) as acp:
        await acp.rpc("initialize", protocolVersion=1, clientCapabilities={})
        sid = (await acp.rpc("session/new", cwd=str(tmp_path), mcpServers=[]))["result"]["sessionId"]
        turn = asyncio.create_task(acp.rpc("session/prompt", sessionId=sid,
                                           prompt=[{"type": "text", "text": "start a long turn"}]))
        await async_wait_until(lambda: _running(ws, sid), timeout=30)
        steered = await _steering(acp, sid, "MID-TURN CORRECTION")
        assert steered["result"] == {"outcome": "injected"}, steered
        assert (await turn)["result"]["stopReason"] == "end_turn"


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_steering_asks_for_a_prompt_when_the_session_is_unknown(daemon, tmp_path):
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc("initialize", protocolVersion=1, clientCapabilities={})
        steered = await _steering(acp, "not-a-session", "hello")
        assert steered["result"] == {"outcome": "promptRequired"}, steered


@pytest.mark.asyncio
async def test_steering_fences_the_running_turn_generation(monkeypatch):
    """The correction names the live claim: idle/compression fall back to a new prompt."""
    from acp_adapter.gateway_server import GatewayACPAgent

    agent = GatewayACPAgent()
    calls = []

    class FakeClient:
        async def rpc(self, verb, **params):
            calls.append((verb, params))
            return {"status": "redirected" if verb == "session.redirect" else "accepted"}

    async def fake_client():
        return FakeClient()

    monkeypatch.setattr(agent, "_client", fake_client)
    payload = {"sessionId": "s1", "prompt": [{"type": "text", "text": "fix it"}]}

    agent._snapshots["s1"] = {"execution_generation": 7, "running": True}
    assert (await agent.ext_method("session/steering", payload)) == {"outcome": "injected"}
    assert calls == [("session.redirect", {"session_id": "s1", "text": "fix it", "execution_generation": 7})]

    calls.clear()
    agent._snapshots["s1"] = {"execution_generation": 8, "running": False}
    assert (await agent.ext_method("session/steering", payload)) == {"outcome": "injected"}
    assert [verb for verb, _ in calls] == ["prompt.submit"]
    assert "execution_generation" not in calls[0][1]


async def _running(ws, sid):
    """True once the owner reports a live turn for ``sid``."""
    try:
        return (await ws.rpc("session.resume", session_id=sid)).get("running") is True
    except Exception:  # a resume race while the row is being opened
        return False
