"""ACP session autonomy surface: the `/yolo` slash command and the `_session/steering` extension.

Both shipped beta and untested; these are the invariant tests the commits called for — the YOLO
on/off/toggle and unknown-argument branches, and the steering redirect -> queue -> promptRequired
fallback with the extension dispatch itself.
"""

import pytest
from acp.schema import TextContentBlock
from acp.exceptions import RequestError

from tests.acp_adapter.test_acp_commands import FakeAgent, make_agent_and_state


def _text_blocks(text: str) -> list[TextContentBlock]:
    """A slash-command prompt: real schema blocks (``prompt`` keys off ``TextContentBlock``)."""
    return [TextContentBlock(type="text", text=text)]


def _wire_blocks(text: str) -> list[dict]:
    """An extension payload: the plain dict shape a client puts on the wire."""
    return [{"type": "text", "text": text}]


@pytest.fixture(autouse=True)
def _clean_yolo():
    from tools.approval import clear_session
    yield
    clear_session("")


class _AgentWithoutRedirect(FakeAgent):
    """A runtime that cannot take mid-turn correction: steering must queue instead."""

    def __init__(self):
        super().__init__()
        self._supports_active_turn_redirect = False


def _reply_text(conn) -> str:
    """The last agent message chunk the adapter pushed for the command."""
    texts = [u.content.text for _sid, u in conn.updates if getattr(u, "content", None) is not None]
    return texts[-1] if texts else ""


def _agent_and_state(fake=None):
    acp_agent, state, _default_fake, _conn = make_agent_and_state()
    if fake is not None:
        state.agent = fake
    return acp_agent, state


@pytest.mark.asyncio
async def test_yolo_on_off_and_toggle_report_and_flip_the_session_flag():
    from tools.approval import is_session_yolo_enabled

    acp_agent, state, _fake, conn = make_agent_and_state()

    await acp_agent.prompt(session_id=state.session_id, prompt=_text_blocks("/yolo on"))
    assert is_session_yolo_enabled(state.session_id) is True
    assert "YOLO mode ON" in _reply_text(conn)

    await acp_agent.prompt(session_id=state.session_id, prompt=_text_blocks("/yolo off"))
    assert is_session_yolo_enabled(state.session_id) is False
    assert "YOLO mode OFF" in _reply_text(conn)

    # No argument toggles: off -> on, and ``toggle`` flips it back.
    await acp_agent.prompt(session_id=state.session_id, prompt=_text_blocks("/yolo"))
    assert is_session_yolo_enabled(state.session_id) is True
    await acp_agent.prompt(session_id=state.session_id, prompt=_text_blocks("/yolo toggle"))
    assert is_session_yolo_enabled(state.session_id) is False
    assert "YOLO mode OFF" in _reply_text(conn)


@pytest.mark.asyncio
async def test_yolo_unknown_argument_prints_usage_and_leaves_the_flag_alone():
    from tools.approval import enable_session_yolo, is_session_yolo_enabled

    acp_agent, state, _fake, conn = make_agent_and_state()
    enable_session_yolo(state.session_id)

    await acp_agent.prompt(session_id=state.session_id, prompt=_text_blocks("/yolo maybe"))
    assert is_session_yolo_enabled(state.session_id) is True
    assert _reply_text(conn) == "Usage: /yolo [on|off]"


@pytest.mark.asyncio
async def test_yolo_is_session_scoped():
    from tools.approval import is_session_yolo_enabled

    acp_agent, state, _fake, _conn = make_agent_and_state()
    other = acp_agent.session_manager.create_session(cwd=".")

    await acp_agent.prompt(session_id=state.session_id, prompt=_text_blocks("/yolo on"))
    assert is_session_yolo_enabled(state.session_id) is True
    assert is_session_yolo_enabled(other.session_id) is False


@pytest.mark.asyncio
async def test_initialize_advertises_steering_support():
    acp_agent, _state = _agent_and_state()
    response = await acp_agent.initialize(protocol_version=1, client_capabilities={})
    assert response.field_meta["steering"] == {"supported": True}


@pytest.mark.asyncio
async def test_steering_redirects_into_a_running_turn():
    acp_agent, state = _agent_and_state()
    state.is_running = True
    fake = state.agent

    result = await acp_agent.ext_method(
        "session/steering", {"sessionId": state.session_id, "prompt": _wire_blocks("use the simpler fix")})

    assert result == {"outcome": "injected"}
    assert fake.redirects == ["use the simpler fix"]
    assert state.queued_prompts == []


@pytest.mark.asyncio
async def test_steering_queues_for_the_next_turn_when_redirect_is_unsupported():
    acp_agent, state = _agent_and_state(_AgentWithoutRedirect())
    state.is_running = True

    result = await acp_agent.ext_method(
        "session/steering", {"sessionId": state.session_id, "prompt": _wire_blocks("keep going")})

    assert result == {"outcome": "injected"}
    assert state.queued_prompts == ["keep going"]
    assert state.agent.redirects == []


@pytest.mark.asyncio
async def test_steering_queues_when_the_redirect_refuses():
    class Refusing(FakeAgent):
        def redirect(self, text):
            self.redirects.append(text)
            return False

    acp_agent, state = _agent_and_state(Refusing())
    state.is_running = True

    result = await acp_agent.ext_method(
        "session/steering", {"sessionId": state.session_id, "prompt": _wire_blocks("hold on")})

    assert result == {"outcome": "injected"}
    assert state.queued_prompts == ["hold on"]


@pytest.mark.asyncio
async def test_steering_asks_for_a_prompt_when_unknown_or_empty():
    acp_agent, state = _agent_and_state()
    state.is_running = True

    unknown = await acp_agent.ext_method(
        "session/steering", {"sessionId": "nope", "prompt": _wire_blocks("hi")})
    empty = await acp_agent.ext_method("session/steering", {"sessionId": state.session_id, "prompt": []})

    assert unknown == {"outcome": "promptRequired"}
    assert empty == {"outcome": "promptRequired"}
    assert state.queued_prompts == []


@pytest.mark.asyncio
async def test_unknown_extension_method_is_method_not_found():
    acp_agent, _state = _agent_and_state()
    with pytest.raises(RequestError) as excinfo:
        await acp_agent.ext_method("session/nope", {})
    assert excinfo.value.code == RequestError.method_not_found("_session/nope").code
    assert excinfo.value.data == {"method": "_session/nope"}
