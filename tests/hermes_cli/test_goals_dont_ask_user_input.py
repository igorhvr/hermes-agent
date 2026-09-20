"""End-to-end integration tests for the /goal dont-ask-user-input feature (US-008).

Exercises the REAL chain against a temp HERMES_HOME — real imports, a real
``GoalManager``, real ``save_goal``/``load_goal`` persistence, and the real
``evaluate_after_turn`` → ``judge_goal`` path. The only seam mocked is
``agent.auxiliary_client.call_llm`` (the model boundary); nothing about the
goal layer itself is stubbed.

Covers the full validation matrix from the task spec:

1. ``/goal dont-ask-user-input: #t Do it all`` sets ``dont_ask_user_input=True``;
   the judge system prompt actually sent by ``evaluate_after_turn`` contains
   no "needs user input → DONE" clause, and ``next_continuation_prompt``
   contains no ask-the-user phrases.
2. ``/goal dont-ask-user-input: #f Do it all`` sets False — existing behavior
   (judge prompt keeps the hand-off DONE clause; a hand-off "done" stays done).
3. ``/goal Do it all`` (no prefix) is backward compatible — False.
4. ``/goal dont-ask-user-input: maybe Do it all`` errors with the exact
   message and persists no goal.
5. ``/goal no-park: #t dont-ask-user-input: #t Do it all`` — both flags
   compose; the sent judge system prompt has neither the WAIT section nor
   the needs-user-input DONE clause.
6. A scripted judge reply ``{"verdict": "done", "reason": "the agent needs
   user input"}`` is treated as CONTINUE (not done) when the flag is True —
   and stays ``done`` when the flag is False.
7. A stored ``state_meta`` row without a ``dont_ask_user_input`` key loads
   as ``False``.
"""

from __future__ import annotations

import json

import pytest

from unittest.mock import patch

# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal-module's DB cache for each test so it re-resolves HERMES_HOME.
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────

# A judge verdict that is really a hand-off to the user: with the flag False
# it completes the goal; with the flag True it must be downgraded to continue.
USER_INPUT_HANDOFF_DONE_RESPONSE = (
    '{"verdict": "done", "reason": "the agent needs user input to proceed"}'
)

# Ask / hand-off phrases that must never appear in a no-ask prompt.
_ASK_PHRASES = (
    "ask the user",
    "ask you",
    "needs user input",
    "need input from the user",
    "needs your input",
    "your input",
)


def _system_prompt_from(captured: dict) -> str:
    """Extract the system prompt the real judge sent to call_llm."""
    sent = captured.get("messages") or []
    return next((m["content"] for m in sent if m["role"] == "system"), "")


def _evaluate_with_mocked_judge(mgr, response_body: str):
    """Run the REAL evaluate_after_turn with call_llm scripted to ``response_body``.

    Returns ``(captured_kwargs, decision_dict)``. The real ``judge_goal`` is
    invoked inside ``evaluate_after_turn``; only the model boundary
    (``agent.auxiliary_client.call_llm``) is mocked, so the system prompt the
    judge actually sent can be inspected and the verdict interpreted by the
    real parser.
    """
    captured = {}

    class _FakeMsg:
        content = response_body

    class _FakeChoice:
        message = _FakeMsg()

    class _FakeResp:
        choices = [_FakeChoice()]

    def _fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _FakeResp()

    with patch("agent.auxiliary_client.call_llm", side_effect=_fake_call_llm):
        decision = mgr.evaluate_after_turn("in progress", user_initiated=True)
    return captured, decision


def _assert_dont_ask_state(mgr, *, expected: bool):
    """Assert the manager's active state carries the expected flag."""
    assert mgr.state is not None
    assert mgr.state.goal.strip() != ""
    assert mgr.state.dont_ask_user_input is expected
    return mgr.state


# ──────────────────────────────────────────────────────────────────────
# Real CLI harness (no goal-layer mocking)
# ──────────────────────────────────────────────────────────────────────


class _RealGoalHarness:
    """A real HermesCLI bound to a REAL GoalManager on the temp HERMES_HOME.

    Nothing about the goal layer is mocked: the slash-command handler talks
    to the real ``GoalManager``, which reads/writes the real SessionDB under
    the temp HERMES_HOME. The only patch applied during a command run is
    ``cli._cprint`` (to capture output instead of rendering).
    """

    SESSION = "e2e-dont-ask-session"

    def __init__(self, session_id: str = SESSION):
        import cli as cli_mod

        self.cli_mod = cli_mod
        self.shell = cli_mod.HermesCLI(compact=True, max_turns=1)
        self.shell.session_id = session_id
        # Force the handler to construct a real GoalManager bound to the
        # temp HERMES_HOME instead of leaving a stale mock in place.
        self.shell._goal_manager = None
        self.mgr = self.shell._get_goal_manager()
        assert self.mgr is not None
        assert type(self.mgr).__name__ == "GoalManager"

    def run_goal_command(self, cmd_text: str):
        """Execute a /goal command; returns the captured _cprint mock."""
        with patch.object(self.cli_mod, "_cprint") as cprint:
            self.shell._handle_goal_command(cmd_text)
        return cprint


# ──────────────────────────────────────────────────────────────────────
# E2E validation matrix
# ──────────────────────────────────────────────────────────────────────


class TestGoalDontAskUserInputE2E:
    """Full-chain verification: CLI -> real GoalManager -> persistence -> judge."""

    def test_flag_true_no_ask_prompts_end_to_end(self, hermes_home):
        """Scenario (1): /goal dont-ask-user-input: #t → True, no-ask prompts."""
        from hermes_cli import goals

        harness = _RealGoalHarness()
        harness.run_goal_command("/goal dont-ask-user-input: #t Do it all")

        # The command acted on the real manager bound to the temp HERMES_HOME;
        # the flag token was stripped from the goal text.
        state = _assert_dont_ask_state(harness.mgr, expected=True)
        assert state.goal == "Do it all"

        # ── Real persistence: a fresh GoalManager on the same session reloads
        # dont_ask_user_input=True from the temp SessionDB.
        fresh = goals.GoalManager(session_id=_RealGoalHarness.SESSION)
        _assert_dont_ask_state(fresh, expected=True)

        # ── The continuation prompt the agent will actually receive contains
        # no ask / hand-off phrase, and tells the agent to keep going.
        prompt = fresh.next_continuation_prompt()
        for phrase in _ASK_PHRASES:
            assert phrase not in prompt
        assert "say so clearly and stop" not in prompt
        assert "work around it" in prompt
        assert "keep going" in prompt

        # ── The judge system prompt actually sent by the real
        # evaluate_after_turn drops the "needs user input → DONE" clause but
        # still describes DONE for genuine completion.
        captured, _decision = _evaluate_with_mocked_judge(
            fresh, USER_INPUT_HANDOFF_DONE_RESPONSE
        )
        system_msg = _system_prompt_from(captured)
        assert "needs user input" not in system_msg
        assert "treat this as DONE with reason describing the block" not in system_msg
        assert "The response explicitly confirms the goal was completed" in system_msg
        assert "NOT DONE" in system_msg

    def test_flag_false_keeps_existing_ask_behavior(self, hermes_home):
        """Scenario (2): /goal dont-ask-user-input: #f → False, existing behavior."""
        from hermes_cli import goals

        harness = _RealGoalHarness()
        harness.run_goal_command("/goal dont-ask-user-input: #f Do it all")

        _assert_dont_ask_state(harness.mgr, expected=False)
        fresh = goals.GoalManager(session_id=_RealGoalHarness.SESSION)
        _assert_dont_ask_state(fresh, expected=False)

        # Default behavior: the continuation prompt keeps the stop-and-ask
        # sentence and the judge prompt keeps the hand-off DONE clause.
        prompt = fresh.next_continuation_prompt()
        assert "need input from the user" in prompt
        assert "say so clearly and stop" in prompt

        captured, decision = _evaluate_with_mocked_judge(
            fresh, USER_INPUT_HANDOFF_DONE_RESPONSE
        )
        system_msg = _system_prompt_from(captured)
        assert "needs user input" in system_msg
        # The user-input hand-off lives in the BLOCKED verdict paragraph of the
        # merged prompt (BLOCKED pauses the goal so the user can re-scope),
        # not in a "treat this as DONE" clause.
        assert "BLOCKED — the goal cannot be satisfied as stated" in system_msg
        # A hand-off "done" verdict stays done — no downgrade when False.
        assert decision["verdict"] == "done"
        assert decision["should_continue"] is False

    def test_no_prefix_is_backward_compatible_false(self, hermes_home):
        """Scenario (3): /goal without a prefix behaves exactly as today."""
        from hermes_cli import goals

        harness = _RealGoalHarness()
        harness.run_goal_command("/goal Do it all")

        _assert_dont_ask_state(harness.mgr, expected=False)
        fresh = goals.GoalManager(session_id=_RealGoalHarness.SESSION)
        _assert_dont_ask_state(fresh, expected=False)

        # Goal text must be unpolluted by any token.
        assert fresh.state.goal == "Do it all"

    def test_invalid_value_errors_and_persists_nothing(self, hermes_home):
        """Scenario (4): /goal dont-ask-user-input: maybe prints the exact error."""
        from hermes_cli import goals

        harness = _RealGoalHarness()
        cprint = harness.run_goal_command("/goal dont-ask-user-input: maybe Do it all")

        rendered = "\n".join(str(a.args[0]) for a in cprint.call_args_list)
        assert (
            "Invalid value for dont-ask-user-input: must be #t or #f, "
            "got 'maybe'" in rendered
        )
        # No goal may be persisted on the invalid path.
        assert harness.mgr.state is None
        assert goals.load_goal(_RealGoalHarness.SESSION) is None

    def test_composes_with_no_park_both_flags(self, hermes_home):
        """Scenario (5): no-park: #t + dont-ask-user-input: #t compose (either order)."""
        from hermes_cli import goals

        harness = _RealGoalHarness()
        harness.run_goal_command("/goal no-park: #t dont-ask-user-input: #t Do it all")

        state = _assert_dont_ask_state(harness.mgr, expected=True)
        assert state.no_park is True
        assert state.goal == "Do it all"

        fresh = goals.GoalManager(session_id=_RealGoalHarness.SESSION)
        _assert_dont_ask_state(fresh, expected=True)
        assert fresh.state.no_park is True

        # The sent judge system prompt has NEITHER the WAIT section (no_park)
        # NOR the needs-user-input DONE clause (dont_ask_user_input).
        captured, decision = _evaluate_with_mocked_judge(
            fresh, USER_INPUT_HANDOFF_DONE_RESPONSE
        )
        system_msg = _system_prompt_from(captured)
        assert "Picking WAIT parks the loop" not in system_msg
        assert "wait_on_pid" not in system_msg
        assert "wait_on_session" not in system_msg
        assert "needs user input" not in system_msg
        assert "treat this as DONE with reason describing the block" not in system_msg
        assert "Decide one of two verdicts" in system_msg
        # The hand-off "done" is still downgraded to continue.
        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True

        # Reverse order composes too.
        harness2 = _RealGoalHarness(session_id="e2e-dont-ask-reversed")
        harness2.run_goal_command("/goal dont-ask-user-input: #t no-park: #t Do it all")
        _assert_dont_ask_state(harness2.mgr, expected=True)
        assert harness2.mgr.state.no_park is True
        assert harness2.mgr.state.goal == "Do it all"

    def test_handoff_done_downgraded_to_continue_when_flag_true(self, hermes_home):
        """Scenario (6): a done verdict whose reason is a user-input hand-off
        is treated as CONTINUE when the flag is True — the goal stays active."""
        from hermes_cli import goals

        harness = _RealGoalHarness()
        harness.run_goal_command("/goal dont-ask-user-input: #t Do it all")
        fresh = goals.GoalManager(session_id=_RealGoalHarness.SESSION)
        _assert_dont_ask_state(fresh, expected=True)

        captured, decision = _evaluate_with_mocked_judge(
            fresh, USER_INPUT_HANDOFF_DONE_RESPONSE
        )
        # The verdict was scripted as "done" — the real judge_goal must
        # downgrade it because the reason is an explicit user-input hand-off.
        assert json.loads(USER_INPUT_HANDOFF_DONE_RESPONSE)["verdict"] == "done"
        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        # The goal is NOT completed: still active with a continuation prompt.
        assert fresh.state.status == "active"
        assert decision["continuation_prompt"] is not None
        # The downgrade kept the judge's reason.
        assert "needs user input" in decision["reason"]

    def test_legacy_row_without_flag_key_loads_false(self, hermes_home):
        """Scenario (7): a stored row lacking dont_ask_user_input loads False."""
        from hermes_cli import goals

        # Build a full serialized goal (via the real set/save path)…
        seeds = goals.GoalManager(session_id="legacy-dont-ask-session")
        seeds.set("legacy goal")
        raw = json.loads(goals.load_goal("legacy-dont-ask-session").to_json())
        # …then simulate a row written by an older Hermes version: the
        # dont_ask_user_input key is absent from the stored JSON.
        assert "dont_ask_user_input" in raw
        raw.pop("dont_ask_user_input")
        db = goals._get_session_db()
        db.set_meta(goals._meta_key("legacy-dont-ask-session"), json.dumps(raw))

        reloaded = goals.load_goal("legacy-dont-ask-session")
        assert reloaded is not None
        assert reloaded.goal == "legacy goal"
        assert reloaded.dont_ask_user_input is False
        # The field must also round-trip save → load without ever having been
        # truthy — the flag stays False through a further persistence cycle.
        reloaded2 = goals.GoalManager(session_id="legacy-dont-ask-session")
        assert reloaded2.state.dont_ask_user_input is False
