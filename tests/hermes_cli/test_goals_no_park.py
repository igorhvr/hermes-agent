"""End-to-end integration tests for the /goal no-park feature (US-007).

Exercises the REAL chain against a temp HERMES_HOME — real imports, a real
``GoalManager``, real ``save_goal``/``load_goal`` persistence, and the real
``evaluate_after_turn`` → ``judge_goal`` path. The only seam mocked is
``agent.auxiliary_client.call_llm`` (the model boundary); nothing about the
goal layer itself is stubbed.

Covers the full validation matrix from the task spec:

1. ``/goal no-park: #t Implement login`` sets ``no_park=True``, the judge
   system prompt omits the WAIT section, and a ``wait`` verdict is
   downgraded to ``continue`` (no wait barrier).
2. ``/goal no-park: #f Implement login`` sets ``no_park=False``; a ``wait``
   verdict parks the loop (barrier set) exactly as before.
3. ``/goal Implement login`` (no prefix) is backward compatible — False.
4. ``/goal no-park: yes Implement login`` errors with the exact message and
   persists no goal.
5. A stored ``state_meta`` row without a ``no_park`` key loads as ``False``.
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

# A judge verdict that, in park-enabled mode, sets a pid wait barrier.
WAIT_RESPONSE = (
    '{"verdict": "wait", "reason": "waiting on build", "wait_on_pid": 4242}'
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


def _detailed_assert_no_park_state(mgr, *, expected_no_park):
    """Assert the manager's active state carries the expected flag."""
    assert mgr.state is not None
    assert mgr.state.goal.strip() != ""
    assert mgr.state.no_park is expected_no_park
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

    SESSION = "e2e-session"

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


class TestGoalNoParkE2E:
    """Full-chain verification: CLI -> real GoalManager -> persistence -> judge."""

    def test_no_park_true_sets_persists_and_downgrades_wait(self, hermes_home):
        """Scenario (1): /goal no-park: #t sets True and NEVER parks the loop."""
        from hermes_cli import goals

        harness = _RealGoalHarness()
        harness.run_goal_command("/goal no-park: #t Implement login")

        # The command acted on the real manager bound to the temp HERMES_HOME.
        _detailed_assert_no_park_state(harness.mgr, expected_no_park=True)

        # ── Real persistence: a fresh GoalManager on the same session reloads
        # no_park=True from the temp SessionDB (save_goal -> load_goal).
        fresh = goals.GoalManager(session_id=_RealGoalHarness.SESSION)
        _detailed_assert_no_park_state(fresh, expected_no_park=True)

        # ── Judge through the real evaluate_after_turn, mocking only the
        # model boundary. A `wait` verdict must be downgraded to `continue`
        # because state.no_park is True.
        captured, decision = _evaluate_with_mocked_judge(
            harness.mgr, WAIT_RESPONSE
        )
        system_msg = _system_prompt_from(captured)
        assert "Picking WAIT parks the loop" not in system_msg
        assert "wait_on_pid" not in system_msg
        assert "Decide one of three verdicts" in system_msg
        # The WAIT paragraph is not the only wait mention the no-park build
        # must drop — the reply-shapes section must not teach the wait verdict.
        assert "wait_on_session" not in system_msg

        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        # No wait barrier may be built for a no-park goal.
        assert harness.mgr.state.waiting_on_pid is None
        assert harness.mgr.state.waiting_until == 0
        assert harness.mgr.is_waiting() is False

    def test_no_park_false_keeps_wait_verdict_and_parks(self, hermes_home):
        """Scenario (2): /goal no-park: #f keeps normal WAIT-park behavior."""
        from hermes_cli import goals

        harness = _RealGoalHarness()
        harness.run_goal_command("/goal no-park: #f Implement login")

        _detailed_assert_no_park_state(harness.mgr, expected_no_park=False)
        fresh = goals.GoalManager(session_id=_RealGoalHarness.SESSION)
        _detailed_assert_no_park_state(fresh, expected_no_park=False)

        # A `wait` verdict with no_park=False must park the loop: the sent
        # system prompt teaches WAIT and the decision sets a pid barrier.
        captured, decision = _evaluate_with_mocked_judge(
            harness.mgr, WAIT_RESPONSE
        )
        system_msg = _system_prompt_from(captured)
        assert "Picking WAIT parks the loop" in system_msg
        assert decision["verdict"] == "wait"
        assert decision["should_continue"] is False
        # The judge-WAIT branch parks the loop on pid 4242. Note: whether a
        # LATER is_waiting() call still reports the barrier depends on whether
        # pid 4242 is alive in this environment (is_waiting self-clears when
        # the pid is gone), so we assert the barrier WAS set, not its liveness.
        assert harness.mgr.state.waiting_on_pid == 4242

    def test_no_prefix_is_backward_compatible_false(self, hermes_home):
        """Scenario (3): /goal without a prefix behaves exactly as today."""
        from hermes_cli import goals

        harness = _RealGoalHarness()
        harness.run_goal_command("/goal Implement login")

        _detailed_assert_no_park_state(harness.mgr, expected_no_park=False)
        fresh = goals.GoalManager(session_id=_RealGoalHarness.SESSION)
        _detailed_assert_no_park_state(fresh, expected_no_park=False)

        # Goal text must be unpolluted by any token.
        assert harness.mgr.state.goal == "Implement login"

    def test_invalid_no_park_value_errors_and_persists_nothing(self, hermes_home):
        """Scenario (4): /goal no-park: yes prints the exact error, sets no goal."""
        from hermes_cli import goals

        harness = _RealGoalHarness()
        cprint = harness.run_goal_command("/goal no-park: yes Implement login")

        rendered = "\n".join(str(a.args[0]) for a in cprint.call_args_list)
        assert (
            "Invalid value for no-park: must be #t or #f, got 'yes'" in rendered
        )
        # No goal may be persisted on the invalid path.
        assert harness.mgr.state is None
        assert goals.load_goal(_RealGoalHarness.SESSION) is None

    def test_legacy_row_without_no_park_key_loads_false(self, hermes_home):
        """Scenario (5): a stored row lacking no_park loads as False (backward compat)."""
        from hermes_cli import goals

        # Build a full serialized goal (via the real set/save path)…
        seeds = goals.GoalManager(session_id="legacy-session")
        seeds.set("legacy goal")
        raw = json.loads(goals.load_goal("legacy-session").to_json())
        # …then simulate a row written by an older Hermes version: the
        # no_park key is absent from the stored JSON.
        assert "no_park" in raw
        raw.pop("no_park")
        db = goals._get_session_db()
        db.set_meta(goals._meta_key("legacy-session"), json.dumps(raw))

        reloaded = goals.load_goal("legacy-session")
        assert reloaded is not None
        assert reloaded.goal == "legacy goal"
        assert reloaded.no_park is False
        # The field must also round-trip save → load without ever having been
        # truthy — no_park stays False through a further persistence cycle.
        reloaded2 = goals.GoalManager(session_id="legacy-session")
        assert reloaded2.state.no_park is False

    def test_evaluate_after_turn_propagates_no_park_true(self, hermes_home):
        """The real evaluator forwards state.no_park=True into the judge."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="e2e-propagate-true")
        mgr.set("Implement login", no_park=True)
        _detailed_assert_no_park_state(mgr, expected_no_park=True)

        captured, decision = _evaluate_with_mocked_judge(mgr, WAIT_RESPONSE)
        system_msg = _system_prompt_from(captured)
        assert "Picking WAIT parks the loop" not in system_msg
        # DONE + CONTINUE must stay fully intact in the no-park build.
        assert "Decide one of three verdicts" in system_msg
        assert decision["verdict"] == "continue"
        assert mgr.state.waiting_on_pid is None

    def test_evaluate_after_turn_propagates_no_park_false(self, hermes_home):
        """The real evaluator forwards state.no_park=False (default) as before."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="e2e-propagate-false")
        mgr.set("Implement login")
        _detailed_assert_no_park_state(mgr, expected_no_park=False)

        captured, decision = _evaluate_with_mocked_judge(mgr, WAIT_RESPONSE)
        system_msg = _system_prompt_from(captured)
        assert "Picking WAIT parks the loop" in system_msg
        assert decision["verdict"] == "wait"
        assert mgr.state.waiting_on_pid == 4242
