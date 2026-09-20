"""End-to-end integration tests for the /goal dont-ask-user-input feature (US-007).

Exercises the REAL chain against a temp HERMES_HOME — real imports, a real
``GoalManager``, real ``save_goal``/``load_goal`` persistence, and the real
``_handle_goal_command`` → ``parse_goal_flags`` → ``mgr.set`` path. The only
seams mocked are ``cli._cprint`` (output capture) and
``agent.auxiliary_client.call_llm`` (the model boundary); nothing about the
goal layer itself is stubbed.

Covers the acceptance criteria from the story spec:

1. ``/goal dont-ask-user-input: #t Implement login`` sets
   ``dont_ask_user_input=True`` with the token stripped from the goal text,
   persisted through a real GoalManager.
2. ``/goal dont-ask-user-input: #f Implement login`` sets False, and
   ``/goal Implement login`` (no prefix) is backward compatible — False.
3. ``/goal dont-ask-user-input: maybe Implement login`` prints the exact
   error and persists no goal.
4. ``/goal no-park: #t dont-ask-user-input: #t <text>`` and
   ``/goal dont-ask-user-input: #t no-park: #t <text>`` both set both flags
   and leave only ``<text>`` as the goal.
5. ``/goal draft dont-ask-user-input: #t <objective>`` sets the flag on the
   drafted goal.
"""

from __future__ import annotations

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


def _assert_flag_state(mgr, *, expected_dont_ask, expected_no_park=False):
    """Assert the manager's active state carries the expected flags + clean text."""
    assert mgr.state is not None
    assert mgr.state.dont_ask_user_input is expected_dont_ask
    assert mgr.state.no_park is expected_no_park
    return mgr.state


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


class TestGoalDontAskUserInputCliE2E:
    """Full-chain verification: CLI -> parse_goal_flags -> real GoalManager -> persistence."""

    def test_dont_ask_true_sets_persists_and_strips_token(self, hermes_home):
        """Scenario (1): /goal dont-ask-user-input: #t sets True, token stripped."""
        from hermes_cli import goals

        harness = _RealGoalHarness()
        harness.run_goal_command("/goal dont-ask-user-input: #t Implement login")

        state = _assert_flag_state(harness.mgr, expected_dont_ask=True)
        assert state.goal == "Implement login"
        assert "dont-ask-user-input" not in state.goal

        # Real persistence: a fresh GoalManager on the same session reloads
        # dont_ask_user_input=True from the temp SessionDB.
        fresh = goals.GoalManager(session_id=_RealGoalHarness.SESSION)
        _assert_flag_state(fresh, expected_dont_ask=True)
        assert fresh.state.goal == "Implement login"

    def test_dont_ask_false_explicit_default(self, hermes_home):
        """Scenario (2a): /goal dont-ask-user-input: #f keeps current behavior."""
        from hermes_cli import goals

        harness = _RealGoalHarness()
        harness.run_goal_command("/goal dont-ask-user-input: #f Implement login")

        _assert_flag_state(harness.mgr, expected_dont_ask=False)
        fresh = goals.GoalManager(session_id=_RealGoalHarness.SESSION)
        _assert_flag_state(fresh, expected_dont_ask=False)
        assert fresh.state.goal == "Implement login"

    def test_no_prefix_is_backward_compatible_false(self, hermes_home):
        """Scenario (2b): /goal without a prefix behaves exactly as today."""
        harness = _RealGoalHarness()
        harness.run_goal_command("/goal Implement login")

        _assert_flag_state(harness.mgr, expected_dont_ask=False)
        assert harness.mgr.state.goal == "Implement login"

    def test_invalid_value_errors_and_persists_nothing(self, hermes_home):
        """Scenario (3): /goal dont-ask-user-input: maybe prints the exact error."""
        from hermes_cli import goals

        harness = _RealGoalHarness()
        cprint = harness.run_goal_command(
            "/goal dont-ask-user-input: maybe Implement login"
        )

        rendered = "\n".join(str(a.args[0]) for a in cprint.call_args_list)
        assert (
            "Invalid value for dont-ask-user-input: must be #t or #f, got 'maybe'"
            in rendered
        )
        # No goal may be persisted on the invalid path.
        assert harness.mgr.state is None
        assert goals.load_goal(_RealGoalHarness.SESSION) is None

    def test_flags_compose_with_no_park_either_order(self, hermes_home):
        """Scenario (4): both flags compose in either order; only the text remains."""
        for order, session_suffix in (("park-first", "a"), ("ask-first", "b")):
            session_id = f"{_RealGoalHarness.SESSION}-{session_suffix}"
            if order == "park-first":
                cmd = "/goal no-park: #t dont-ask-user-input: #t Ship the release"
            else:
                cmd = "/goal dont-ask-user-input: #t no-park: #t Ship the release"

            harness = _RealGoalHarness(session_id=session_id)
            harness.run_goal_command(cmd)

            _assert_flag_state(
                harness.mgr, expected_dont_ask=True, expected_no_park=True
            )
            assert harness.mgr.state.goal == "Ship the release"

    def test_draft_path_sets_flag_on_drafted_goal(self, hermes_home):
        """Scenario (5): /goal draft dont-ask-user-input: #t <obj> carries the flag."""
        from hermes_cli import goals

        harness = _RealGoalHarness()

        # Script the aux model boundary so draft_contract's call returns an
        # unparseable body -> contract=None -> bare-goal fallback. The flag
        # must survive onto the drafted goal either way.
        class _FakeMsg:
            content = "not a contract"

        class _FakeChoice:
            message = _FakeMsg()

        class _FakeResp:
            choices = [_FakeChoice()]

        with patch(
            "agent.auxiliary_client.call_llm", side_effect=lambda **kw: _FakeResp()
        ):
            harness.run_goal_command(
                "/goal draft dont-ask-user-input: #t implement login"
            )

        state = _assert_flag_state(harness.mgr, expected_dont_ask=True)
        assert state.goal == "implement login"
        assert "dont-ask-user-input" not in state.goal

        fresh = goals.GoalManager(session_id=_RealGoalHarness.SESSION)
        _assert_flag_state(fresh, expected_dont_ask=True)
