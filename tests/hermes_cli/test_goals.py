"""Tests for hermes_cli/goals.py — persistent cross-turn goals."""

from __future__ import annotations

import json
import time
from unittest.mock import patch, MagicMock

import pytest


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
# _parse_judge_response
# ──────────────────────────────────────────────────────────────────────


class TestParseJudgeResponse:
    def test_clean_json_done(self):
        from hermes_cli.goals import _parse_judge_response

        verdict, reason, _pf, wait = _parse_judge_response('{"done": true, "reason": "all good"}')
        assert verdict == "done"
        assert reason == "all good"
        assert wait is None




    def test_wait_verdict_with_pid(self):
        from hermes_cli.goals import _parse_judge_response

        v, reason, pf, wait = _parse_judge_response(
            '{"verdict": "wait", "wait_on_pid": 4242, "reason": "CI running"}'
        )
        assert v == "wait"
        assert pf is False
        assert wait == {"pid": 4242}
        assert reason == "CI running"




# ──────────────────────────────────────────────────────────────────────
# judge_goal — fail-open semantics
# ──────────────────────────────────────────────────────────────────────


class TestJudgeGoal:


    def test_api_error_continues(self):
        """Judge exception → fail-open continue (don't wedge progress on judge bugs)."""
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("boom"),
        ):
            verdict, reason, _, _wd, _tf = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert "judge error" in reason.lower()

    def test_judge_says_done(self):
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content='{"done": true, "reason": "achieved"}'))]
            ),
        ):
            verdict, reason, _, _wd, _tf = goals.judge_goal("goal", "agent response")
        assert verdict == "done"
        assert reason == "achieved"

    def test_judge_is_told_to_quote_errors_verbatim_and_never_infer_a_service(self):
        """A bare provider 401 in the response must not become 'the GitHub token is invalidated' in
        the block reason (#114012): the system prompt the judge actually receives carries the rule."""
        from hermes_cli import goals

        seen = {}

        def fake_call_llm(*a, **kw):
            seen["messages"] = kw.get("messages") or a
            return MagicMock(choices=[MagicMock(message=MagicMock(content='{"verdict": "blocked", "reason": "x"}'))])

        with patch("agent.auxiliary_client.call_llm", side_effect=fake_call_llm):
            goals.judge_goal("ship it", "HTTP 401: invalidated oauth token (code: token_revoked)")
        system_text = str(seen["messages"])
        assert "quote the error text verbatim" in system_text
        assert "Never infer one the response does not name" in system_text


class TestJudgeGoalNoPark:
    """judge_goal(no_park=True) never yields the WAIT verdict.

    The no-park build of the judge system prompt omits the WAIT section, and
    any ``"wait"`` the model still returns is downgraded to ``"continue"``
    with the ``wait_directive`` dropped. With ``no_park=False`` (the default,
    including for positional callers like the kanban path) WAIT passes through
    unchanged.
    """

    WAIT_RESPONSE = (
        '{"verdict": "wait", "reason": "waiting on build", '
        '"wait_on_pid": 4242}'
    )

    def _run_judge(self, response_body, **judge_kwargs):
        """Run judge_goal against a canned judge body; capture the call."""
        from unittest.mock import patch
        from hermes_cli import goals

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
            result = goals.judge_goal("ship the feature", "in progress", **judge_kwargs)
        return captured, result

    def _system_prompt(self, captured):
        sent = captured.get("messages") or []
        return next(
            (m["content"] for m in sent if m["role"] == "system"), ""
        )

    def test_no_park_true_system_prompt_omits_wait_section(self):
        captured, result = self._run_judge(
            '{"verdict": "continue", "reason": "still working"}',
            no_park=True,
        )
        system_msg = self._system_prompt(captured)
        assert "Picking WAIT parks the loop" not in system_msg
        assert "wait_on_pid" not in system_msg
        assert "Decide one of three verdicts" in system_msg
        assert result[0] == "continue"

    def test_no_park_true_downgrades_wait_to_continue(self):
        captured, result = self._run_judge(self.WAIT_RESPONSE, no_park=True)
        verdict, reason, parse_failed, wait_directive, transport_failed = result
        assert verdict == "continue"
        assert wait_directive is None
        assert reason == "waiting on build"
        assert parse_failed is False
        assert transport_failed is False

    def test_no_park_true_downgrades_session_wait_too(self):
        captured, result = self._run_judge(
            '{"verdict": "wait", "reason": "waiting on session", '
            '"wait_on_session": "sess-1"}',
            no_park=True,
        )
        verdict, _reason, _pf, wait_directive, _tf = result
        assert verdict == "continue"
        assert wait_directive is None

    def test_no_park_false_keeps_wait_verdict_unchanged(self):
        captured, result = self._run_judge(self.WAIT_RESPONSE, no_park=False)
        verdict, _reason, parse_failed, wait_directive, transport_failed = result
        assert verdict == "wait"
        assert wait_directive == {"pid": 4242}
        assert parse_failed is False
        assert transport_failed is False

    def test_positional_call_defaults_to_no_park_false(self):
        # The kanban path calls judge_goal(goal_text, last_response)
        # positionally — must default to no_park=False and keep WAIT intact.
        captured, result = self._run_judge(self.WAIT_RESPONSE)
        verdict, _reason, _pf, wait_directive, _tf = result
        assert verdict == "wait"
        assert wait_directive == {"pid": 4242}
        # The no-park flag was left at its default, so the sent prompt still
        # instructs the judge on the WAIT verdict.
        assert "Picking WAIT parks the loop" in self._system_prompt(captured)


class TestJudgeGoalDontAskUserInput:
    """judge_goal(dont_ask_user_input=True) for fully unattended goals.

    The hand-off guard downgrades a ``done`` verdict whose only basis is an
    explicit user-input hand-off to ``continue`` (an unattended goal has no
    user to hand off to, so a block must keep the loop pushing). With
    ``dont_ask_user_input=False`` (the default, including for positional
    callers like the kanban path) the same body stays ``done`` — behavior is
    unchanged. The guard is deterministic and phrase-based; it must never
    over-downgrade a genuine completion, and bare ``blocked``/``stuck``
    wording alone is not treated as a hand-off.
    """

    HANDOFF_DONE = (
        '{"verdict": "done", "reason": "the agent needs user input to continue"}'
    )

    def _run_judge(self, response_body, **judge_kwargs):
        """Run judge_goal against a canned judge body; capture the call."""
        from hermes_cli import goals

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
            result = goals.judge_goal("deploy the service", "still working", **judge_kwargs)
        return captured, result

    def _system_prompt(self, captured):
        sent = captured.get("messages") or []
        return next((m["content"] for m in sent if m["role"] == "system"), "")

    def test_handoff_done_downgraded_to_continue_when_dont_ask_user_input(self):
        captured, result = self._run_judge(self.HANDOFF_DONE, dont_ask_user_input=True)
        verdict, reason, parse_failed, wait_directive, transport_failed = result
        assert verdict == "continue"
        # the reason is preserved through the downgrade
        assert reason == "the agent needs user input to continue"
        assert parse_failed is False
        assert wait_directive is None
        assert transport_failed is False

    def test_handoff_done_kept_when_dont_ask_user_input_false(self):
        # The same body with the flag False (default) is unchanged — done.
        captured, result = self._run_judge(self.HANDOFF_DONE, dont_ask_user_input=False)
        verdict, reason, parse_failed, _wd, transport_failed = result
        assert verdict == "done"
        assert reason == "the agent needs user input to continue"
        assert parse_failed is False
        assert transport_failed is False

    def test_handoff_done_kept_by_default_positional(self):
        # The kanban path calls judge_goal(goal_text, last_response)
        # positionally — dont_ask_user_input must default to False so a
        # hand-off body stays done exactly as before this feature.
        captured, result = self._run_judge(self.HANDOFF_DONE)
        verdict, _reason, _pf, _wd, _tf = result
        assert verdict == "done"

    def test_genuine_completion_not_overdowngraded(self):
        captured, result = self._run_judge(
            '{"verdict": "done", "reason": "goal achieved, deliverable produced, verification passed"}',
            dont_ask_user_input=True,
        )
        verdict, reason, _pf, _wd, _tf = result
        assert verdict == "done"
        assert "achieved" in reason

    def test_plain_block_reason_not_treated_as_handoff(self):
        # Bare 'blocked' wording alone is not an explicit user-input hand-off —
        # the prompt change already redirects plain blocks; the guard must not
        # over-downgrade a judge that returns done alongside a plain block.
        captured, result = self._run_judge(
            '{"verdict": "done", "reason": "the agent is blocked and could not progress"}',
            dont_ask_user_input=True,
        )
        verdict, _reason, _pf, _wd, _tf = result
        assert verdict == "done"

    def test_dont_ask_user_input_true_system_prompt_omits_needs_user_input_done_clause(
        self,
    ):
        captured, result = self._run_judge(
            '{"verdict": "continue", "reason": "still working"}',
            dont_ask_user_input=True,
        )
        system_msg = self._system_prompt(captured)
        assert "needs user input" not in system_msg
        assert "treat this as DONE with reason describing the block" not in system_msg
        # DONE is still described for genuine completion.
        assert "DONE — the goal is fully satisfied" in system_msg
        assert result[0] == "continue"

    def test_dont_ask_user_input_composes_with_no_park(self):
        # Both flags set: the sent system prompt has neither the WAIT section
        # nor the needs-user-input DONE clause.
        captured, result = self._run_judge(
            '{"verdict": "continue", "reason": "keep going"}',
            no_park=True,
            dont_ask_user_input=True,
        )
        system_msg = self._system_prompt(captured)
        assert "Picking WAIT parks the loop" not in system_msg
        assert "wait_on_pid" not in system_msg
        assert "needs user input" not in system_msg
        assert "treat this as DONE with reason describing the block" not in system_msg
        assert result[0] == "continue"

    def test_no_park_wait_downgrade_and_handoff_downgrade_stay_independent(self):
        # A wait verdict with dont_ask_user_input=True and no_park=False is
        # untouched (dont_ask_user_input only guards the hand-off done path).
        captured, result = self._run_judge(
            '{"verdict": "wait", "reason": "waiting on build", "wait_on_pid": 4242}',
            dont_ask_user_input=True,
        )
        verdict, _reason, _pf, wait_directive, _tf = result
        assert verdict == "wait"
        assert wait_directive == {"pid": 4242}

    def test_handoff_done_downgraded_even_when_no_park_true(self):
        # Both downgrades are independent: with no_park=True and
        # dont_ask_user_input=True, a hand-off done is still downgraded.
        captured, result = self._run_judge(
            self.HANDOFF_DONE, no_park=True, dont_ask_user_input=True
        )
        verdict, _reason, _pf, _wd, _tf = result
        assert verdict == "continue"


class TestReasonIndicatesUserInputBlock:
    """Unit tests for the deterministic hand-off guard helper."""

    def _match(self, reason):
        from hermes_cli import goals

        return goals._reason_indicates_user_input_block(reason)

    def test_explicit_handoff_phrases_match(self):
        from hermes_cli.goals import _USER_INPUT_HANDOFF_PHRASES

        examples = {
            "user input": "the agent needs user input to continue",
            "input from the user": "waiting on input from the user",
            "needs your input": "this needs your input",
            "ask the user": "I should ask the user what to do",
            "ask you": "let me ask you",
            "your input": "I await your input",
            "hand off": "hand off to the human operator",
            "handoff": "handoff to the user",
            "stopped to ask": "stopped to ask the user",
            "await your": "I await your decision",
            "waiting for your": "waiting for your reply",
            "need you to": "I need you to decide",
        }
        for phrase, sentence in examples.items():
            assert phrase in _USER_INPUT_HANDOFF_PHRASES
            assert self._match(sentence) is True

    def test_case_insensitive_match(self):
        assert self._match("The Agent Needs User Input To Continue") is True
        assert self._match("ASK THE USER what to do") is True

    def test_plain_blocked_stuck_alone_do_not_match(self):
        assert self._match("the agent is blocked") is False
        assert self._match("blocked on the build") is False
        assert self._match("stuck") is False
        assert self._match("could not make progress") is False

    def test_empty_reason_does_not_match(self):
        assert self._match("") is False
        assert self._match("no reason provided") is False


# ──────────────────────────────────────────────────────────────────────
# GoalManager lifecycle + persistence
# ──────────────────────────────────────────────────────────────────────


class TestGoalManager:

    def test_set_then_status(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-2", default_max_turns=5)
        state = mgr.set("port the thing")
        assert state.goal == "port the thing"
        assert state.status == "active"
        assert state.max_turns == 5
        assert state.turns_used == 0
        assert mgr.is_active()
        assert "active" in mgr.status_line().lower()
        assert "port the thing" in mgr.status_line()








    def test_continuation_prompt_shape(self, hermes_home):
        """The continuation prompt must include the goal text verbatim —
        and must be safe to inject as a user-role message (prompt-cache
        invariants: no system-prompt mutation)."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="cont-sid")
        mgr.set("port goal command to hermes")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "port goal command to hermes" in prompt
        assert prompt.strip()  # non-empty


class TestGoalManagerNoPark:
    """US-004 — GoalManager accepts and propagates no_park end-to-end."""

    def test_set_no_park_true_roundtrips_through_save_reload(self, hermes_home):
        from hermes_cli.goals import GoalManager

        sid = "np-roundtrip"
        mgr = GoalManager(session_id=sid)
        state = mgr.set("deploy the service", no_park=True)
        assert state.no_park is True
        # set() persisted the state; a fresh manager bound to the same
        # session must reload it with no_park intact (real save/reload path).
        mgr2 = GoalManager(session_id=sid)
        assert mgr2.state is not None
        assert mgr2.state.goal == "deploy the service"
        assert mgr2.state.no_park is True

    def test_set_without_no_park_defaults_false(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="np-default")
        state = mgr.set("do the thing")
        assert state.no_park is False

    def test_set_positional_and_contract_signatures_unchanged(self, hermes_home):
        from hermes_cli.goals import GoalContract, GoalManager

        mgr = GoalManager(session_id="np-positional")
        # Positioning goal only (existing signature) — no_park defaults False.
        s1 = mgr.set("x")
        assert s1.goal == "x" and s1.no_park is False
        # goal + keyword contract (the signature the draft/gateway paths use).
        s2 = mgr.set("x", contract=GoalContract())
        assert s2.goal == "x" and s2.no_park is False
        # goal + keyword no_park.
        s3 = mgr.set("x", no_park=True)
        assert s3.goal == "x" and s3.no_park is True

    def test_evaluate_after_turn_forwards_no_park_true(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        judge = MagicMock(return_value=("continue", "keep going", False, None, False))
        mgr = GoalManager(session_id="np-fwd-t")
        mgr.set("leak-free build", no_park=True)
        with patch.object(goals, "judge_goal", judge):
            decision = mgr.evaluate_after_turn("built it")
        assert decision["verdict"] == "continue"
        assert judge.call_args.kwargs["no_park"] is True
        # existing judge kwargs are still passed untouched.
        assert "subgoals" in judge.call_args.kwargs
        assert "background_processes" in judge.call_args.kwargs

    def test_evaluate_after_turn_forwards_no_park_false(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        judge = MagicMock(return_value=("continue", "keep going", False, None, False))
        mgr = GoalManager(session_id="np-fwd-f")
        mgr.set("plain goal")
        assert mgr.state.no_park is False
        with patch.object(goals, "judge_goal", judge):
            decision = mgr.evaluate_after_turn("built it")
        assert decision["verdict"] == "continue"
        assert judge.call_args.kwargs["no_park"] is False
        # existing judge kwargs are still passed untouched.
        assert "subgoals" in judge.call_args.kwargs
        assert "background_processes" in judge.call_args.kwargs


class TestGoalManagerDontAskUserInput:
    """US-006 — GoalManager accepts and propagates dont_ask_user_input end-to-end."""

    def test_set_dont_ask_user_input_true_roundtrips_through_save_reload(
        self, hermes_home
    ):
        from hermes_cli.goals import GoalManager

        sid = "dua-roundtrip"
        mgr = GoalManager(session_id=sid)
        state = mgr.set("deploy the service", dont_ask_user_input=True)
        assert state.dont_ask_user_input is True
        # set() persisted the state; a fresh manager bound to the same
        # session must reload it with the flag intact (real save/reload path).
        mgr2 = GoalManager(session_id=sid)
        assert mgr2.state is not None
        assert mgr2.state.goal == "deploy the service"
        assert mgr2.state.dont_ask_user_input is True

    def test_set_without_dont_ask_user_input_defaults_false(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="dua-default")
        state = mgr.set("do the thing")
        assert state.dont_ask_user_input is False

    def test_set_composes_with_no_park(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="dua-compose")
        state = mgr.set("unattended run", no_park=True, dont_ask_user_input=True)
        assert state.no_park is True
        assert state.dont_ask_user_input is True

    def test_evaluate_after_turn_forwards_dont_ask_user_input_true(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        judge = MagicMock(return_value=("continue", "keep going", False, None, False))
        mgr = GoalManager(session_id="dua-fwd-t")
        mgr.set("leak-free build", dont_ask_user_input=True)
        with patch.object(goals, "judge_goal", judge):
            decision = mgr.evaluate_after_turn("built it")
        assert decision["verdict"] == "continue"
        assert judge.call_args.kwargs["dont_ask_user_input"] is True
        # the sibling flag is still forwarded untouched.
        assert judge.call_args.kwargs["no_park"] is False

    def test_evaluate_after_turn_forwards_dont_ask_user_input_false(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        judge = MagicMock(return_value=("continue", "keep going", False, None, False))
        mgr = GoalManager(session_id="dua-fwd-f")
        mgr.set("plain goal")
        assert mgr.state.dont_ask_user_input is False
        with patch.object(goals, "judge_goal", judge):
            decision = mgr.evaluate_after_turn("built it")
        assert decision["verdict"] == "continue"
        assert judge.call_args.kwargs["dont_ask_user_input"] is False

    def test_next_continuation_prompt_no_ask_plain(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="dua-cont-plain")
        mgr.set("deploy the service", dont_ask_user_input=True)
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        low = prompt.lower()
        assert "ask the user" not in low
        assert "need input from the user" not in low
        assert "deploy the service" in prompt

    def test_next_continuation_prompt_no_ask_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="dua-cont-sub")
        state = mgr.set("deploy the service", dont_ask_user_input=True)
        state.subgoals = ["ping the health endpoint"]
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        low = prompt.lower()
        assert "ask the user" not in low
        assert "need input from the user" not in low
        assert "ping the health endpoint" in prompt

    def test_next_continuation_prompt_no_ask_contract(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalContract, GoalManager

        mgr = GoalManager(session_id="dua-cont-contract")
        contract = GoalContract(
            outcome="service is live",
            verification="curl the health endpoint",
        )
        mgr.set("deploy the service", contract=contract, dont_ask_user_input=True)
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        low = prompt.lower()
        assert "ask the user" not in low
        assert "need input from the user" not in low
        # contract priority preserved: the contract block is present.
        assert "service is live" in prompt

    def test_next_continuation_prompt_flag_false_matches_public_templates(
        self, hermes_home
    ):
        """Backward compat: with the flag False the prompts are byte-identical
        to the public templates (contract > subgoals > plain priority)."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalContract, GoalManager

        contract = GoalContract(outcome="o", verification="v")

        mgr = GoalManager(session_id="dua-compat-plain")
        mgr.set("plain goal")
        assert mgr.next_continuation_prompt() == (
            goals.CONTINUATION_PROMPT_TEMPLATE.format(goal="plain goal")
        )

        mgr = GoalManager(session_id="dua-compat-sub")
        state = mgr.set("sub goal")
        state.subgoals = ["extra"]
        assert mgr.next_continuation_prompt() == (
            goals.CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
                goal="sub goal",
                subgoals_block=state.render_subgoals_block(),
            )
        )

        mgr = GoalManager(session_id="dua-compat-contract")
        mgr.set("contract goal", contract=contract)
        assert mgr.next_continuation_prompt() == (
            goals.CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE.format(
                goal="contract goal",
                contract_block=contract.render_block(),
            )
        )


# ──────────────────────────────────────────────────────────────────────
# Smoke: CommandDef is wired
# ──────────────────────────────────────────────────────────────────────


def test_goal_command_in_registry():
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("goal")
    assert cmd is not None
    assert cmd.name == "goal"


def test_goal_command_dispatches_in_cli_registry_helpers():
    """goal shows up in autocomplete / help categories alongside other Session cmds."""
    from hermes_cli.commands import COMMANDS, COMMANDS_BY_CATEGORY

    assert "/goal" in COMMANDS
    session_cmds = COMMANDS_BY_CATEGORY.get("Session", {})
    assert "/goal" in session_cmds


# ──────────────────────────────────────────────────────────────────────
# Auto-pause on consecutive judge parse failures
# ──────────────────────────────────────────────────────────────────────


class TestJudgeParseFailureAutoPause:
    """Regression: weak judge models (e.g. deepseek-v4-flash) that return
    empty strings or non-JSON prose must auto-pause the loop after N turns
    instead of burning the whole turn budget."""




    def test_api_error_does_not_count_as_parse_failure(self):
        """Transient network/API errors must not trip the auto-pause guard."""
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("connection reset"),
        ):
            verdict, _, parse_failed, _wd, transport_failed = goals.judge_goal(
                "goal", "response"
            )
        assert verdict == "continue"
        assert parse_failed is False
        assert transport_failed is True


    def test_auto_pause_after_three_consecutive_parse_failures(self, hermes_home):
        """N=3 consecutive parse failures → auto-pause with config pointer."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager, DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES

        assert DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES == 3
        mgr = GoalManager(session_id="parse-fail-sid-1", default_max_turns=20)
        mgr.set("do a thing")

        with patch.object(
            goals, "judge_goal", return_value=("continue", "judge returned empty response", True, None, False)
        ):
            d1 = mgr.evaluate_after_turn("step 1")
            assert d1["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 1

            d2 = mgr.evaluate_after_turn("step 2")
            assert d2["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 2

            d3 = mgr.evaluate_after_turn("step 3")
            assert d3["should_continue"] is False
            assert d3["status"] == "paused"
            assert mgr.state.consecutive_parse_failures == 3
            # Message points at the config surface so the user can fix it.
            assert "auxiliary" in d3["message"]
            assert "goal_judge" in d3["message"]
            assert "config.yaml" in d3["message"]





# ──────────────────────────────────────────────────────────────────────
# /subgoal — user-added criteria
# ──────────────────────────────────────────────────────────────────────


class TestGoalStateSubgoalsBackcompat:
    def test_old_state_meta_row_loads_without_subgoals(self):
        """A goal serialized BEFORE the subgoals field existed must
        round-trip with an empty list, not crash."""
        from hermes_cli.goals import GoalState

        legacy = json.dumps({
            "goal": "do a thing",
            "status": "active",
            "turns_used": 2,
            "max_turns": 20,
            "created_at": 1.0,
            "last_turn_at": 2.0,
            "consecutive_parse_failures": 0,
        })
        state = GoalState.from_json(legacy)
        assert state.goal == "do a thing"
        assert state.subgoals == []


class TestMigrateGoalToSession:
    """migrate_goal_to_session carries a /goal from a parent session to its
    compression continuation child (#33618). load_goal does a flat
    per-session lookup with no lineage walk, so without migration an active
    goal silently dies when compression rotates session_id."""

    def test_migrates_active_goal_to_child(self, hermes_home):
        from hermes_cli.goals import save_goal, load_goal, migrate_goal_to_session, GoalState
        save_goal("parent-sid", GoalState(goal="ship the feature"))
        assert migrate_goal_to_session("parent-sid", "child-sid", reason="compression") is True
        child = load_goal("child-sid")
        assert child is not None and child.goal == "ship the feature"
        # Parent row archived (cleared) so only the child is active.
        parent = load_goal("parent-sid")
        assert parent is not None and parent.status == "cleared"


    def test_does_not_clobber_existing_child_goal(self, hermes_home):
        from hermes_cli.goals import save_goal, load_goal, migrate_goal_to_session, GoalState
        save_goal("p3", GoalState(goal="parent goal"))
        save_goal("c3", GoalState(goal="child already has one"))
        assert migrate_goal_to_session("p3", "c3") is False
        assert load_goal("c3").goal == "child already has one"


class TestSessionDbCacheAfterProfileDelete:
    """``hermes profile delete`` force-closes every registry handle under the profile home
    (``close_all_under``) and rmtrees it; recreating the same name in the long-lived dashboard
    process must not keep persisting goals into the torn-down handle."""

    def test_delete_then_recreate_gets_a_live_store(self, hermes_home):
        import shutil

        import hermes_state
        import hermes_state_registry as registry
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.goals import GoalState, _get_session_db, load_goal, save_goal

        # conftest re-points DEFAULT_DB_PATH at one fixed file; the registry must resolve the
        # scoped profile home here, as production does.
        with patch.object(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH):
            profile = hermes_home / "profiles" / "p1"
            profile.mkdir(parents=True)
            token = set_hermes_home_override(profile)
            try:
                save_goal("s1", GoalState(goal="before delete"))
                stale = _get_session_db()
                assert registry.close_all_under(profile) == 1
                shutil.rmtree(profile)
                profile.mkdir(parents=True)

                save_goal("s2", GoalState(goal="after recreate"))

                assert _get_session_db() is not stale
                assert (profile / "state.db").exists()
                assert load_goal("s2").goal == "after recreate"
            finally:
                reset_hermes_home_override(token)
                registry.close_all_under(profile)


class TestGoalManagerSubgoals:
    def test_add_subgoal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-add")
        mgr.set("main goal")
        text = mgr.add_subgoal("  use bullet points  ")
        assert text == "use bullet points"
        assert mgr.state.subgoals == ["use bullet points"]


    def test_remove_subgoal_out_of_range(self, hermes_home):
        import pytest
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-oob")
        mgr.set("g")
        mgr.add_subgoal("only")
        with pytest.raises(IndexError):
            mgr.remove_subgoal(5)
        with pytest.raises(IndexError):
            mgr.remove_subgoal(0)


class TestJudgeGoalWithSubgoals:
    def test_judge_uses_subgoals_template_when_provided(self, hermes_home):
        """judge_goal switches templates when subgoals is non-empty.

        We don't actually call the model — we patch the aux client to
        capture the prompt that would be sent.
        """
        from unittest.mock import patch
        from hermes_cli import goals

        captured = {}

        class _FakeMsg:
            content = '{"done": true, "reason": "all done"}'
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]
        def _fake_call_llm(**kwargs):
            captured.update(kwargs)
            return _FakeResp()

        with patch("agent.auxiliary_client.call_llm", side_effect=_fake_call_llm):
            verdict, reason, parse_failed, _wd, _tf = goals.judge_goal(
                "ship the feature",
                "ok shipped",
                subgoals=["write tests", "update docs"],
            )

        # The aux client was called with a prompt that includes the subgoals.
        sent_messages = captured.get("messages") or []
        user_msg = next((m["content"] for m in sent_messages if m["role"] == "user"), "")
        assert "Additional criteria" in user_msg
        assert "1. write tests" in user_msg
        assert "2. update docs" in user_msg
        assert "every additional criterion" in user_msg
        assert verdict == "done"

    def test_judge_uses_original_template_when_no_subgoals(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals

        captured = {}

        class _FakeMsg:
            content = '{"done": true, "reason": "ok"}'
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]
        def _fake_call_llm(**kwargs):
            captured.update(kwargs)
            return _FakeResp()

        with patch("agent.auxiliary_client.call_llm", side_effect=_fake_call_llm):
            goals.judge_goal("ship it", "done", subgoals=None)

        sent_messages = captured.get("messages") or []
        user_msg = next((m["content"] for m in sent_messages if m["role"] == "user"), "")
        assert "Additional criteria" not in user_msg
        assert "ship it" in user_msg


class TestStatusLineSubgoalCount:

    def test_status_line_with_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sl-with")
        mgr.set("ship it")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        line = mgr.status_line()
        assert "2 subgoals" in line


# ──────────────────────────────────────────────────────────────────────
# Wait barrier — parking the goal loop on a background process
# ──────────────────────────────────────────────────────────────────────


class TestWaitBarrier:
    """The /goal wait barrier parks the loop on a live PID and resumes when
    the process exits, without burning turns or calling the judge."""

    @staticmethod
    def _spawn_sleeper():
        """Start a short-lived child process; return its Popen handle."""
        import subprocess
        import sys
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    @staticmethod
    def _dead_pid():
        """A PID that is essentially guaranteed not to be running."""
        return 2_000_000_000


    def test_parked_on_live_pid_does_not_continue_or_judge(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="wb-live")
            mgr.set("ship it", max_turns=5)
            mgr.wait_on(proc.pid, reason="CI green")
            assert mgr.is_waiting() is True

            # The judge must NOT be called while parked, and no turn is burned.
            judge = MagicMock(return_value=("continue", "x", False, None, False))
            with patch.object(goals, "judge_goal", judge):
                decision = mgr.evaluate_after_turn("still waiting on CI")

            judge.assert_not_called()
            assert decision["verdict"] == "waiting"
            assert decision["should_continue"] is False
            assert decision["continuation_prompt"] is None
            assert mgr.state.turns_used == 0  # no turn consumed while parked
            assert "CI green" in decision["message"]
            assert mgr.state.status == "active"  # still active, just parked
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    def test_wait_on_rejects_a_pid_not_alive_on_this_host(self, hermes_home, monkeypatch):
        """Regression for #110826: do not persist a barrier for remote/dead PIDs."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        monkeypatch.setattr(goals, "_pid_alive", lambda pid: False)
        mgr = GoalManager(session_id="wb-dead")
        mgr.set("ship it")

        with pytest.raises(ValueError, match="not alive on this host"):
            mgr.wait_on(4242, reason="remote CI")

        assert mgr.state.waiting_on_pid is None


    def test_stop_waiting_clears_barrier(self, hermes_home):
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="wb-stop")
            mgr.set("g")
            mgr.wait_on(proc.pid)
            assert mgr.is_waiting() is True
            assert mgr.stop_waiting() is True
            assert mgr.state.waiting_on_pid is None
            assert mgr.is_waiting() is False
            assert mgr.stop_waiting() is False  # idempotent
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    def test_barrier_on_a_process_that_never_exits_expires(self, hermes_home):
        """A poller that outlives the work parked one run for 3h22m; a live barrier ages out."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="wb-expire")
            mgr.set("g")
            mgr.wait_on(proc.pid, reason="poller")
            assert mgr.is_waiting() is True
            mgr.state.waiting_since = time.time() - goals._MAX_BARRIER_WAIT_S - 1
            mgr._save()
            assert mgr.is_waiting() is False
            assert mgr.state.waiting_on_pid is None
        finally:
            proc.terminate()
            proc.wait(timeout=10)


class TestGatherBackgroundProcessesOwnership:
    def test_only_the_owning_sessions_processes_are_seen(self, monkeypatch):
        """The registry task_id collapses to one container key for every agent in the process, so a
        fan-out parent's judge must filter by owner; otherwise a grandchild's poller parks the goal."""
        from hermes_cli import goals

        class _Reg:
            def list_sessions(self, task_id=None, session_key=None):
                return [
                    {"session_id": "proc_mine", "status": "running", "owner_task_id": "root-sid", "task_id": "default"},
                    {"session_id": "proc_child", "status": "running", "owner_task_id": "sa-1-abc", "task_id": "default"},
                    {"session_id": "proc_done", "status": "exited", "owner_task_id": "root-sid", "task_id": "default"},
                ]

        import tools.process_registry as pr
        monkeypatch.setattr(pr, "process_registry", _Reg())
        assert [p["session_id"] for p in goals.gather_background_processes()] == ["proc_mine", "proc_child"]
        assert [p["session_id"] for p in goals.gather_background_processes(owner_task_id="root-sid")] == ["proc_mine"]



# ──────────────────────────────────────────────────────────────────────
# Judge-driven auto-wait — the judge parks the loop on its own
# ──────────────────────────────────────────────────────────────────────


class TestJudgeDrivenWait:
    """The judge returns a `wait` verdict (given live background-process
    context) and the loop parks automatically — no manual /goal wait."""

    @staticmethod
    def _spawn_sleeper():
        import subprocess, sys
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    def test_judge_wait_on_dead_pid_continues_instead_of_parking(self, hermes_home):
        """#110826: a judge ``wait_on_pid`` naming a pid this host cannot observe (remote, or
        already exited) must not park — the barrier would lift and re-park every turn."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="jw-dead-pid", default_max_turns=10)
        mgr.set("ship the PR")
        with patch.object(goals, "_pid_alive", return_value=False), patch.object(
            goals, "judge_goal",
            return_value=("wait", "remote job still running", False, {"pid": 4242}, False),
        ):
            decision = mgr.evaluate_after_turn("Started the job over ssh (pid 4242).")
        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        assert mgr.state.waiting_on_pid is None
        assert mgr.is_waiting() is False

    def test_judge_wait_on_pid_dying_between_check_and_park_continues(self, hermes_home):
        """The pid may exit between the judge path's liveness probe and ``wait_on``'s own
        re-check; that race must land on the same continue decision, not raise out of
        ``evaluate_after_turn`` (callers swallow the error and the continuation is lost)."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="jw-toctou-pid", default_max_turns=10)
        mgr.set("ship the PR")
        with patch.object(goals, "_pid_alive", return_value=True), patch.object(
            GoalManager, "wait_on", side_effect=ValueError("pid is not alive on this host"),
        ), patch.object(
            goals, "judge_goal",
            return_value=("wait", "job still running", False, {"pid": 4242}, False),
        ):
            decision = mgr.evaluate_after_turn("Started the job (pid 4242).")
        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        assert mgr.state.waiting_on_pid is None
        assert mgr.is_waiting() is False

    def test_judge_wait_pid_parks_loop(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="jw-pid", default_max_turns=10)
            mgr.set("ship the PR")
            # Judge sees the running process and says wait-on-pid.
            with patch.object(
                goals, "judge_goal",
                return_value=("wait", "CI watcher still running", False, {"pid": proc.pid}, False),
            ):
                decision = mgr.evaluate_after_turn(
                    "Pushed the PR, watching CI.",
                    background_processes=[{
                        "pid": proc.pid, "command": "wait_for_pr_green.sh",
                        "status": "running", "uptime_seconds": 12,
                    }],
                )
            assert decision["verdict"] == "wait"
            assert decision["should_continue"] is False
            assert decision["continuation_prompt"] is None
            assert mgr.state.waiting_on_pid == proc.pid
            assert mgr.is_waiting() is True

            # Next turn while still parked: judge must NOT be called again.
            judge = MagicMock()
            with patch.object(goals, "judge_goal", judge):
                d2 = mgr.evaluate_after_turn("still going")
            judge.assert_not_called()
            assert d2["verdict"] == "waiting"
            assert d2["should_continue"] is False
        finally:
            proc.terminate()
            proc.wait(timeout=10)


    def test_time_barrier_clears_after_deadline(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="jw-deadline")
        mgr.set("g")
        mgr.wait_for_seconds(120, reason="backoff")
        assert mgr.is_waiting() is True
        # Force the deadline into the past → barrier auto-clears.
        mgr.state.waiting_until = time.time() - 1
        assert mgr.is_waiting() is False
        assert mgr.state.waiting_until == 0.0

    def test_continue_verdict_still_continues_with_background(self, hermes_home):
        """A running process present but judge says continue → normal loop."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="jw-cont", default_max_turns=10)
        mgr.set("do work")
        with patch.object(
            goals, "judge_goal",
            return_value=("continue", "more to do", False, None, False),
        ):
            decision = mgr.evaluate_after_turn(
                "made progress",
                background_processes=[{"pid": 999999, "command": "x", "status": "running"}],
            )
        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        assert mgr.state.waiting_on_pid is None


# ──────────────────────────────────────────────────────────────────────
# Session/trigger barrier — wait on a process's OWN trigger, not just exit
# ──────────────────────────────────────────────────────────────────────


class TestSessionTriggerBarrier:
    """The session barrier (wait_on_session) releases when a process's own
    trigger fires — a watch_patterns match mid-run (process may never exit)
    OR exit — not only on PID exit. CI-safe: uses synthetic registry session
    objects, no real child processes."""

    @staticmethod
    def _inject(sid, *, watch_patterns=None, exited=False):
        import time as _t
        from tools.process_registry import process_registry, ProcessSession
        s = ProcessSession(id=sid, command="watcher.sh", task_id="t",
                           session_key="", cwd="/tmp", started_at=_t.time())
        if watch_patterns:
            s.watch_patterns = list(watch_patterns)
        s.exited = exited
        if exited:
            process_registry._finished[sid] = s
        else:
            process_registry._running[sid] = s
        return s, process_registry


    def test_registry_releases_on_watch_match_while_alive(self, hermes_home):
        s, reg = self._inject("proc_t2", watch_patterns=["READY"])
        assert reg.is_session_waiting("proc_t2") is True
        s._watch_hits = 1  # what _check_watch_patterns sets on a match
        # Released even though the process is STILL running (never exited).
        assert s.exited is False
        assert reg.is_session_waiting("proc_t2") is False


    def test_wait_on_session_validation(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="st-val")
        # No active goal → RuntimeError
        try:
            mgr.wait_on_session("proc_x")
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
        mgr.set("g")
        try:
            mgr.wait_on_session("")
            assert False, "expected ValueError"
        except ValueError:
            pass




# ──────────────────────────────────────────────────────────────────────
# Completion contract (Codex-inspired structured goals)
# ──────────────────────────────────────────────────────────────────────


class TestParseContract:


    def test_inline_fields_parsed(self):
        from hermes_cli.goals import parse_contract

        text = (
            "Migrate auth to JWT\n"
            "verify: the auth test suite passes\n"
            "constraints: keep the /login response shape unchanged\n"
            "boundaries: only touch services/auth and its tests\n"
            "stop when: a schema change needs product sign-off"
        )
        headline, contract = parse_contract(text)
        assert headline == "Migrate auth to JWT"
        assert contract.verification == "the auth test suite passes"
        assert contract.constraints == "keep the /login response shape unchanged"
        assert contract.boundaries == "only touch services/auth and its tests"
        assert contract.stop_when == "a schema change needs product sign-off"
        assert not contract.is_empty()

    def test_alias_variants(self):
        from hermes_cli.goals import parse_contract

        _, c = parse_contract("Goal\nverified by: tests green\npreserve: public API")
        assert c.verification == "tests green"
        assert c.constraints == "public API"


class TestGoalContractSerialization:
    def test_roundtrip_with_contract(self):
        from hermes_cli.goals import GoalState, GoalContract

        state = GoalState(
            goal="ship it",
            contract=GoalContract(
                verification="pytest passes",
                constraints="don't break the API",
            ),
        )
        restored = GoalState.from_json(state.to_json())
        assert restored.goal == "ship it"
        assert restored.contract.verification == "pytest passes"
        assert restored.contract.constraints == "don't break the API"
        assert restored.has_contract()

    def test_old_row_without_contract_loads_clean(self):
        # A state_meta row written before this feature has no "contract" key.
        from hermes_cli.goals import GoalState

        legacy = '{"goal": "old goal", "status": "active", "turns_used": 2}'
        state = GoalState.from_json(legacy)
        assert state.goal == "old goal"
        assert state.turns_used == 2
        assert state.contract.is_empty()
        assert not state.has_contract()

    def test_render_block_omits_empty_fields(self):
        from hermes_cli.goals import GoalContract

        block = GoalContract(outcome="X", verification="Y").render_block()
        assert "Outcome: X" in block
        assert "Verification: Y" in block
        assert "Constraints" not in block


class TestGoalStateNoPark:
    """GoalState.no_park — per-goal WAIT-park override serialization."""

    def test_no_park_field_defaults_to_false(self):
        from hermes_cli.goals import GoalState

        state = GoalState(goal="ship it")
        assert state.no_park is False

    def test_asdict_serialization_includes_no_park(self):
        from dataclasses import asdict
        from hermes_cli.goals import GoalState

        state = GoalState(goal="ship it", no_park=True)
        data = asdict(state)
        assert data.get("no_park") is True
        assert json.loads(state.to_json()).get("no_park") is True

    def test_roundtrip_preserves_no_park_true(self):
        from hermes_cli.goals import GoalState

        state = GoalState(goal="ship it", no_park=True)
        restored = GoalState.from_json(state.to_json())
        assert restored.goal == "ship it"
        assert restored.no_park is True

    def test_roundtrip_preserves_no_park_false(self):
        from hermes_cli.goals import GoalState

        state = GoalState(goal="ship it", no_park=False)
        restored = GoalState.from_json(state.to_json())
        assert restored.no_park is False

    def test_old_row_without_no_park_loads_false(self):
        # A state_meta row written before this feature has no "no_park" key
        # and must load with no_park=False (backward compatible).
        from hermes_cli.goals import GoalState

        legacy = json.dumps({
            "goal": "old goal",
            "status": "active",
            "turns_used": 2,
            "max_turns": 20,
            "created_at": 1.0,
            "last_turn_at": 2.0,
        })
        state = GoalState.from_json(legacy)
        assert state.goal == "old goal"
        assert state.no_park is False

    def test_non_boolean_values_are_coerced(self):
        from hermes_cli.goals import GoalState

        # Truthy values coerce to True, falsy to False.
        assert GoalState.from_json('{"goal": "x", "no_park": 1}').no_park is True
        assert GoalState.from_json('{"goal": "x", "no_park": true}').no_park is True
        assert GoalState.from_json('{"goal": "x", "no_park": 0}').no_park is False


class TestGoalStateDontAskUserInput:
    """GoalState.dont_ask_user_input — per-goal unattended-run override serialization."""

    def test_dont_ask_user_input_field_defaults_to_false(self):
        from hermes_cli.goals import GoalState

        state = GoalState(goal="ship it")
        assert state.dont_ask_user_input is False

    def test_asdict_serialization_includes_dont_ask_user_input(self):
        from dataclasses import asdict
        from hermes_cli.goals import GoalState

        state = GoalState(goal="ship it", dont_ask_user_input=True)
        data = asdict(state)
        assert data.get("dont_ask_user_input") is True
        assert json.loads(state.to_json()).get("dont_ask_user_input") is True

    def test_roundtrip_preserves_dont_ask_user_input_true(self):
        from hermes_cli.goals import GoalState

        state = GoalState(goal="ship it", dont_ask_user_input=True)
        restored = GoalState.from_json(state.to_json())
        assert restored.goal == "ship it"
        assert restored.dont_ask_user_input is True

    def test_roundtrip_preserves_dont_ask_user_input_false(self):
        from hermes_cli.goals import GoalState

        state = GoalState(goal="ship it", dont_ask_user_input=False)
        restored = GoalState.from_json(state.to_json())
        assert restored.dont_ask_user_input is False

    def test_old_row_without_dont_ask_user_input_loads_false(self):
        # A state_meta row written before this feature has no
        # "dont_ask_user_input" key and must load with False (backward
        # compatible).
        from hermes_cli.goals import GoalState

        legacy = json.dumps({
            "goal": "old goal",
            "status": "active",
            "turns_used": 2,
            "max_turns": 20,
            "created_at": 1.0,
            "last_turn_at": 2.0,
        })
        state = GoalState.from_json(legacy)
        assert state.goal == "old goal"
        assert state.dont_ask_user_input is False

    def test_non_boolean_values_are_coerced(self):
        from hermes_cli.goals import GoalState

        # Truthy values coerce to True, falsy to False.
        assert GoalState.from_json('{"goal": "x", "dont_ask_user_input": 1}').dont_ask_user_input is True
        assert GoalState.from_json('{"goal": "x", "dont_ask_user_input": true}').dont_ask_user_input is True
        assert GoalState.from_json('{"goal": "x", "dont_ask_user_input": 0}').dont_ask_user_input is False


class TestJudgeSystemPromptBuilder:
    """_build_judge_system_prompt — the no-park judge prompt omits WAIT.

    These assert the relationship contracts between the prompt constants and
    the builder, not snapshot values: the backward-compatible constant is the
    concatenation of BASE + WAIT, the default build equals that constant, and
    the no-park build drops every WAIT mention while keeping DONE and
    CONTINUE fully intact.
    """

    WAIT_MARKERS = (
        "Picking WAIT parks the loop",
        "wait_on_session",
        "wait_on_pid",
        "wait_for_seconds",
    )

    def test_judge_system_prompt_is_base_plus_wait(self):
        from hermes_cli.goals import (
            JUDGE_SYSTEM_PROMPT,
            JUDGE_SYSTEM_PROMPT_BASE,
            JUDGE_SYSTEM_PROMPT_WAIT,
        )

        # Backward-compat invariant: the default constant is the BASE block
        # plus the WAIT paragraph, and it still instructs the judge on WAIT.
        assert JUDGE_SYSTEM_PROMPT == JUDGE_SYSTEM_PROMPT_BASE + JUDGE_SYSTEM_PROMPT_WAIT
        assert "Picking WAIT parks the loop" in JUDGE_SYSTEM_PROMPT
        assert JUDGE_SYSTEM_PROMPT_WAIT.startswith("WAIT — the goal is NOT done")
        assert JUDGE_SYSTEM_PROMPT_WAIT.endswith("until the async thing finishes.")

    def test_default_build_equals_judge_system_prompt(self):
        from hermes_cli.goals import JUDGE_SYSTEM_PROMPT, _build_judge_system_prompt

        assert _build_judge_system_prompt(False) == JUDGE_SYSTEM_PROMPT

    def test_no_park_build_drops_all_wait_mentions(self):
        from hermes_cli.goals import _build_judge_system_prompt

        prompt = _build_judge_system_prompt(True)
        for marker in self.WAIT_MARKERS:
            assert marker not in prompt, f"no-park prompt must not mention {marker!r}"
        # The no-park preamble drops to three verdicts (DONE/BLOCKED/CONTINUE)
        # — the BLOCKED verdict survives the no-park build.
        assert "Decide one of three verdicts" in prompt
        assert "Decide one of four verdicts" not in prompt

    def test_no_park_build_keeps_done_and_continue_intact(self):
        from hermes_cli.goals import (
            JUDGE_SYSTEM_PROMPT_CONTINUE,
            JUDGE_SYSTEM_PROMPT_DONE,
            _build_judge_system_prompt,
        )

        prompt = _build_judge_system_prompt(True)
        # DONE and CONTINUE are described fully and identically in both
        # builds — the no-park variant must not degrade either verdict.
        assert JUDGE_SYSTEM_PROMPT_DONE in prompt
        assert JUDGE_SYSTEM_PROMPT_CONTINUE in prompt
        assert prompt.count("DONE — the goal is fully satisfied") == 1
        assert prompt.count("CONTINUE — not done") == 1

    def test_full_build_still_teaches_wait_shapes(self):
        from hermes_cli.goals import _build_judge_system_prompt

        prompt = _build_judge_system_prompt(False)
        for marker in self.WAIT_MARKERS:
            assert marker in prompt
        # The full build offers all four verdicts (DONE/BLOCKED/WAIT/CONTINUE).
        assert "Decide one of four verdicts" in prompt

    def test_legacy_done_shape_present_in_both_builds(self):
        from hermes_cli.goals import _build_judge_system_prompt

        for prompt in (_build_judge_system_prompt(False), _build_judge_system_prompt(True)):
            assert "The legacy shape" in prompt
            assert "true=done, false=continue" in prompt


class TestJudgeSystemPromptDontAskUserInput:
    """_build_judge_system_prompt(..., dont_ask_user_input=True) drops the
    blocked / user-input hand-off clause from the DONE paragraph (US-004).

    The invariants here are: with the flag False (for either no_park value)
    every build stays byte-identical to today; with the flag True neither
    build contains the "needs user input → DONE" hand-off wording, while the
    genuine-completion DONE bullets remain intact.
    """

    def _default_build(self):
        from hermes_cli.goals import JUDGE_SYSTEM_PROMPT, _build_judge_system_prompt

        return _build_judge_system_prompt(), JUDGE_SYSTEM_PROMPT

    def test_dont_ask_build_omits_blocked_handoff_clause(self):
        from hermes_cli.goals import _build_judge_system_prompt

        for no_park in (False, True):
            prompt = _build_judge_system_prompt(no_park, dont_ask_user_input=True)
            # The hand-off clause must be gone entirely.
            assert "needs user input" not in prompt
            assert "treat this as DONE" not in prompt
            assert "The response explains the goal is unachievable / blocked" not in prompt
            # DONE is still described for genuine completion.
            assert "DONE — the goal is fully satisfied" in prompt
            assert "The response explicitly confirms the goal was completed" in prompt
            assert "The response clearly shows the final deliverable was produced" in prompt
            # And it is explicit that a mere block is NOT a DONE reason.
            assert "NOT DONE" in prompt

    def test_dont_ask_build_keeps_continue_section(self):
        from hermes_cli.goals import JUDGE_SYSTEM_PROMPT_CONTINUE, _build_judge_system_prompt

        for no_park in (False, True):
            prompt = _build_judge_system_prompt(no_park, dont_ask_user_input=True)
            assert JUDGE_SYSTEM_PROMPT_CONTINUE in prompt
            assert prompt.count("CONTINUE — not done") == 1

    def test_dont_ask_with_no_park_omits_wait_section(self):
        from hermes_cli.goals import _build_judge_system_prompt

        prompt = _build_judge_system_prompt(True, dont_ask_user_input=True)
        for marker in ("Picking WAIT parks the loop", "wait_on_session", "wait_on_pid", "wait_for_seconds"):
            assert marker not in prompt, f"no-park no-ask prompt must not mention {marker!r}"
        assert "Decide one of two verdicts" in prompt

    def test_dont_ask_with_wait_keeps_wait_section(self):
        from hermes_cli.goals import _build_judge_system_prompt

        prompt = _build_judge_system_prompt(False, dont_ask_user_input=True)
        assert "Picking WAIT parks the loop" in prompt
        assert "Decide one of three verdicts" in prompt

    def test_default_and_no_park_builds_byte_identical_when_flag_false(self):
        # Extending the signature must not change any existing output.
        from hermes_cli.goals import (
            JUDGE_SYSTEM_PROMPT,
            _JUDGE_SYSTEM_PROMPT_PREAMBLE_NO_WAIT,
            _JUDGE_SYSTEM_PROMPT_REPLY_NO_WAIT,
            _build_judge_system_prompt,
            JUDGE_SYSTEM_PROMPT_DONE,
            JUDGE_SYSTEM_PROMPT_BLOCKED,
            JUDGE_SYSTEM_PROMPT_CONTINUE,
        )

        assert _build_judge_system_prompt(False, dont_ask_user_input=False) == JUDGE_SYSTEM_PROMPT
        legacy_no_park = (
            _JUDGE_SYSTEM_PROMPT_PREAMBLE_NO_WAIT
            + JUDGE_SYSTEM_PROMPT_DONE
            + JUDGE_SYSTEM_PROMPT_BLOCKED
            + JUDGE_SYSTEM_PROMPT_CONTINUE
            + _JUDGE_SYSTEM_PROMPT_REPLY_NO_WAIT
        )
        assert _build_judge_system_prompt(True, dont_ask_user_input=False) == legacy_no_park

    def test_judge_system_prompt_done_constant_unchanged(self):
        from hermes_cli.goals import JUDGE_SYSTEM_PROMPT_DONE

        assert JUDGE_SYSTEM_PROMPT_DONE == (
            "DONE — the goal is fully satisfied:\n"
            "- The response explicitly confirms the goal was completed, OR\n"
            "- The response clearly shows the final deliverable was produced.\n"
            "DONE requires the deliverable to actually exist. If the response only "
            "explains why the goal cannot be reached, the verdict is BLOCKED, not "
            "DONE.\n\n"
        )


class TestBuildJudgeUserPrompt:
    """_build_judge_user_prompt — the judge user-prompt builder (US-004)."""

    GOAL = "ship the feature"
    RESPONSE = "I made progress on the module."
    CONTRACT_BLOCK = "Verification: pytest -q passes.\nConstraints: no API change."
    SUBGOALS_BLOCK = (
        "- Extra criterion 1: keep the public API stable.\n"
        "- Extra criterion 2: update the changelog."
    )

    def _plain(self, **kwargs):
        from hermes_cli.goals import _build_judge_user_prompt

        return _build_judge_user_prompt(
            self.GOAL, self.RESPONSE, "", "2026-09-01 00:00:00 UTC", **kwargs
        )

    def test_default_contract_matches_template_exactly(self):
        from hermes_cli.goals import (
            JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE,
            _build_judge_user_prompt,
        )

        built = self._plain(contract_block=self.CONTRACT_BLOCK)
        expected = JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE.format(
            goal=self.GOAL,
            contract_block=self.CONTRACT_BLOCK,
            response=self.RESPONSE,
            background_block="",
            current_time="2026-09-01 00:00:00 UTC",
        )
        assert built == expected

    def test_default_subgoals_matches_template_exactly(self):
        from hermes_cli.goals import (
            JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE,
            _build_judge_user_prompt,
        )

        built = self._plain(subgoals_block=self.SUBGOALS_BLOCK)
        expected = JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
            goal=self.GOAL,
            subgoals_block=self.SUBGOALS_BLOCK,
            response=self.RESPONSE,
            background_block="",
            current_time="2026-09-01 00:00:00 UTC",
        )
        assert built == expected

    def test_default_plain_matches_template_exactly(self):
        from hermes_cli.goals import (
            JUDGE_USER_PROMPT_TEMPLATE,
            _build_judge_user_prompt,
        )

        built = self._plain()
        expected = JUDGE_USER_PROMPT_TEMPLATE.format(
            goal=self.GOAL,
            response=self.RESPONSE,
            background_block="",
            current_time="2026-09-01 00:00:00 UTC",
        )
        assert built == expected

    def test_contract_priority_over_subgoals(self):
        from hermes_cli.goals import _build_judge_user_prompt

        both = self._plain(contract_block=self.CONTRACT_BLOCK, subgoals_block=self.SUBGOALS_BLOCK)
        only_contract = self._plain(contract_block=self.CONTRACT_BLOCK)
        # The contract block wins; the subgoals block is not injected raw.
        assert both == only_contract
        assert self.CONTRACT_BLOCK in both

    def test_dont_ask_contract_flips_blocked_rule_to_continue(self):
        from hermes_cli.goals import _build_judge_user_prompt

        prompt = self._plain(
            contract_block=self.CONTRACT_BLOCK, dont_ask_user_input=True
        )
        # The old "blocked / unachievable / needs user input → DONE" rule is gone.
        assert "needs user input" not in prompt
        assert "treat it as DONE" not in prompt
        assert "treat this as DONE" not in prompt
        # The flipped rule tells the judge to keep going autonomously.
        assert "treat it as CONTINUE" in prompt
        assert "must keep working" in prompt
        # Genuine-completion and constraint rules are preserved.
        assert "The goal is DONE only when the Verification criterion is satisfied" in prompt
        assert "If any stated Constraint was violated, the goal is NOT done" in prompt
        assert "Otherwise the goal is NOT done — CONTINUE" in prompt
        assert "done, continue, or wait?" in prompt

    def test_dont_ask_subgoals_and_plain_unchanged(self):
        from hermes_cli.goals import _build_judge_user_prompt

        for extra in ({}, {"subgoals_block": self.SUBGOALS_BLOCK}):
            default = self._plain(**extra)
            no_ask = self._plain(dont_ask_user_input=True, **extra)
            # The plain/subgoals templates carry no hand-off wording, so the
            # flag must leave them byte-identical.
            assert no_ask == default
            assert "needs user input" not in default

    def test_dont_ask_contract_with_no_park_keeps_both_removals(self):
        from hermes_cli.goals import (
            JUDGE_SYSTEM_PROMPT_WAIT,
            _build_judge_system_prompt,
            _build_judge_user_prompt,
        )

        user = self._plain(contract_block=self.CONTRACT_BLOCK, dont_ask_user_input=True)
        system = _build_judge_system_prompt(True, dont_ask_user_input=True)
        assert "needs user input" not in user
        assert "needs user input" not in system
        assert JUDGE_SYSTEM_PROMPT_WAIT not in system
        assert "Picking WAIT parks the loop" not in system


class TestGoalManagerContract:



    def test_set_contract_after_the_fact(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        mgr = GoalManager(session_id="c-after")
        mgr.set("ship it")
        assert not mgr.has_contract()
        mgr.set_contract(GoalContract(verification="x"))
        assert mgr.has_contract()
        # Survives reload.
        from hermes_cli.goals import GoalManager as GM2
        assert GM2(session_id="c-after").has_contract()

    def test_persistence_roundtrip(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        GoalManager(session_id="c-persist").set(
            "ship it", contract=GoalContract(outcome="O", verification="V")
        )
        reloaded = GoalManager(session_id="c-persist")
        assert reloaded.state.contract.outcome == "O"
        assert reloaded.state.contract.verification == "V"


class TestJudgeWithContract:
    def _fake_call_llm(self, captured, content='{"done": false, "reason": "more"}'):
        """judge_goal routes through call_llm (#35566) — capture its kwargs."""
        class _FakeMsg:
            pass
        _FakeMsg.content = content
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]

        def _fake(**kwargs):
            captured.update(kwargs)
            return _FakeResp()
        return _fake

    def test_judge_uses_contract_template(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals
        from hermes_cli.goals import GoalContract

        captured = {}
        with patch("agent.auxiliary_client.call_llm",
                   side_effect=self._fake_call_llm(captured)):
            goals.judge_goal(
                "ship it", "I think it's done",
                contract=GoalContract(verification="pytest -q passes"),
            )
        user_msg = next(
            (m["content"] for m in (captured.get("messages") or []) if m["role"] == "user"), ""
        )
        assert "completion contract" in user_msg.lower()
        assert "pytest -q passes" in user_msg
        assert "concrete evidence" in user_msg


class TestDraftContract:
    def test_draft_parses_json(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals

        class _FakeMsg:
            content = (
                '{"outcome": "auth on JWT", "verification": "auth suite green", '
                '"constraints": "no API change", "boundaries": "services/auth", '
                '"stop_when": "schema change needed"}'
            )
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]
        with patch("agent.auxiliary_client.call_llm",
                   return_value=_FakeResp()):
            contract = goals.draft_contract("Migrate auth to JWT")
        assert contract is not None
        assert contract.outcome == "auth on JWT"
        assert contract.verification == "auth suite green"
        assert not contract.is_empty()


    def test_draft_returns_none_when_no_client(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals

        with patch("agent.auxiliary_client.call_llm",
                   side_effect=RuntimeError("No LLM provider configured")):
            assert goals.draft_contract("anything") is None


# ──────────────────────────────────────────────────────────────────────
# Compose: completion contract + wait barrier in one judge call
# ──────────────────────────────────────────────────────────────────────


class TestContractAndBackgroundCompose:
    """A contract goal blocked on a background process must surface BOTH
    the contract block and the background-process list to the judge, so it
    can return either done (evidence met) or wait (parked on the poller)."""

    def _capture_call_llm(self, captured, content='{"verdict": "wait", "wait_on_pid": 4242, "reason": "CI still running"}'):
        """judge_goal routes through call_llm (#35566) — capture its kwargs."""
        class _FakeMsg:
            pass
        _FakeMsg.content = content
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]

        def _fake(**kwargs):
            captured.update(kwargs)
            return _FakeResp()
        return _fake

    def test_judge_prompt_carries_contract_and_background(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals
        from hermes_cli.goals import GoalContract

        captured = {}
        bg = [{
            "session_id": "ci-watch", "pid": 4242, "status": "running",
            "command": "wait_for_pr_green.sh 50501", "trigger": "exit",
        }]
        with patch("agent.auxiliary_client.call_llm",
                   side_effect=self._capture_call_llm(captured)):
            verdict, reason, parse_failed, wait_directive, _tf = goals.judge_goal(
                "ship the PR",
                "I pushed and started the CI watcher; waiting on it now.",
                contract=GoalContract(verification="PR CI goes green"),
                background_processes=bg,
            )
        user_msg = next(
            (m["content"] for m in (captured.get("messages") or []) if m["role"] == "user"), ""
        )
        # Both surfaces present in one prompt.
        assert "completion contract" in user_msg.lower()
        assert "PR CI goes green" in user_msg
        assert "Background processes" in user_msg
        assert "4242" in user_msg
        # The judge can return a wait verdict on a contract goal.
        assert verdict == "wait"
        assert wait_directive and wait_directive.get("pid") == 4242


class TestBlockedVerdict:
    """#100954: a genuinely unachievable goal must be refused, not completed."""

    def test_parse_judge_response_accepts_blocked(self):
        from hermes_cli.goals import _parse_judge_response

        verdict, reason, parse_failed, _wd = _parse_judge_response(
            '{"verdict": "blocked", "reason": "the repo was deleted"}'
        )
        assert verdict == "blocked"
        assert reason == "the repo was deleted"
        assert parse_failed is False

    def test_blocked_verdict_pauses_goal_instead_of_done(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="blocked-sid")
        mgr.set("delete a repository that does not exist")
        with patch(
            "hermes_cli.goals.judge_goal",
            return_value=("blocked", "the repo does not exist", False, None, False),
        ):
            decision = mgr.evaluate_after_turn(
                "The repo cannot be deleted: it does not exist."
            )

        assert decision["verdict"] == "blocked"
        assert decision["status"] == "paused"
        assert decision["should_continue"] is False
        assert "unachievable" in decision["message"].lower()
        assert mgr.state is not None
        assert mgr.state.status == "paused"
        assert "unachievable" in (mgr.state.paused_reason or "").lower()


def test_goal_session_db_is_the_registry_shared_handle(hermes_home):
    """GoalManager must borrow the process-wide registry handle for ``state.db`` rather than
    minting a bare ``SessionDB()``: a second writer per profile carries its own token-writer
    thread and close-time checkpoint beside the gateway's handle (the #90837 corruption shape)."""
    from hermes_cli import goals
    import hermes_state_registry as registry

    db = goals._get_session_db()
    assert db is not None
    try:
        assert any(shared is db for shared in registry.live_shared_session_dbs())
        assert goals._get_session_db() is db
    finally:
        goals._DB_CACHE.clear()
        registry.release_or_close(db)
# ──────────────────────────────────────────────────────────────────────
# parse_no_park_prefix — SRFI-88-style /goal no-park: prefix
# ──────────────────────────────────────────────────────────────────────


class TestParseNoParkPrefix:
    """Unit tests for the pure no-park: prefix parser (US-005)."""

    def _parse(self, arg):
        from hermes_cli.goals import parse_no_park_prefix

        return parse_no_park_prefix(arg)

    def test_t_prefix_sets_true_and_strips_token(self):
        assert self._parse("no-park: #t Implement login") == (True, "Implement login")

    def test_f_prefix_sets_false_and_strips_token(self):
        assert self._parse("no-park: #f Implement login") == (False, "Implement login")

    def test_f_prefix_strips_token_keeping_rest(self):
        # The remainder after the value must survive intact.
        assert self._parse("no-park: #f X Y Z") == (False, "X Y Z")

    def test_no_prefix_returns_unchanged_false(self):
        assert self._parse("Implement login") == (False, "Implement login")

    def test_empty_arg_returns_false_unchanged(self):
        assert self._parse("") == (False, "")

    def test_prefix_identifier_is_case_insensitive(self):
        assert self._parse("No-Park: #t Implement login") == (True, "Implement login")

    def test_whitespace_around_token_is_tolerated(self):
        assert self._parse("  no-park:   #t   Implement login") == (True, "Implement login")

    def test_invalid_value_raises_exact_error(self):
        with pytest.raises(ValueError) as exc:
            self._parse("no-park: yes Implement login")
        assert str(exc.value) == (
            "Invalid value for no-park: must be #t or #f, got 'yes'"
        )

    def test_numeric_value_raises_exact_error(self):
        with pytest.raises(ValueError) as exc:
            self._parse("no-park: 42 Implement login")
        assert str(exc.value) == (
            "Invalid value for no-park: must be #t or #f, got '42'"
        )

    def test_bare_boolean_word_raises_exact_error(self):
        with pytest.raises(ValueError) as exc:
            self._parse("no-park: true Implement login")
        assert str(exc.value) == (
            "Invalid value for no-park: must be #t or #f, got 'true'"
        )

    def test_missing_value_raises_exact_error(self):
        with pytest.raises(ValueError) as exc:
            self._parse("no-park:")
        assert str(exc.value) == (
            "Invalid value for no-park: must be #t or #f, got ''"
        )

    def test_goal_that_merely_contains_no_park_colon_later_is_untouched(self):
        # The token only counts at the very start; a later `no-park:` in the
        # goal prose is ordinary text.
        assert self._parse("Implement login no-park: #t please") == (
            False,
            "Implement login no-park: #t please",
        )


# ──────────────────────────────────────────────────────────────────────
# _parse_srfi88_value — shared SRFI-88 boolean validator for per-goal flags
# ──────────────────────────────────────────────────────────────────────


class TestParseSrfI88Value:
    """Unit tests for the shared SRFI-88 boolean validator (US-002)."""

    def _parse(self, key, value):
        from hermes_cli.goals import _parse_srfi88_value

        return _parse_srfi88_value(key, value)

    def test_t_parses_to_true(self):
        assert self._parse("dont-ask-user-input", "#t") is True

    def test_f_parses_to_false(self):
        assert self._parse("dont-ask-user-input", "#f") is False

    def test_any_other_value_raises_exact_error(self):
        with pytest.raises(ValueError) as exc:
            self._parse("dont-ask-user-input", "maybe")
        assert str(exc.value) == (
            "Invalid value for dont-ask-user-input: must be #t or #f, got 'maybe'"
        )

    def test_error_names_the_requested_key(self):
        # The error message carries the flag key so the caller can surface it
        # verbatim for any per-goal flag.
        with pytest.raises(ValueError) as exc:
            self._parse("no-park", "yes")
        assert str(exc.value) == (
            "Invalid value for no-park: must be #t or #f, got 'yes'"
        )

    def test_empty_value_raises_exact_error(self):
        with pytest.raises(ValueError) as exc:
            self._parse("dont-ask-user-input", "")
        assert str(exc.value) == (
            "Invalid value for dont-ask-user-input: must be #t or #f, got ''"
        )


# ──────────────────────────────────────────────────────────────────────
# parse_dont_ask_user_input_prefix — SRFI-88-style /goal dont-ask-user-input: prefix
# ──────────────────────────────────────────────────────────────────────


class TestParseDontAskUserInputPrefix:
    """Unit tests for the pure dont-ask-user-input: prefix parser (US-002)."""

    def _parse(self, arg):
        from hermes_cli.goals import parse_dont_ask_user_input_prefix

        return parse_dont_ask_user_input_prefix(arg)

    def test_t_prefix_sets_true_and_strips_token(self):
        assert self._parse("dont-ask-user-input: #t run migration") == (
            True,
            "run migration",
        )

    def test_f_prefix_sets_false_and_strips_token(self):
        assert self._parse("dont-ask-user-input: #f run migration") == (
            False,
            "run migration",
        )

    def test_f_prefix_strips_token_keeping_rest(self):
        assert self._parse("dont-ask-user-input: #f X Y Z") == (False, "X Y Z")

    def test_no_prefix_returns_unchanged_false(self):
        assert self._parse("run migration") == (False, "run migration")

    def test_empty_arg_returns_false_unchanged(self):
        assert self._parse("") == (False, "")

    def test_prefix_identifier_is_case_insensitive(self):
        assert self._parse("Dont-Ask-User-Input: #t run migration") == (
            True,
            "run migration",
        )

    def test_pascal_case_identifier_mixed_case_accepted(self):
        assert self._parse("DoNt-AsK-uSER-iNPUT: #t run migration") == (
            True,
            "run migration",
        )

    def test_whitespace_around_token_is_tolerated(self):
        assert self._parse("  dont-ask-user-input:   #t   run migration") == (
            True,
            "run migration",
        )

    def test_invalid_value_raises_exact_error(self):
        with pytest.raises(ValueError) as exc:
            self._parse("dont-ask-user-input: maybe run migration")
        assert str(exc.value) == (
            "Invalid value for dont-ask-user-input: must be #t or #f, got 'maybe'"
        )

    def test_numeric_value_raises_exact_error(self):
        with pytest.raises(ValueError) as exc:
            self._parse("dont-ask-user-input: 42 run migration")
        assert str(exc.value) == (
            "Invalid value for dont-ask-user-input: must be #t or #f, got '42'"
        )

    def test_bare_boolean_word_raises_exact_error(self):
        with pytest.raises(ValueError) as exc:
            self._parse("dont-ask-user-input: true run migration")
        assert str(exc.value) == (
            "Invalid value for dont-ask-user-input: must be #t or #f, got 'true'"
        )

    def test_missing_value_raises_exact_error(self):
        with pytest.raises(ValueError) as exc:
            self._parse("dont-ask-user-input:")
        assert str(exc.value) == (
            "Invalid value for dont-ask-user-input: must be #t or #f, got ''"
        )

    def test_goal_that_merely_contains_token_later_is_untouched(self):
        # The token only counts at the very start; a later
        # `dont-ask-user-input:` in the goal prose is ordinary text.
        assert self._parse("run migration dont-ask-user-input: #t please") == (
            False,
            "run migration dont-ask-user-input: #t please",
        )

    def test_flag_composes_with_no_park_then_goal_text(self):
        # /goal dont-ask-user-input: #t <text> — the flag token is stripped and
        # the remaining text (which may itself carry a no-park token that a later
        # story composes) is preserved verbatim for composition.
        assert self._parse("dont-ask-user-input: #t no-park: #t run migration") == (
            True,
            "no-park: #t run migration",
        )


# ──────────────────────────────────────────────────────────────────────
# parse_goal_flags — combined SRFI-88 flag parsing (no-park + dont-ask)
# ──────────────────────────────────────────────────────────────────────


class TestParseGoalFlags:
    """Unit tests for the combined per-goal flag parser (US-007)."""

    def _parse(self, arg):
        from hermes_cli.goals import parse_goal_flags

        return parse_goal_flags(arg)

    def test_no_flags_returns_false_false_unchanged(self):
        assert self._parse("Implement login") == (False, False, "Implement login")

    def test_empty_arg(self):
        assert self._parse("") == (False, False, "")

    def test_dont_ask_only(self):
        assert self._parse("dont-ask-user-input: #t run migration") == (
            False,
            True,
            "run migration",
        )

    def test_dont_ask_false_explicit(self):
        assert self._parse("dont-ask-user-input: #f run migration") == (
            False,
            False,
            "run migration",
        )

    def test_no_park_only(self):
        assert self._parse("no-park: #t run migration") == (
            True,
            False,
            "run migration",
        )

    def test_both_flags_no_park_first(self):
        assert self._parse("no-park: #t dont-ask-user-input: #t ship it") == (
            True,
            True,
            "ship it",
        )

    def test_both_flags_dont_ask_first(self):
        assert self._parse("dont-ask-user-input: #t no-park: #f ship it") == (
            False,
            True,
            "ship it",
        )

    def test_flags_case_insensitive_identifiers(self):
        # Identifier matching is case-insensitive (values stay exact #t/#f).
        assert self._parse("NO-PARK: #t Dont-Ask-User-Input: #f ship it") == (
            True,
            False,
            "ship it",
        )

    def test_invalid_dont_ask_value_raises_exact_error(self):
        with pytest.raises(ValueError) as exc:
            self._parse("dont-ask-user-input: maybe ship it")
        assert str(exc.value) == (
            "Invalid value for dont-ask-user-input: must be #t or #f, got 'maybe'"
        )

    def test_invalid_no_park_value_raises_exact_error(self):
        with pytest.raises(ValueError) as exc:
            self._parse("no-park: yes ship it")
        assert str(exc.value) == (
            "Invalid value for no-park: must be #t or #f, got 'yes'"
        )

    def test_errors_surface_left_to_right(self):
        # The FIRST invalid token (leftmost) is the one reported.
        with pytest.raises(ValueError) as exc:
            self._parse("no-park: bad1 dont-ask-user-input: bad2 ship it")
        assert "got 'bad1'" in str(exc.value)
        with pytest.raises(ValueError) as exc:
            self._parse("dont-ask-user-input: bad2 no-park: bad1 ship it")
        assert "got 'bad2'" in str(exc.value)

    def test_repeated_flag_last_wins(self):
        assert self._parse("no-park: #t no-park: #f ship it") == (
            False,
            False,
            "ship it",
        )

    def test_token_later_in_prose_is_not_a_flag(self):
        # Only leading tokens count; the same token in the goal prose is text.
        assert self._parse("ship it no-park: #t please") == (
            False,
            False,
            "ship it no-park: #t please",
        )

    def test_delegates_validation_to_per_key_parsers(self):
        # parse_goal_flags must not reimplement value validation: the error
        # text comes from the shared _parse_srfi88_value via the per-key
        # parsers (already asserted above); here we pin the no-token default
        # path preserves whitespace-heavy input verbatim.
        arg = "   spaced   goal text"
        assert self._parse(arg) == (False, False, arg)


# ──────────────────────────────────────────────────────────────────────
# /goal no-park: wiring through _handle_goal_command (mocked goal manager)
# ──────────────────────────────────────────────────────────────────────


class _FakeGoalState:
    """Minimal stand-in for GoalState consumed by _handle_goal_command."""

    def __init__(self, goal_text: str):
        self.goal = goal_text
        self.max_turns = 20
        self.contract = None

    def has_contract(self):
        return False


class _GoalCommandHarness:
    """Wraps a real HermesCLI with a mocked goal manager for /goal tests."""

    def __init__(self):
        from unittest.mock import MagicMock

        import cli

        self.cli = cli
        self.shell = cli.HermesCLI(compact=True, max_turns=1)
        self.mgr = MagicMock()
        self.mgr.session_id = "test-session"
        self.shell.session_id = "test-session"
        self.shell._goal_manager = self.mgr

    def run(self, cmd_text: str, goal_text: str = "fake goal"):
        """Execute a /goal command capturing _cprint output instead of rendering."""
        with patch.object(self.cli, "_cprint") as cprint:
            self.mgr.set.return_value = _FakeGoalState(goal_text)
            self.shell._handle_goal_command(cmd_text)
        return cprint

    def assert_set_called(self, expected_text, expected_no_park):
        assert self.mgr.set.called, "mgr.set was not called"
        call = self.mgr.set.call_args
        assert call.args[0] == expected_text
        assert call.kwargs.get("no_park") is expected_no_park


class TestHandleGoalCommandNoPark:
    """_handle_goal_command honors the no-park: prefix (US-005)."""

    def test_no_park_true_goes_to_set(self):
        harness = _GoalCommandHarness()
        harness.run("/goal no-park: #t Implement login", goal_text="Implement login")
        harness.assert_set_called("Implement login", True)

    def test_no_park_false_goes_to_set(self):
        harness = _GoalCommandHarness()
        harness.run("/goal no-park: #f Implement login", goal_text="Implement login")
        harness.assert_set_called("Implement login", False)

    def test_no_prefix_keeps_backward_compat(self):
        harness = _GoalCommandHarness()
        harness.run("/goal Implement login", goal_text="Implement login")
        harness.assert_set_called("Implement login", False)

    def test_headline_never_polluted_with_token(self):
        # With verbose goal text, the no-park token must not leak into the
        # text handed to parse_contract/set.
        harness = _GoalCommandHarness()
        harness.run("/goal no-park: #t Migrate auth to JWT", goal_text="Migrate auth to JWT")
        harness.assert_set_called("Migrate auth to JWT", True)

    def test_invalid_value_prints_exact_error_and_sets_no_goal(self):
        harness = _GoalCommandHarness()
        cprint = harness.run("/goal no-park: yes Implement login")
        assert harness.mgr.set.called is False
        rendered = "\n".join(str(a.args[0]) for a in cprint.call_args_list)
        assert (
            "Invalid value for no-park: must be #t or #f, got 'yes'" in rendered
        )

    def test_task4018_numeric_value_sets_no_goal(self):
        harness = _GoalCommandHarness()
        cprint = harness.run("/goal no-park: 42 Implement login")
        assert harness.mgr.set.called is False
        rendered = "\n".join(str(a.args[0]) for a in cprint.call_args_list)
        assert "got '42'" in rendered

    def test_draft_path_strips_leading_override_token(self):
        # `no-park:` placed before the draft subcommand is stripped and the
        # draft objective stays clean. (Drafting is dispatched through
        # hermes_cli.goal_command._set in the current architecture, so the
        # assertion is on mgr.set + the stubbed draft_contract.)
        harness = _GoalCommandHarness()
        with patch("hermes_cli.goals.draft_contract") as draft:
            harness.run("/goal no-park: #t draft build login", goal_text="draft build login")
        assert draft.called
        assert draft.call_args.args[0] == "build login"
        call = harness.mgr.set.call_args
        assert call.args[0] == "build login"
        assert call.kwargs.get("no_park") is True


class TestHandleGoalDraftNoPark:
    """/goal draft honors the no-park: override (US-006)."""

    def _run_draft(self, cmd_text: str, goal_text: str = "fake goal"):
        """Run a /goal draft command with draft_contract stubbed out."""
        harness = _GoalCommandHarness()
        with patch("hermes_cli.goals.draft_contract") as draft:
            cprint = harness.run(cmd_text, goal_text=goal_text)
        return harness, draft, cprint

    def test_draft_no_park_true_goes_to_set(self):
        harness, draft, _ = self._run_draft(
            "/goal draft no-park: #t implement login"
        )
        assert draft.called
        # draft_contract must receive the objective WITHOUT the no-park token.
        assert draft.call_args.args[0] == "implement login"
        call = harness.mgr.set.call_args
        assert call.args[0] == "implement login"
        assert call.kwargs.get("no_park") is True

    def test_draft_no_park_false_explicit_default(self):
        harness, draft, _ = self._run_draft(
            "/goal draft no-park: #f implement login"
        )
        assert draft.call_args.args[0] == "implement login"
        call = harness.mgr.set.call_args
        assert call.args[0] == "implement login"
        assert call.kwargs.get("no_park") is False

    def test_draft_invalid_value_prints_exact_error_and_sets_no_goal(self):
        harness, _, cprint = self._run_draft(
            "/goal draft no-park: yes implement login"
        )
        assert harness.mgr.set.called is False
        rendered = "\n".join(str(a.args[0]) for a in cprint.call_args_list)
        assert (
            "Invalid value for no-park: must be #t or #f, got 'yes'" in rendered
        )

    def test_draft_numeric_value_prints_exact_error_and_sets_no_goal(self):
        harness, _, cprint = self._run_draft(
            "/goal draft no-park: 42 implement login"
        )
        assert harness.mgr.set.called is False
        rendered = "\n".join(str(a.args[0]) for a in cprint.call_args_list)
        assert "Invalid value for no-park: must be #t or #f, got '42'" in rendered

    def test_draft_without_prefix_behaves_as_today(self):
        harness, draft, _ = self._run_draft("/goal draft implement login")
        assert draft.call_args.args[0] == "implement login"
        call = harness.mgr.set.call_args
        assert call.args[0] == "implement login"
        assert call.kwargs.get("no_park") is False

    def test_draft_leading_ordering_passes_no_park_through(self):
        # /goal no-park: #t draft <obj> — the leading ordering is stripped at
        # the top of the handler and still reaches mgr.set as no_park=True.
        harness, draft, _ = self._run_draft(
            "/goal no-park: #t draft implement login"
        )
        assert draft.call_args.args[0] == "implement login"
        call = harness.mgr.set.call_args
        assert call.args[0] == "implement login"
        assert call.kwargs.get("no_park") is True


class TestHandleGoalCommandDontAskUserInput:
    """_handle_goal_command honors the dont-ask-user-input: prefix (US-007)."""

    def _run(self, cmd_text: str, goal_text: str = "fake goal"):
        harness = _GoalCommandHarness()
        cprint = harness.run(cmd_text, goal_text=goal_text)
        return harness, cprint

    def _assert_set_called(self, harness, expected_text, expected_flags):
        assert harness.mgr.set.called, "mgr.set was not called"
        call = harness.mgr.set.call_args
        assert call.args[0] == expected_text
        assert call.kwargs.get("no_park") is expected_flags[0]
        assert call.kwargs.get("dont_ask_user_input") is expected_flags[1]

    def test_dont_ask_true_goes_to_set(self):
        harness, _ = self._run(
            "/goal dont-ask-user-input: #t Implement login",
            goal_text="Implement login",
        )
        self._assert_set_called(harness, "Implement login", (False, True))

    def test_dont_ask_false_goes_to_set(self):
        harness, _ = self._run(
            "/goal dont-ask-user-input: #f Implement login",
            goal_text="Implement login",
        )
        self._assert_set_called(harness, "Implement login", (False, False))

    def test_no_prefix_keeps_backward_compat(self):
        harness, _ = self._run("/goal Implement login", goal_text="Implement login")
        self._assert_set_called(harness, "Implement login", (False, False))

    def test_both_flags_either_order(self):
        for cmd in (
            "/goal no-park: #t dont-ask-user-input: #t Migrate auth",
            "/goal dont-ask-user-input: #t no-park: #t Migrate auth",
        ):
            harness, _ = self._run(cmd, goal_text="Migrate auth")
            self._assert_set_called(harness, "Migrate auth", (True, True))

    def test_flag_token_never_misread_as_subcommand(self):
        # A flag token must not be mistaken for the draft subcommand or goal
        # text — the prefixes are stripped BEFORE subcommand routing.
        # (Drafting is dispatched through hermes_cli.goal_command._set, so the
        # assertion is on the stubbed draft_contract + mgr.set.)
        harness = _GoalCommandHarness()
        with patch("hermes_cli.goals.draft_contract") as draft:
            harness.run(
                "/goal dont-ask-user-input: #t draft build login",
                goal_text="draft build login",
            )
        assert draft.called
        assert draft.call_args.args[0] == "build login"
        call = harness.mgr.set.call_args
        assert call.args[0] == "build login"
        assert call.kwargs.get("dont_ask_user_input") is True

    def test_invalid_value_prints_exact_error_and_sets_no_goal(self):
        harness, cprint = self._run(
            "/goal dont-ask-user-input: maybe Implement login"
        )
        assert harness.mgr.set.called is False
        rendered = "\n".join(str(a.args[0]) for a in cprint.call_args_list)
        assert (
            "Invalid value for dont-ask-user-input: must be #t or #f, got 'maybe'"
            in rendered
        )


class TestHandleGoalDraftDontAskUserInput:
    """/goal draft honors the dont-ask-user-input: override (US-007)."""

    def _run_draft(self, cmd_text: str, goal_text: str = "fake goal"):
        harness = _GoalCommandHarness()
        with patch("hermes_cli.goals.draft_contract") as draft:
            cprint = harness.run(cmd_text, goal_text=goal_text)
        return harness, draft, cprint

    def test_draft_trailing_dont_ask_true_goes_to_set(self):
        harness, draft, _ = self._run_draft(
            "/goal draft dont-ask-user-input: #t implement login"
        )
        assert draft.called
        assert draft.call_args.args[0] == "implement login"
        call = harness.mgr.set.call_args
        assert call.args[0] == "implement login"
        assert call.kwargs.get("dont_ask_user_input") is True

    def test_draft_leading_dont_ask_true_goes_to_set(self):
        harness, draft, _ = self._run_draft(
            "/goal dont-ask-user-input: #t draft implement login"
        )
        assert draft.called
        assert draft.call_args.args[0] == "implement login"
        call = harness.mgr.set.call_args
        assert call.kwargs.get("dont_ask_user_input") is True

    def test_draft_composes_with_no_park_either_order(self):
        for cmd in (
            "/goal draft no-park: #t dont-ask-user-input: #t implement login",
            "/goal dont-ask-user-input: #t no-park: #t draft implement login",
        ):
            harness, draft, _ = self._run_draft(cmd)
            assert draft.call_args.args[0] == "implement login"
            call = harness.mgr.set.call_args
            assert call.kwargs.get("no_park") is True
            assert call.kwargs.get("dont_ask_user_input") is True

    def test_draft_invalid_value_prints_exact_error_and_sets_no_goal(self):
        harness, _, cprint = self._run_draft(
            "/goal draft dont-ask-user-input: 42 implement login"
        )
        assert harness.mgr.set.called is False
        rendered = "\n".join(str(a.args[0]) for a in cprint.call_args_list)
        assert (
            "Invalid value for dont-ask-user-input: must be #t or #f, got '42'"
            in rendered
        )


# ──────────────────────────────────────────────────────────────────────
# _build_continuation_prompt (US-003)
# ──────────────────────────────────────────────────────────────────────


class TestBuildContinuationPrompt:
    """The continuation-prompt builder must stay byte-identical to the
    public templates when dont_ask_user_input is unset, and drop every
    ask-the-user hand-off phrase when it is True (US-003)."""

    GOAL = "port the thing"
    CONTRACT_BLOCK = (
        "Verification: the new module passes its tests.\n"
        "Constraints: do not touch the legacy parser."
    )
    SUBGOALS_BLOCK = (
        "- Extra criterion 1: keep the public API stable.\n"
        "- Extra criterion 2: update the changelog."
    )
    # Phrases that signal an ask-the-user hand-off and must never appear in
    # a no-ask continuation prompt.
    HANDOFF_PHRASES = (
        "ask the user",
        "needs user input",
        "need input from the user",
        "say so clearly and stop",
    )
    # Autonomous-progress guidance that must replace the hand-off sentence.
    GUIDANCE_PHRASES = ("keep going", "work around it", "best judgment")

    def _plain(self, dont_ask_user_input=False):
        from hermes_cli.goals import _build_continuation_prompt

        return _build_continuation_prompt(
            self.GOAL, dont_ask_user_input=dont_ask_user_input
        )

    def _contract(self, dont_ask_user_input=False):
        from hermes_cli.goals import _build_continuation_prompt

        return _build_continuation_prompt(
            self.GOAL,
            contract_block=self.CONTRACT_BLOCK,
            dont_ask_user_input=dont_ask_user_input,
        )

    def _subgoals(self, dont_ask_user_input=False):
        from hermes_cli.goals import _build_continuation_prompt

        return _build_continuation_prompt(
            self.GOAL,
            subgoals_block=self.SUBGOALS_BLOCK,
            dont_ask_user_input=dont_ask_user_input,
        )

    # --- default flag: byte-identical to the public templates ---------

    def test_default_plain_byte_identical_to_template(self):
        from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

        assert self._plain() == CONTINUATION_PROMPT_TEMPLATE.format(goal=self.GOAL)

    def test_default_contract_byte_identical_to_contract_template(self):
        from hermes_cli.goals import CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE

        assert self._contract() == CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE.format(
            goal=self.GOAL, contract_block=self.CONTRACT_BLOCK
        )

    def test_default_subgoals_byte_identical_to_subgoals_template(self):
        from hermes_cli.goals import CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE

        assert self._subgoals() == CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
            goal=self.GOAL, subgoals_block=self.SUBGOALS_BLOCK
        )

    def test_default_output_contains_the_handoff_sentence(self):
        # The flag-False build must retain the original stop-and-ask sentence,
        # so the byte-identity above is not achieved by silently stripping it.
        assert "need input from the user, say so clearly and stop" in self._plain()
        assert "blocked and need\n        user input, say so clearly and stop".replace(
            "\n        ", " "
        ) in self._contract()
        assert "blocked and need input from the user" in self._subgoals()

    def test_default_contract_priority_over_subgoals(self):
        from hermes_cli.goals import (
            _build_continuation_prompt,
            CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE,
        )

        both = _build_continuation_prompt(
            self.GOAL,
            contract_block=self.CONTRACT_BLOCK,
            subgoals_block=self.SUBGOALS_BLOCK,
        )
        assert both == CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE.format(
            goal=self.GOAL, contract_block=self.CONTRACT_BLOCK
        )

    # --- no-ask: hand-off phrases removed ------------------------------

    def test_no_ask_plain_omits_all_handoff_phrases(self):
        prompt = self._plain(dont_ask_user_input=True)
        for phrase in self.HANDOFF_PHRASES:
            assert phrase not in prompt, (phrase, prompt)

    def test_no_ask_contract_omits_all_handoff_phrases(self):
        prompt = self._contract(dont_ask_user_input=True)
        for phrase in self.HANDOFF_PHRASES:
            assert phrase not in prompt, (phrase, prompt)

    def test_no_ask_subgoals_omits_all_handoff_phrases(self):
        prompt = self._subgoals(dont_ask_user_input=True)
        for phrase in self.HANDOFF_PHRASES:
            assert phrase not in prompt, (phrase, prompt)

    # --- no-ask: autonomous-progress guidance present -----------------

    def test_no_ask_plain_contains_autonomous_guidance(self):
        prompt = self._plain(dont_ask_user_input=True)
        for phrase in self.GUIDANCE_PHRASES:
            assert phrase in prompt, (phrase, prompt)

    def test_no_ask_contract_contains_autonomous_guidance(self):
        prompt = self._contract(dont_ask_user_input=True)
        for phrase in self.GUIDANCE_PHRASES:
            assert phrase in prompt, (phrase, prompt)

    def test_no_ask_subgoals_contains_autonomous_guidance(self):
        prompt = self._subgoals(dont_ask_user_input=True)
        for phrase in self.GUIDANCE_PHRASES:
            assert phrase in prompt, (phrase, prompt)

    # --- no-ask: priority + structure preserved ------------------------

    def test_no_ask_contract_priority_over_subgoals(self):
        from hermes_cli.goals import (
            _build_continuation_prompt,
            _CONTINUATION_PROMPT_WITH_CONTRACT_NO_ASK_TEMPLATE,
        )

        both = _build_continuation_prompt(
            self.GOAL,
            contract_block=self.CONTRACT_BLOCK,
            subgoals_block=self.SUBGOALS_BLOCK,
            dont_ask_user_input=True,
        )
        assert both == _CONTINUATION_PROMPT_WITH_CONTRACT_NO_ASK_TEMPLATE.format(
            goal=self.GOAL, contract_block=self.CONTRACT_BLOCK
        )

    def test_no_ask_still_targets_the_goal_verbatim(self):
        prompt = self._plain(dont_ask_user_input=True)
        assert self.GOAL in prompt

    def test_no_ask_keeps_completion_instruction(self):
        # The no-ask prompt must still tell the agent to stop when genuinely
        # finished — only the user-hand-off reason for stopping is removed.
        prompt = self._plain(dont_ask_user_input=True)
        assert "state so explicitly and stop" in prompt
