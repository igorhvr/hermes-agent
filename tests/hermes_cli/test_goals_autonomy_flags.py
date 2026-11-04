"""Per-goal autonomy flags on /goal: ``no-park:`` and ``dont-ask-user-input:``.

Covers judge verdict downgrades and prompt builds, GoalState/GoalManager persistence, the prefix
parsers, /goal command wiring, and the continuation prompts. Split out of ``test_goals.py`` (its
line cap); the shared ``hermes_home`` fixture comes from there.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))


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
        _captured, result = self._run_judge(self.WAIT_RESPONSE, no_park=True)
        verdict, reason, parse_failed, wait_directive, transport_failed = result
        assert verdict == "continue"
        assert wait_directive is None
        assert reason == "waiting on build"
        assert parse_failed is False
        assert transport_failed is False

    def test_no_park_true_downgrades_session_wait_too(self):
        _captured, result = self._run_judge(
            '{"verdict": "wait", "reason": "waiting on session", '
            '"wait_on_session": "sess-1"}',
            no_park=True,
        )
        verdict, _reason, _pf, wait_directive, _tf = result
        assert verdict == "continue"
        assert wait_directive is None

    def test_no_park_false_keeps_wait_verdict_unchanged(self):
        _captured, result = self._run_judge(self.WAIT_RESPONSE, no_park=False)
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
        _captured, result = self._run_judge(self.HANDOFF_DONE, dont_ask_user_input=True)
        verdict, reason, parse_failed, wait_directive, transport_failed = result
        assert verdict == "continue"
        # the reason is preserved through the downgrade
        assert reason == "the agent needs user input to continue"
        assert parse_failed is False
        assert wait_directive is None
        assert transport_failed is False

    def test_handoff_done_kept_when_dont_ask_user_input_false(self):
        # The same body with the flag False (default) is unchanged — done.
        _captured, result = self._run_judge(self.HANDOFF_DONE, dont_ask_user_input=False)
        verdict, reason, parse_failed, _wd, transport_failed = result
        assert verdict == "done"
        assert reason == "the agent needs user input to continue"
        assert parse_failed is False
        assert transport_failed is False

    def test_handoff_done_kept_by_default_positional(self):
        # The kanban path calls judge_goal(goal_text, last_response)
        # positionally — dont_ask_user_input must default to False so a
        # hand-off body stays done exactly as before this feature.
        _captured, result = self._run_judge(self.HANDOFF_DONE)
        verdict, _reason, _pf, _wd, _tf = result
        assert verdict == "done"

    def test_genuine_completion_not_overdowngraded(self):
        _captured, result = self._run_judge(
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
        _captured, result = self._run_judge(
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
        _captured, result = self._run_judge(
            '{"verdict": "wait", "reason": "waiting on build", "wait_on_pid": 4242}',
            dont_ask_user_input=True,
        )
        verdict, _reason, _pf, wait_directive, _tf = result
        assert verdict == "wait"
        assert wait_directive == {"pid": 4242}

    def test_handoff_done_downgraded_even_when_no_park_true(self):
        # Both downgrades are independent: with no_park=True and
        # dont_ask_user_input=True, a hand-off done is still downgraded.
        _captured, result = self._run_judge(
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
        # A plain offline CLI keeps main's in-process managers: with a gateway session store
        # and no open gateway connection the automation lookup would answer None.
        self.shell._session_db = None
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
