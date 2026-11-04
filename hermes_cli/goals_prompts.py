"""Goal prompt text and /goal flag parsing for ``hermes_cli.goals``.

Split out of the ``goals.py`` facade (its line cap): the judge/continuation prompt templates and the
SRFI-88-style ``no-park:`` / ``dont-ask-user-input:`` prefix parsers are pure text handling. The
judge and continuation builders late-import the public base templates from the facade, which imports
this module.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# No-ask variants of the continuation prompts above. When a goal is set with
# `dont-ask-user-input: #t` the agent is expected to run fully unattended —
# it must never stop to hand off to the user. The default templates tell the
# agent "if you are blocked and need input from the user, say so clearly and
# stop", which makes the model halt at decision points instead of pushing
# forward. These variants replace that sentence with autonomous-progress
# guidance so the agent keeps working until the goal is genuinely done or the
# budget is exhausted. They are deliberately private: callers select them
# through `_build_continuation_prompt`, never directly, so the flag-False
# output stays byte-identical to the public templates above.
_CONTINUATION_PROMPT_NO_ASK_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Continue working toward this goal. Take the next concrete step. "
    "If you believe the goal is complete, state so explicitly and stop. "
    "If you are blocked, work around it, use your best judgment, and keep going."
)

_CONTINUATION_PROMPT_WITH_CONTRACT_NO_ASK_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Completion contract:\n"
    "{contract_block}\n\n"
    "Continue working toward the outcome above. Take the next concrete step. "
    "Stay within the stated boundaries and do not violate the constraints. "
    "Before claiming the goal is done, satisfy the Verification criterion and "
    "show the concrete evidence (command output, file contents, test result). "
    "If you hit the stated stop condition or are otherwise blocked, work "
    "around it, use your best judgment, and keep going."
)

_CONTINUATION_PROMPT_WITH_SUBGOALS_NO_ASK_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Additional criteria the user added mid-loop:\n"
    "{subgoals_block}\n\n"
    "Continue working toward the goal AND all additional criteria. Take "
    "the next concrete step. If you believe the goal and every "
    "additional criterion are complete, state so explicitly and stop. "
    "If you are blocked, work around it, use your best judgment, and "
    "keep going."
)


def _build_continuation_prompt(
    goal: str,
    *,
    contract_block: Optional[str] = None,
    subgoals_block: Optional[str] = None,
    dont_ask_user_input: bool = False,
) -> str:
    """Build the continuation prompt sent back to an agent pursuing a goal.

    ``goal`` is the objective text. ``contract_block`` and ``subgoals_block``
    select the variant — contract takes priority over subgoals (mirroring
    ``GoalManager.next_continuation_prompt``, which folds subgoals into the
    contract block). With both ``None`` the plain template is used.

    When ``dont_ask_user_input=True`` the no-ask variants are used: the final
    "if you are blocked ... say so clearly and stop" sentence is replaced with
    autonomous guidance ("work around it, use your best judgment, and keep
    going") so a goal set with `dont-ask-user-input: #t` never stops to hand
    off to the user. When False (the default) the output is byte-identical to
    ``CONTINUATION_PROMPT_TEMPLATE`` / ``CONTINUATION_PROMPT_WITH_CONTRACT_``
    ``TEMPLATE`` / ``CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE``.
    """
    from hermes_cli.goals import (
        CONTINUATION_PROMPT_TEMPLATE,
        CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE,
        CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE,
    )

    if dont_ask_user_input:
        if contract_block is not None:
            return _CONTINUATION_PROMPT_WITH_CONTRACT_NO_ASK_TEMPLATE.format(
                goal=goal, contract_block=contract_block
            )
        if subgoals_block is not None:
            return _CONTINUATION_PROMPT_WITH_SUBGOALS_NO_ASK_TEMPLATE.format(
                goal=goal, subgoals_block=subgoals_block
            )
        return _CONTINUATION_PROMPT_NO_ASK_TEMPLATE.format(goal=goal)
    if contract_block is not None:
        return CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE.format(
            goal=goal, contract_block=contract_block
        )
    if subgoals_block is not None:
        return CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
            goal=goal, subgoals_block=subgoals_block
        )
    return CONTINUATION_PROMPT_TEMPLATE.format(goal=goal)


# The goal judge can return a WAIT verdict that parks the loop — the loop
# stops burning turns and pauses until a background process exits, a session
# trigger fires, or a timer elapses. A user can disable this per-goal with
# `no-park: #t` on /goal. For such goals the system prompt is built WITHOUT
# the WAIT paragraph (and without the wait JSON shapes or the four-verdicts
# wording) so the judge never emits a verdict the caller would
# have to downgrade. The prompt is therefore composed from the sections
# below; JUDGE_SYSTEM_PROMPT stays BASE + WAIT for backward compatibility.

# Preambles — the WAIT/no-park variants differ only in how many verdicts the
# judge is told to decide between.
_JUDGE_SYSTEM_PROMPT_PREAMBLE = (
    "You are a strict judge evaluating whether an autonomous agent has "
    "achieved a user's stated goal. You receive the goal text, the agent's "
    "most recent response, and — when present — a list of background "
    "processes the agent has running. Decide one of four verdicts.\n\n"
)

_JUDGE_SYSTEM_PROMPT_PREAMBLE_NO_WAIT = (
    "You are a strict judge evaluating whether an autonomous agent has "
    "achieved a user's stated goal. You receive the goal text, the agent's "
    "most recent response, and — when present — a list of background "
    "processes the agent has running. Decide one of three verdicts.\n\n"
)

# No-ask preambles. With `dont-ask-user-input: #t` the goal runs unattended
# and BLOCKED is not offered (there is no user to hand off to), so the
# no-ask variants drop one more verdict: three (DONE/WAIT/CONTINUE) when
# parking is allowed, two (DONE/CONTINUE) when no-park is also set.
_JUDGE_SYSTEM_PROMPT_PREAMBLE_NO_ASK = (
    "You are a strict judge evaluating whether an autonomous agent has "
    "achieved a user's stated goal. You receive the goal text, the agent's "
    "most recent response, and — when present — a list of background "
    "processes the agent has running. Decide one of three verdicts.\n\n"
)

_JUDGE_SYSTEM_PROMPT_PREAMBLE_NO_ASK_NO_WAIT = (
    "You are a strict judge evaluating whether an autonomous agent has "
    "achieved a user's stated goal. You receive the goal text, the agent's "
    "most recent response, and — when present — a list of background "
    "processes the agent has running. Decide one of two verdicts.\n\n"
)

# The "DONE — the goal is fully satisfied" paragraph is split from the
# BLOCKED verdict paragraph so a single builder can assemble the
# `dont-ask-user-input` variant that omits BLOCKED without duplicating any
# DONE wording. The public constant JUDGE_SYSTEM_PROMPT_DONE stays
# byte-identical to the original genuine-completion paragraph.
_JUDGE_SYSTEM_PROMPT_DONE_GENUINE = (
    "DONE — the goal is fully satisfied:\n"
    "- The response explicitly confirms the goal was completed, OR\n"
    "- The response clearly shows the final deliverable was produced.\n"
    "DONE requires the deliverable to actually exist. If the response only "
    "explains why the goal cannot be reached, the verdict is BLOCKED, not "
    "DONE.\n\n"
)

# DONE is exactly the genuine-completion paragraph above: in the current
# prompt the blocked / user-input hand-off is a verdict of its own (BLOCKED),
# not a DONE reason, so there is no hand-off clause to append.
JUDGE_SYSTEM_PROMPT_DONE = _JUDGE_SYSTEM_PROMPT_DONE_GENUINE

# When a goal is set with `dont-ask-user-input: #t` the agent runs fully
# unattended and has no user to hand off to. The no-ask build omits the
# BLOCKED verdict entirely, so this DONE paragraph drops the "verdict is
# BLOCKED, not DONE" pointer and instead states plainly that a block is NOT
# a stop condition — the agent must keep working.
_JUDGE_SYSTEM_PROMPT_DONE_NO_ASK = (
    "DONE — the goal is fully satisfied:\n"
    "- The response explicitly confirms the goal was completed, OR\n"
    "- The response clearly shows the final deliverable was produced.\n"
    "DONE requires the deliverable to actually exist. If the response only "
    "reports the work is blocked or could not make progress, that is NOT "
    "DONE: the goal runs unattended, so the agent must keep working and "
    "continue.\n\n"
)

JUDGE_SYSTEM_PROMPT_BLOCKED = (
    "BLOCKED — the goal cannot be satisfied as stated:\n"
    "- The response explains the goal is genuinely unachievable (impossible, "
    "out of scope, no valid path to the deliverable), or refuses to "
    "fabricate a deliverable that cannot exist, OR\n"
    "- The response explains progress is blocked and the next step needs "
    "user input to proceed.\n"
    "Return BLOCKED with the reason describing what is blocking. BLOCKED is "
    "a refusal, not a completion — never return BLOCKED for a goal that "
    "was achieved.\n"
    "When the block is an error the agent hit (an HTTP status, an API, "
    "sign-in or token failure), quote the error text verbatim in the reason "
    "and attribute it only to a provider, service or credential the response "
    "itself names. Never infer one the response does not name — an unnamed "
    "401 belongs to the model provider the agent was calling, not to some "
    "other service's token.\n\n"
)

JUDGE_SYSTEM_PROMPT_WAIT = (
    "WAIT — the goal is NOT done, but the next step is to wait for async "
    "work to finish rather than act again. Choose this ONLY when the agent's "
    "progress is genuinely gated on something running on its own:\n"
    "- A background process listed below is still running AND the response "
    "shows the agent is waiting on its result (e.g. a CI poller, build, "
    "test run, deploy). If the process has a session id, return it in "
    "``wait_on_session`` — that releases when the process exits OR its "
    "watch_patterns trigger fires (use this for a long-lived watcher that "
    "signals mid-run and may never exit). Otherwise return its pid in "
    "``wait_on_pid`` (releases on exit only).\n"
    "- The agent says it is rate-limited / backing off / must wait a fixed "
    "period — return seconds in ``wait_for_seconds``.\n"
    "- The agent has delegated subagents still running (stated below as "
    "active delegations) and the response says it is waiting on them with "
    "nothing else dispatchable — return ``wait_for_seconds`` between 600 and "
    "1800. Their results wake the agent on their own; re-poking it now only "
    "produces a status recap.\n"
    "Picking WAIT parks the loop without burning a turn; it resumes "
    "automatically when the pid exits or the time elapses. Do NOT pick WAIT "
    "just because work remains — only when re-poking now would be pure "
    "busy-work because the agent can't progress until the async thing "
    "finishes."
)

JUDGE_SYSTEM_PROMPT_CONTINUE = (
    "CONTINUE — not done, and there is a concrete next step the agent can "
    "take right now. This is the default when in doubt.\n\n"
)

# Reply-shapes sections. The WAIT variant instructs the model on the wait_*
# JSON shapes; the no-park variant omits them entirely.
_JUDGE_SYSTEM_PROMPT_REPLY = (
    "Reply ONLY with a single JSON object on one line. Shapes:\n"
    '{"verdict": "done", "reason": "<one sentence>"}\n'
    '{"verdict": "blocked", "reason": "<one sentence>"}\n'
    '{"verdict": "continue", "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_on_session": "<id>", "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_on_pid": <int>, "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_for_seconds": <int>, "reason": "<one sentence>"}\n'
    "The legacy shape {\"done\": <true|false>, \"reason\": \"...\"} is still "
    "accepted (true=done, false=continue)."
)

_JUDGE_SYSTEM_PROMPT_REPLY_NO_WAIT = (
    "Reply ONLY with a single JSON object on one line. Shapes:\n"
    '{"verdict": "done", "reason": "<one sentence>"}\n'
    '{"verdict": "blocked", "reason": "<one sentence>"}\n'
    '{"verdict": "continue", "reason": "<one sentence>"}\n'
    "The legacy shape {\"done\": <true|false>, \"reason\": \"...\"} is still "
    "accepted (true=done, false=continue)."
)

# No-ask reply shapes (BLOCKED is not offered for unattended goals).
_JUDGE_SYSTEM_PROMPT_REPLY_NO_ASK = (
    "Reply ONLY with a single JSON object on one line. Shapes:\n"
    '{"verdict": "done", "reason": "<one sentence>"}\n'
    '{"verdict": "continue", "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_on_session": "<id>", "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_on_pid": <int>, "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_for_seconds": <int>, "reason": "<one sentence>"}\n'
    "The legacy shape {\"done\": <true|false>, \"reason\": \"...\"} is still "
    "accepted (true=done, false=continue)."
)

_JUDGE_SYSTEM_PROMPT_REPLY_NO_ASK_NO_WAIT = (
    "Reply ONLY with a single JSON object on one line. Shapes:\n"
    '{"verdict": "done", "reason": "<one sentence>"}\n'
    '{"verdict": "continue", "reason": "<one sentence>"}\n'
    "The legacy shape {\"done\": <true|false>, \"reason\": \"...\"} is still "
    "accepted (true=done, false=continue)."
)

# Everything except the WAIT paragraph. The WAIT paragraph itself is a
# self-contained block appended at the end so JUDGE_SYSTEM_PROMPT can be
# expressed as BASE + WAIT (and so a no-park build simply omits WAIT).
JUDGE_SYSTEM_PROMPT_BASE = (
    _JUDGE_SYSTEM_PROMPT_PREAMBLE
    + JUDGE_SYSTEM_PROMPT_DONE
    + JUDGE_SYSTEM_PROMPT_BLOCKED
    + JUDGE_SYSTEM_PROMPT_CONTINUE
    + _JUDGE_SYSTEM_PROMPT_REPLY
    + "\n\n"
)

# Backward-compatible default: the full judge prompt including the WAIT
# verdict. Every existing caller that references JUDGE_SYSTEM_PROMPT keeps
# identical behavior.
JUDGE_SYSTEM_PROMPT = JUDGE_SYSTEM_PROMPT_BASE + JUDGE_SYSTEM_PROMPT_WAIT

# Judge prompt line for live delegated subagents (WAIT-for-seconds vs CONTINUE).
JUDGE_DELEGATIONS_BLOCK_TEMPLATE = (
    "Active delegations: the agent has {count} delegated subagent batch(es) still running; "
    "their results are delivered to it automatically when they finish.\n\n"
)


def _build_judge_system_prompt(
    no_park: bool = False, dont_ask_user_input: bool = False
) -> str:
    """Return the goal-judge system prompt.

    With ``no_park=False`` (the default) this is exactly
    ``JUDGE_SYSTEM_PROMPT``: the judge may emit a ``wait`` verdict that parks
    the loop. With ``no_park=True`` the WAIT verdict is removed entirely —
    the preamble wording drops to three verdicts, the WAIT paragraph is
    omitted, and the reply shapes no longer include the ``wait_on_*`` forms,
    so the judge is never told a verdict exists that the caller would
    downgrade. DONE, BLOCKED and CONTINUE are described fully in both
    variants.

    ``dont_ask_user_input`` (default False) drops the BLOCKED verdict (and
    the blocked / user-input hand-off) from the build: when a goal runs
    unattended there is no user to hand off to, so a merely "blocked"
    response is a CONTINUE signal, never a DONE reason. It composes with
    ``no_park`` — one builder assembles every combination — and when False
    (with either ``no_park`` value) the output is byte-identical to
    today's builds.
    """
    done_section = (
        _JUDGE_SYSTEM_PROMPT_DONE_NO_ASK
        if dont_ask_user_input
        else JUDGE_SYSTEM_PROMPT_DONE
    )
    if dont_ask_user_input:
        # Unattended goal: no user to hand off to, so BLOCKED is not offered
        # — a merely blocked response is a CONTINUE signal, and the judge
        # decides between DONE (genuine completion only), WAIT (when parking
        # is allowed) and CONTINUE.
        return (
            (
                _JUDGE_SYSTEM_PROMPT_PREAMBLE_NO_ASK_NO_WAIT
                if no_park
                else _JUDGE_SYSTEM_PROMPT_PREAMBLE_NO_ASK
            )
            + done_section
            + (JUDGE_SYSTEM_PROMPT_WAIT if not no_park else "")
            + JUDGE_SYSTEM_PROMPT_CONTINUE
            + (
                _JUDGE_SYSTEM_PROMPT_REPLY_NO_ASK_NO_WAIT
                if no_park
                else _JUDGE_SYSTEM_PROMPT_REPLY_NO_ASK
            )
        )
    if no_park:
        return (
            _JUDGE_SYSTEM_PROMPT_PREAMBLE_NO_WAIT
            + done_section
            + JUDGE_SYSTEM_PROMPT_BLOCKED
            + JUDGE_SYSTEM_PROMPT_CONTINUE
            + _JUDGE_SYSTEM_PROMPT_REPLY_NO_WAIT
        )
    return JUDGE_SYSTEM_PROMPT

# Judge prompt block listing running background processes (WAIT vs CONTINUE, which pid).
JUDGE_BACKGROUND_BLOCK_TEMPLATE = (
    "Background processes the agent currently has running (it may be waiting "
    "on one of these):\n{background_lines}\n\n"
)

JUDGE_USER_PROMPT_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "{background_block}"
    "Current time: {current_time}\n\n"
    "Is the goal satisfied — done, blocked, continue, or wait?"
)

# With /subgoal criteria: the judge must see ALL of them met, not just the original goal.
JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Additional criteria the user added mid-loop (all must also be "
    "satisfied for the goal to be DONE):\n{subgoals_block}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "{background_block}"
    "Current time: {current_time}\n\n"
    "Decision: For each numbered criterion above, find concrete "
    "evidence in the agent's response that the criterion is "
    "satisfied. Do not accept generic phrases like 'all requirements "
    "met' or 'implying it was done' — require specific evidence (a "
    "file contents excerpt, an output line, a command result). If "
    "ANY criterion lacks specific evidence in the response, the goal "
    "is NOT done — return CONTINUE (or WAIT if blocked on a listed "
    "background process).\n\n"
    "Is the goal AND every additional criterion satisfied?"
)

# With a contract: DONE strictly against the Verification criterion; a violated constraint refuses.
JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Completion contract (the authoritative definition of done):\n"
    "{contract_block}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "{background_block}"
    "Current time: {current_time}\n\n"
    "Decision rules:\n"
    "- The goal is DONE only when the Verification criterion is satisfied AND "
    "the response shows concrete evidence of it (a command result, file "
    "contents excerpt, test/benchmark output) — not a claim like 'done' or "
    "'all tests pass' without evidence.\n"
    "- If any stated Constraint was violated, the goal is NOT done — CONTINUE.\n"
    "- If the response shows the agent is waiting on a listed background "
    "process to satisfy the Verification criterion (e.g. CI is the "
    "verification and it's still running), return WAIT on that process "
    "instead of re-poking — re-poking now would be pure busy-work.\n"
    "- If the response explains the work is genuinely unachievable or hits "
    "the stated Stop condition and needs user input, the goal is NOT done — "
    "return BLOCKED with the reason describing the block.\n"
    "- Otherwise the goal is NOT done — CONTINUE.\n\n"
    "Is the goal satisfied per its completion contract — done, blocked, continue, or wait?"
)

# /goal draft: turn a plain objective into a reviewable contract (after Codex's "draft the goal").


# No-ask variant of the contract judge template above. When a goal is set
# with `dont-ask-user-input: #t` the agent runs fully unattended and never
# hands off to the user, so the "blocked / unachievable / needs user input →
# DONE" decision rule must flip: a block is a CONTINUE signal, not a DONE
# reason. This template keeps every other decision rule identical and only
# rewrites that one bullet. Deliberately private — callers route through
# `_build_judge_user_prompt`, never directly, so the flag-False output stays
# byte-identical to `JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE`.
_JUDGE_USER_PROMPT_WITH_CONTRACT_NO_ASK_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Completion contract (the authoritative definition of done):\n"
    "{contract_block}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "{background_block}"
    "Current time: {current_time}\n\n"
    "Decision rules:\n"
    "- The goal is DONE only when the Verification criterion is satisfied AND "
    "the response shows concrete evidence of it (a command result, file "
    "contents excerpt, test/benchmark output) — not a claim like 'done' or "
    "'all tests pass' without evidence.\n"
    "- If any stated Constraint was violated, the goal is NOT done — CONTINUE.\n"
    "- If the response shows the agent is waiting on a listed background "
    "process to satisfy the Verification criterion (e.g. CI is the "
    "verification and it's still running), return WAIT on that process "
    "instead of re-poking — re-poking now would be pure busy-work.\n"
    "- If the response reports the work is blocked / unachievable / the "
    "stated Stop condition was hit, treat it as CONTINUE: the goal runs "
    "unattended, so the agent must keep working and take the next concrete "
    "step rather than stop.\n"
    "- Otherwise the goal is NOT done — CONTINUE.\n\n"
    "Is the goal satisfied per its completion contract — done, continue, or wait?"
)


def _build_judge_user_prompt(
    goal: str,
    response: str,
    background_block: str,
    current_time: str,
    *,
    contract_block: Optional[str] = None,
    subgoals_block: Optional[str] = None,
    dont_ask_user_input: bool = False,
) -> str:
    """Build the goal-judge user prompt (contract > subgoals > plain).

    Mirrors the exact selection priority and truncation rules ``judge_goal``
    uses: a contract block wins over a subgoals block, and plain is the
    fallback when neither is provided.

    When ``dont_ask_user_input=True`` the contract template is swapped for
    its no-ask variant, whose decision rules treat a merely "blocked"
    response as CONTINUE rather than DONE — the goal runs unattended, so the
    agent must keep working instead of hand-waving a stop. The plain and
    subgoals templates carry no user-input hand-off phrasing and are used
    unchanged in both flag states. When False (the default) the output is
    byte-identical to ``JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE`` /
    ``JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE`` /
    ``JUDGE_USER_PROMPT_TEMPLATE``.
    """
    from hermes_cli.goals import _JUDGE_RESPONSE_SNIPPET_CHARS, _truncate

    if contract_block is not None:
        template = (
            _JUDGE_USER_PROMPT_WITH_CONTRACT_NO_ASK_TEMPLATE
            if dont_ask_user_input
            else JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE
        )
        return template.format(
            goal=_truncate(goal, 2000),
            contract_block=_truncate(contract_block, 2500),
            response=_truncate(response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )
    if subgoals_block is not None:
        return JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            subgoals_block=_truncate(subgoals_block, 2000),
            response=_truncate(response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )
    return JUDGE_USER_PROMPT_TEMPLATE.format(
        goal=_truncate(goal, 2000),
        response=_truncate(response, _JUDGE_RESPONSE_SNIPPET_CHARS),
        background_block=background_block,
        current_time=current_time,
    )


# System prompt for /goal draft — turns a plain-language objective into a
# structured completion contract the user can review before activating.
# Adapted from Codex's "let Codex draft the goal" guidance.
DRAFT_CONTRACT_SYSTEM_PROMPT = (
    "You turn a user's plain-language objective into a structured completion "
    "contract for an autonomous coding agent. The contract has five fields:\n"
    "- outcome: the single end state that must be true when done\n"
    "- verification: the specific test / command / artifact that PROVES the "
    "outcome (must be concrete and checkable)\n"
    "- constraints: what must NOT change or regress\n"
    "- boundaries: which files, dirs, tools, or systems are in scope\n"
    "- stop_when: the condition under which the agent should stop and ask "
    "for human input instead of pushing on\n\n"
    "Infer sensible, specific values from the objective and any project "
    "context implied by it. Prefer concrete verification (a named test "
    "command, a build, a benchmark) over vague phrases. Keep each field to "
    "one or two sentences. If a field genuinely cannot be inferred, use an "
    "empty string for it.\n\n"
    "Reply ONLY with a single JSON object on one line:\n"
    '{"outcome": "...", "verification": "...", "constraints": "...", '
    '"boundaries": "...", "stop_when": "..."}'
)


# ── Completion contract ───────────────────────────────────────────────

# The five contract fields, in display order (after OpenAI Codex's "strong goal" guidance: what
# "done" means, how to prove it, what must not regress, what is in bounds, when to stop and ask).
# A bare free-form goal stays fully supported — empty fields are omitted from every prompt.
_CONTRACT_FIELDS = ("outcome", "verification", "constraints", "boundaries", "stop_when")

_CONTRACT_LABELS = {
    "outcome": "Outcome", "verification": "Verification", "constraints": "Constraints",
    "boundaries": "Boundaries", "stop_when": "Stop when blocked",
}

# Inline-input aliases the user may type before a value (`verify: tests pass`, `done when: ...`).
_CONTRACT_ALIASES = {
    "outcome": "outcome", "goal": "outcome", "done": "outcome", "done when": "outcome",
    "verification": "verification", "verify": "verification", "verified by": "verification",
    "evidence": "verification", "proof": "verification",
    "constraints": "constraints", "constraint": "constraints", "preserve": "constraints",
    "must not": "constraints", "do not change": "constraints",
    "boundaries": "boundaries", "boundary": "boundaries", "scope": "boundaries",
    "allowed": "boundaries", "files": "boundaries",
    "stop when": "stop_when", "stop_when": "stop_when", "blocked": "stop_when",
    "stop if blocked": "stop_when", "give up when": "stop_when",
}


# Leading SRFI-88-style ``no-park:`` / ``dont-ask-user-input:`` tokens accepted
# at the start of a /goal argument (case-insensitive identifier). A single token
# after the colon is the value; anything after it is the goal text the token was
# prepended to.
_NO_PARK_PREFIX_RE = re.compile(r"^no-park\s*:\s*(.*)$", re.IGNORECASE)
_DONT_ASK_USER_INPUT_PREFIX_RE = re.compile(
    r"^dont-ask-user-input\s*:\s*(.*)$", re.IGNORECASE
)


def _parse_srfi88_value(key: str, value: str) -> bool:
    """Validate an SRFI-88-style boolean token for a per-goal flag.

    Accepts exactly ``#t`` → ``True`` and ``#f`` → ``False``. Any other
    value (including an empty token) raises ``ValueError`` naming the flag
    ``key`` and the offending ``value``. Shared by every per-goal SRFI-88
    flag (``no-park``, ``dont-ask-user-input``) so the flags cannot drift in
    accepted syntax or error text.
    """
    if value == "#t":
        return True
    if value == "#f":
        return False
    raise ValueError(f"Invalid value for {key}: must be #t or #f, got '{value}'")


def parse_no_park_prefix(arg: str) -> tuple[bool, str]:
    """Parse an optional leading ``no-park:`` token from a /goal argument.

    SRFI-88-style syntax: ``no-park: #t <goal text>`` enables the per-goal
    WAIT-park override (``no_park=True``), ``no-park: #f <goal text>`` keeps
    the default behavior (``no_park=False``). Any other value raises
    ``ValueError`` with a message naming the offending token. When the token
    is absent the argument is returned unchanged with ``no_park=False`` (the
    default) — the caller should treat that exactly like today.

    The returned remainder has the ``no-park:`` token (and its value) stripped
    so it can be fed straight to ``parse_contract`` without polluting the
    goal headline or contract.
    """
    stripped = arg.lstrip()
    match = _NO_PARK_PREFIX_RE.match(stripped) if stripped else None
    if not match:
        return False, arg
    tokens = match.group(1).split(None, 1)
    value = tokens[0] if tokens else ""
    remainder = tokens[1].strip() if len(tokens) > 1 else ""
    return _parse_srfi88_value("no-park", value), remainder


def parse_dont_ask_user_input_prefix(arg: str) -> tuple[bool, str]:
    """Parse an optional leading ``dont-ask-user-input:`` token from a /goal argument.

    SRFI-88-style syntax: ``dont-ask-user-input: #t <goal text>`` marks the
    goal as fully unattended (``dont_ask_user_input=True`` — the continuation
    and judge prompts never ask the user for input), ``dont-ask-user-input: #f
    <goal text>`` keeps the default behavior (``dont_ask_user_input=False``).
    Any other value raises ``ValueError`` naming the offending token. When the
    token is absent the argument is returned unchanged with ``False`` (the
    default) — the caller should treat that exactly like today.

    The returned remainder has the ``dont-ask-user-input:`` token (and its
    value) stripped so it can be fed straight to ``parse_contract`` without
    polluting the goal headline or contract.
    """
    stripped = arg.lstrip()
    match = _DONT_ASK_USER_INPUT_PREFIX_RE.match(stripped) if stripped else None
    if not match:
        return False, arg
    tokens = match.group(1).split(None, 1)
    value = tokens[0] if tokens else ""
    remainder = tokens[1].strip() if len(tokens) > 1 else ""
    return _parse_srfi88_value("dont-ask-user-input", value), remainder


def parse_goal_flags(arg: str) -> tuple[bool, bool, str]:
    """Parse the optional leading SRFI-88-style per-goal flag tokens.

    Repeatedly strips leading ``no-park:`` and ``dont-ask-user-input:``
    tokens (in ANY order, each possibly repeated — last one wins) from the
    start of a /goal argument and returns ``(no_park, dont_ask_user_input,
    remainder)``. Value validation is delegated to the per-key parsers
    (:func:`parse_no_park_prefix` / :func:`parse_dont_ask_user_input_prefix`),
    which share :func:`_parse_srfi88_value` — so an invalid value raises the
    exact same ``ValueError`` as the single-prefix parsers, surfaced
    left-to-right.

    Examples::

        parse_goal_flags("no-park: #t dont-ask-user-input: #t ship it")
        # -> (True, True, "ship it")
        parse_goal_flags("dont-ask-user-input: #t no-park: #f ship it")
        # -> (False, True, "ship it")
        parse_goal_flags("ship it")  # -> (False, False, "ship it")

    The returned remainder has every flag token (and its value) stripped so
    it can be fed straight to ``parse_contract`` / subcommand detection
    without polluting the goal headline, contract, or command routing.
    """
    no_park = False
    dont_ask_user_input = False
    remainder = arg
    while True:
        candidate, rest = parse_no_park_prefix(remainder)
        if rest != remainder:
            no_park = candidate
            remainder = rest
            continue
        candidate, rest = parse_dont_ask_user_input_prefix(remainder)
        if rest != remainder:
            dont_ask_user_input = candidate
            remainder = rest
            continue
        return no_park, dont_ask_user_input, remainder


_USER_INPUT_HANDOFF_PHRASES: frozenset[str] = frozenset({
    "user input",
    "input from the user",
    "needs your input",
    "ask the user",
    "ask you",
    "your input",
    "hand off",
    "handoff",
    "stopped to ask",
    "await your",
    "waiting for your",
    "need you to",
})


def _reason_indicates_user_input_block(reason: str) -> bool:
    """Return True when a judge reason is really a hand-off to the user.

    Case-insensitive substring match against explicit user-input hand-off
    language (``_USER_INPUT_HANDOFF_PHRASES``). Used only as the deterministic
    backstop for ``dont_ask_user_input=True`` judge calls — see the note above
    for why bare ``blocked``/``stuck`` wording is excluded.
    """
    if not reason:
        return False
    lowered = reason.lower()
    return any(phrase in lowered for phrase in _USER_INPUT_HANDOFF_PHRASES)
