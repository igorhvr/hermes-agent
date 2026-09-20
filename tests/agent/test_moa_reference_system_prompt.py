"""
Test that the MoA reference system prompt contains explicit warnings
against claiming tool execution.

Related issue: #61452
"""

from agent.moa_loop import _ADVISORY_INSTRUCTION, _REFERENCE_SYSTEM_PROMPT


def test_reference_system_prompt_prohibits_claiming_execution():
    """
    Verify that the reference system prompt contains explicit warnings
    against claiming tool execution.

    The prompt should:
    1. State that reference models cannot execute anything
    2. Warn against claiming/implying execution
    3. Provide bad/good examples

    This addresses #61452 where reference models were fabricating
    tool execution in their text output.
    """
    prompt_lower = _REFERENCE_SYSTEM_PROMPT.lower()

    # Critical constraints
    assert "you cannot call tools" in prompt_lower or "you do not execute" in prompt_lower, \
        "Prompt must explicitly state that reference models cannot execute"

    assert "never claim" in prompt_lower or "never imply" in prompt_lower, \
        "Prompt must warn against claiming/implying execution"

    # Check for examples (helps models understand what NOT to do)
    assert "bad:" in prompt_lower or "avoid:" in prompt_lower, \
        "Prompt should provide negative examples"

    # Specific action verbs that should NOT appear as claimed actions
    # (these are common patterns of hallucinated execution)
    forbidden_patterns = [
        "i ran", "i executed", "i downloaded", "i accessed",
        "i checked", "i called", "i browsed"
    ]

    # The prompt should mention these as bad examples
    # (i.e., in the context of what to avoid, not as instruction)
    has_any_forbidden = any(
        f"bad: \"{pattern}" in _REFERENCE_SYSTEM_PROMPT.lower() or
        f"avoid \"{pattern}" in _REFERENCE_SYSTEM_PROMPT.lower()
        for pattern in forbidden_patterns
    )

    # At least one bad example pattern should exist
    assert has_any_forbidden or "examples" in _REFERENCE_SYSTEM_PROMPT.lower(), \
        "Prompt should contain examples of what to avoid"


def test_reference_system_prompt_structure():
    """
    Verify the reference system prompt has a clear structure.

    A well-structured prompt helps models follow instructions better.
    """
    # Prompt should not be empty
    assert len(_REFERENCE_SYSTEM_PROMPT) > 100, \
        "Reference system prompt should be substantive"

    # Should have multiple paragraphs (structured guidance)
    assert _REFERENCE_SYSTEM_PROMPT.count("\n\n") >= 2, \
        "Prompt should be structured with multiple sections"

    # Should contain the word "advisor" (defines role)
    assert "advisor" in _REFERENCE_SYSTEM_PROMPT.lower(), \
        "Prompt should clearly define the advisor role"


def test_reference_system_prompt_anti_hallucination_clauses():
    """
    Verify the reference system prompt carries the strengthened clauses that
    prevent tool-execution narration by a tool-less advisor.

    Regression for the observed bug where the reference advisor (no tools)
    fabricated tool-output narratives ("output was garbled", "came back
    empty", "hit SSL errors", "returned 400") that only the acting agent
    could have observed. The prompt must:
    1. State the advisor has no tools and no tool results
    2. Prohibit emitting output/log/error-shaped text
    3. Require "unclear from the transcript" honesty instead of invention
    4. Attribute every event to the acting agent in third person

    Behavior-contract assertions only: substring checks, no full-text equality.
    """
    prompt = _REFERENCE_SYSTEM_PROMPT
    prompt_lower = prompt.lower()

    # No-tools framing: the advisor must never mistake transcript tool events
    # for its own capabilities or observations.
    assert "you have no tools and no tool results" in prompt_lower, \
        "Prompt must state the advisor has no tools and no tool results"

    # Prohibition on emitting tool/output/log/error-shaped text.
    assert "never output any" in prompt_lower, \
        "Prompt must prohibit emitting tool/output/log-shaped text"
    assert "log lines, error strings, status messages" in prompt_lower, \
        "Prompt must forbid log/error/status-shaped text"
    assert "tool-call blocks" in prompt_lower, \
        "Prompt must forbid tool-call blocks"

    # Honesty clause: say so when the transcript does not establish a fact.
    assert "unclear from the transcript" in prompt_lower, \
        "Prompt must require explicit 'unclear from the transcript' honesty"

    # Third-person attribution: events belong to the acting agent, and output
    # must never be presented as the advisor's own observation.
    assert "the acting agent's" in prompt, \
        "Prompt must attribute transcript events to the acting agent"
    assert "third person" in prompt_lower, \
        "Prompt must require third-person attribution"

    # Corrupted-looking tool results are the acting agent's report to advise
    # on re-verification, never something to re-run ourselves.
    assert "acting agent's report" in prompt_lower, \
        "Prompt must treat tool results as the acting agent's report"
    assert "re-verify" in prompt_lower, \
        "Prompt must advise re-verification instead of re-running"


def test_advisory_instruction_reminds_constraints_at_generation_point():
    """
    The trailing synthetic advisory instruction must restate the no-tools
    constraints immediately before the reference model generates, not only at
    the front of the system prompt.

    Regression guard for the observed bug where the reference advisor (no
    tools) fabricated tool-execution narratives despite the system-prompt
    prohibitions: recency favors the transcript's tool-call blocks sitting
    right before the output position, so the trailing instruction must repeat
    the constraints at that spot.

    Behavior-contract assertions only: substring checks, no full-text equality.
    """
    # Original judgement request preserved verbatim in semantics.
    assert "most intelligent judgement" in _ADVISORY_INSTRUCTION

    # Reminder point 1: the advisor has no tools and no tool results.
    assert "NO tools and NO tool results" in _ADVISORY_INSTRUCTION, \
        "Advisory instruction must restate that the advisor has no tools"

    # Reminder point 2: never emit tool-call blocks / output-shaped text.
    assert "Never emit tool-call" in _ADVISORY_INSTRUCTION, \
        "Advisory instruction must forbid tool-call blocks"

    # Reminder point 3: third-person attribution to the acting agent.
    assert "third person" in _ADVISORY_INSTRUCTION, \
        "Advisory instruction must require third-person attribution"
    assert "the acting agent reported" in _ADVISORY_INSTRUCTION, \
        "Advisory instruction must show the third-person attribution shape"

    # Reminder point 4: honesty when the transcript does not establish a fact.
    assert "unclear from the transcript" in _ADVISORY_INSTRUCTION, \
        "Advisory instruction must require 'unclear from the transcript' honesty"

    # Reminder point 5: plain advisory prose only.
    assert "plain advisory prose" in _ADVISORY_INSTRUCTION, \
        "Advisory instruction must require plain advisory prose"

