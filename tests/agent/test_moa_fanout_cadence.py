"""every_n fanout cadence: advisors refresh every Nth tool iteration, and
reference guidance is attached ONLY on the call where the advisors actually
ran (the on-cadence cache-MISS fan-out call).

Redesigned from PR #63448's intent (issue #63393 — advisor fan-out multiplies
turn latency/cost by the tool-iteration count). Off-cadence iterations are
cache HITs: no advisor calls, no display re-emit, and NO re-attach of the last
on-cadence advice — the aggregator's transcript has advanced since that run, so
re-presenting it would go stale. The aggregator acts alone on those calls until
the next on-cadence run injects fresh advice.
"""

from types import SimpleNamespace


def _response(content="done", *, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def _cadence_config(home, fanout="every_n:3"):
    home.mkdir()
    (home / "config.yaml").write_text(
        f"""
moa:
  default_preset: review
  presets:
    review:
      fanout: "{fanout}"
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )


def _install_fake_llm(monkeypatch, ref_runs):
    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            ref_runs.append(kwargs["model"])
            return _response(f"advice #{len(ref_runs)}")
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)


def _iteration_messages(base, iterations):
    """Yield message lists simulating a growing tool loop: the base user turn,
    then one new (assistant tool_call, tool result) pair per iteration."""
    msgs = list(base)
    yield list(msgs)
    for i in range(1, iterations):
        msgs = msgs + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"c{i}", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"},
        ]
        yield list(msgs)


def _flatten_text(messages):
    """All textual content of a message list, for asserting advice presence."""
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text") or "")
    return "\n".join(parts)


def test_every_n_cadence_runs_references_every_nth_iteration(monkeypatch, tmp_path):
    """With every_n:3, references run on iterations 1 and 4 of a 6-iteration
    tool loop (1 on-cadence, then every 3rd), not on all 6."""
    home = tmp_path / ".hermes"
    _cadence_config(home, "every_n:3")
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.moa_loop import MoAChatCompletions

    events = []
    facade = MoAChatCompletions("review", reference_callback=lambda ev, **kw: events.append(ev))
    base = [{"role": "user", "content": "do the thing"}]
    for msgs in _iteration_messages(base, 6):
        facade.create(messages=msgs, tools=[{"type": "function"}])

    # 1 reference model × iterations {1, 4} on-cadence = 2 advisor runs.
    assert len(ref_runs) == 2
    # Display blocks only surface when references actually ran.
    assert events.count("moa.reference") == 2
    assert events.count("moa.aggregating") == 2


def test_every_n_off_cadence_iterations_carry_no_guidance(monkeypatch, tmp_path):
    """Reference guidance rides ONLY on the calls where the references ran
    (every_n:3 over one user turn → runs on calls 1 and 4). Off-cadence calls
    must NOT reuse the last run's advice: their prepared guidance is None and
    no advice text reaches the aggregator messages."""
    home = tmp_path / ".hermes"
    _cadence_config(home, "every_n:3")
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    base = [{"role": "user", "content": "task"}]
    prepared = [
        facade.create(messages=msgs, tools=[], _moa_prepare_only=True)
        for msgs in _iteration_messages(base, 6)
    ]

    # Cadence unchanged: 6 calls, references ran on iterations 1 and 4 only.
    assert len(ref_runs) == 2
    # The on-cadence (cache-MISS) calls carry that run's fresh advice.
    assert prepared[0]["guidance"] and "advice #1" in prepared[0]["guidance"]
    assert prepared[3]["guidance"] and "advice #2" in prepared[3]["guidance"]
    # Off-cadence (cache-HIT) calls carry NO guidance and NO stale advice text.
    for idx in (1, 2, 4, 5):
        assert prepared[idx]["guidance"] is None
        joined = _flatten_text(prepared[idx]["messages"])
        assert "Mixture of Agents reference context" not in joined
        assert "advice #1" not in joined
        assert "advice #2" not in joined


def test_user_turn_guidance_only_on_first_iteration_of_turn(monkeypatch, tmp_path):
    """user_turn fanout: iteration 1 gets the reference guidance; later
    tool-loop iterations of the SAME user turn carry no guidance — the
    aggregator works through the rest of the tool loop alone (per the mode's
    documentation) instead of being re-fed the first run's advice."""
    home = tmp_path / ".hermes"
    _cadence_config(home, "user_turn")
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    base = [{"role": "user", "content": "task"}]
    prepared = [
        facade.create(messages=msgs, tools=[], _moa_prepare_only=True)
        for msgs in _iteration_messages(base, 4)
    ]

    # One advisor run for the whole user turn.
    assert len(ref_runs) == 1
    # Guidance attaches only on the call where the references actually ran; the
    # off-cadence iterations get none (re-presenting it would go stale).
    assert prepared[0]["guidance"] and "advice #1" in prepared[0]["guidance"]
    for idx in (1, 2, 3):
        assert prepared[idx]["guidance"] is None
        joined = _flatten_text(prepared[idx]["messages"])
        assert "Mixture of Agents reference context" not in joined
        assert "advice #1" not in joined


def test_per_iteration_default_unchanged_by_cadence_state(monkeypatch, tmp_path):
    """Default fanout still re-runs references on every state change."""
    home = tmp_path / ".hermes"
    _cadence_config(home, "per_iteration")
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    base = [{"role": "user", "content": "task"}]
    prepared = [
        facade.create(messages=msgs, tools=[], _moa_prepare_only=True)
        for msgs in _iteration_messages(base, 3)
    ]

    assert len(ref_runs) == 3
    # Every state-changing call runs references AND gets guidance.
    assert all(p["guidance"] for p in prepared)
