from engine.simulation_loop import run
from tests.fakes import end_turn_action, make_fake_agent_factory


def _factory():
    def actions(name, turn):
        return [end_turn_action(name)]

    return make_fake_agent_factory(actions)


def test_run_emits_full_lifecycle():
    events = []
    checkpoints = []
    run(
        max_turns=3,
        verbose=False,
        on_event=events.append,
        embeddings=object(),
        agent_factory=_factory(),
        lang="en",
        on_checkpoint=checkpoints.append,
    )

    types = [e["type"] for e in events]
    assert types[0] == "init"
    assert types.count("turn_start") == 3
    assert types[-1] == "end"

    end = next(e for e in events if e["type"] == "end")
    assert end["winner"]
    assert end["rankings"]
    # One score point per completed turn.
    assert [p["turn"] for p in end["timeline"]["score_history"]] == [0, 1, 2]
    assert end["report"]["source"] == "template"
    assert end["report"]["outcome"]


def test_checkpoints_are_built_at_turn_boundaries():
    checkpoints = []
    run(
        max_turns=2,
        verbose=False,
        embeddings=object(),
        agent_factory=_factory(),
        on_checkpoint=checkpoints.append,
    )
    assert len(checkpoints) == 2
    assert checkpoints[0]["world"]["turn"] == 1
    assert checkpoints[1]["world"]["turn"] == 2
    assert checkpoints[1]["ended"] is False
    assert "agents" in checkpoints[1]


def test_dedicated_report_model_is_used():
    from types import SimpleNamespace

    class FakeLLM:
        def invoke(self, prompt):
            return SimpleNamespace(
                content="Headline: Dedicated dispatch\nBody:\nOne paragraph here."
            )

    events = []
    run(
        max_turns=1,
        verbose=False,
        on_event=events.append,
        embeddings=object(),
        agent_factory=_factory(),
        report_llm=FakeLLM(),
        report_label="test/model",
    )
    end = next(e for e in events if e["type"] == "end")
    assert end["report"]["source"] == "llm"
    assert end["report"]["headline"] == "Dedicated dispatch"
    assert end["report"]["model"] == "test/model"


def test_report_can_be_disabled():
    from types import SimpleNamespace

    class FakeLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content="Headline: nope\nBody:\nShould not run.")

    base = _factory()

    def factory(*args, **kwargs):
        agents = base(*args, **kwargs)
        for bundle in agents.values():
            bundle["llm"] = FakeLLM()
        return agents

    events = []
    run(
        max_turns=1,
        verbose=False,
        on_event=events.append,
        embeddings=object(),
        agent_factory=factory,
        report_enabled=False,
    )
    end = next(e for e in events if e["type"] == "end")
    assert end["report"]["source"] == "template"


def test_stop_short_circuits_and_reports_interruption():
    events = []
    state = {"stopped": False}

    def should_stop():
        # Stop after the first turn begins.
        return state["stopped"]

    def on_event(event):
        events.append(event)
        if event["type"] == "turn_start":
            state["stopped"] = True

    run(
        max_turns=5,
        verbose=False,
        on_event=on_event,
        embeddings=object(),
        agent_factory=_factory(),
        should_stop=should_stop,
    )
    interrupted = [e for e in events if e["type"] == "interrupted"]
    assert interrupted
    assert interrupted[-1]["report"]["source"] == "template"
