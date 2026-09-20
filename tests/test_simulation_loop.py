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


def test_reflection_is_post_turn_and_uses_current_context():
    from tests.fakes import StreamingLLM

    def actions(name, turn):
        return [end_turn_action(name) for _ in range(3)]

    llm = StreamingLLM(["I reflected ", "on turn three."])
    base = make_fake_agent_factory(actions)
    captured = {}

    def factory(*args, **kwargs):
        agents = base(*args, **kwargs)
        for bundle in agents.values():
            bundle["llm"] = llm
        captured.update(agents)
        return agents

    events = []
    run(
        max_turns=3,
        verbose=False,
        on_event=events.append,
        embeddings=object(),
        agent_factory=factory,
    )

    reflections = [
        e for e in events
        if e["type"] == "llm_start" and e.get("node") == "reflect"
    ]
    # Four factions, each reflects once at the end of turn index 2.
    assert len(reflections) == 4
    assert all(e.get("turn") == 2 for e in reflections)
    # Reflection appears after that faction's actions, not before the turn.
    for event in reflections:
        earlier = events[: events.index(event)]
        assert any(
            e["type"] == "agent_action" and e["agent"] == event["agent"]
            for e in earlier
        )
    # The prompt carries fresh context (current situation + this turn's work).
    assert llm.prompts
    assert "Current situation (after turn 3)" in llm.prompts[0]
    assert "What you did this turn" in llm.prompts[0]
    # The reflection was stored in memory for later planning.
    assert all(captured[name]["memory"].reflections for name in captured)


def test_no_reflection_before_the_cadence():
    from tests.fakes import StreamingLLM

    llm = StreamingLLM()
    base = make_fake_agent_factory(lambda name, turn: [end_turn_action(name)])

    def factory(*args, **kwargs):
        agents = base(*args, **kwargs)
        for bundle in agents.values():
            bundle["llm"] = llm
        return agents

    events = []
    run(
        max_turns=2,
        verbose=False,
        on_event=events.append,
        embeddings=object(),
        agent_factory=factory,
    )
    assert not [e for e in events if e["type"] == "llm_start" and e.get("node") == "reflect"]


def test_human_turn_uses_directive_and_faction_model():
    from tests.fakes import FakeHuman

    human = FakeHuman("Germany", intent={"choice": 0}, confirm={"accept": True})
    base = _factory()
    created = {}

    def factory(*args, **kwargs):
        agents = base(*args, **kwargs)
        created.update(agents)
        return agents

    events = []
    run(
        max_turns=1,
        verbose=False,
        on_event=events.append,
        embeddings=object(),
        agent_factory=factory,
        human=human,
    )

    types = [e["type"] for e in events]
    assert "human_phase" in types
    assert "human_action" in types
    # The human was offered suggestions and asked to confirm a plan.
    assert human.prompts and human.prompts[0]["suggestions"]
    assert human.plans and "objective" in human.plans[0]["plan"]

    # Germany's turn executed via the injected directive; others did not.
    germany_states = created["Germany"]["graph"].states
    assert germany_states[-1]["directive"] is True
    assert "COMMANDER'S ORDER" in germany_states[-1]["plan"]
    for name in ("France", "United Kingdom", "Soviet Union"):
        assert created[name]["graph"].states[-1].get("directive") is None

    human_actions = [e for e in events if e["type"] == "human_action"]
    assert human_actions and human_actions[0]["agent"] == "Germany"


def test_human_autopilot_falls_back_to_the_llm():
    from tests.fakes import FakeHuman

    human = FakeHuman("Germany", intent={"kind": "autopilot"})
    base = _factory()
    created = {}

    def factory(*args, **kwargs):
        agents = base(*args, **kwargs)
        created.update(agents)
        return agents

    events = []
    run(
        max_turns=1,
        verbose=False,
        on_event=events.append,
        embeddings=object(),
        agent_factory=factory,
        human=human,
    )

    assert any(e["type"] == "human_autopilot" for e in events)
    assert human.prompts and not human.plans
    # Fell back to the normal (non-directive) graph path.
    assert created["Germany"]["graph"].states[-1].get("directive") is None


def test_human_confirm_modify_then_accept():
    from tests.fakes import FakeHuman

    class ModifyingHuman(FakeHuman):
        def __init__(self):
            super().__init__("Germany", intent={"text": "attack Paris"})
            self._confirmations = [
                {"accept": False, "text": "actually defend the capital"},
                {"accept": True},
            ]

        def request_confirm(self, plan, timeout=None):
            self.plans.append(plan)
            return self._confirmations.pop(0) if self._confirmations else {"accept": True}

    human = ModifyingHuman()
    base = _factory()
    created = {}

    def factory(*args, **kwargs):
        agents = base(*args, **kwargs)
        created.update(agents)
        return agents

    events = []
    run(
        max_turns=1,
        verbose=False,
        on_event=events.append,
        embeddings=object(),
        agent_factory=factory,
        human=human,
    )
    # Re-planned once after the modification, then executed the modified order.
    assert len(human.plans) == 2
    assert human.plans[1]["plan"]["objective"] == "actually defend the capital"
    assert human.plans[1]["intent"] == "actually defend the capital"
    human_actions = [e for e in events if e["type"] == "human_action"]
    assert human_actions and human_actions[0]["intent"] == "actually defend the capital"


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
