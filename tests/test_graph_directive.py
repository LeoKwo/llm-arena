"""Exercise the real LangGraph agent with the human "directive" entry point."""

from langchain_core.messages import AIMessage

from agent_graph.graph import build_agent_graph
from tests.fakes import FakeMemory


class FakeToolLLM:
    """Minimal chat model that supports bind_tools + streaming."""

    def __init__(self, text="Executed the order."):
        self.text = text
        self.bound = False

    def bind_tools(self, tools):
        self.bound = True
        return self

    def stream(self, messages):
        yield AIMessage(content=self.text)


def _graph(world, events):
    return build_agent_graph(
        name="Germany",
        persona="test persona",
        goals={"primary": "control cities"},
        llm=FakeToolLLM(),
        world=world,
        memory=FakeMemory(),
        on_event=events.append,
        lang="en",
    )


def test_directive_entry_skips_observe_and_plan(world):
    events = []
    graph = _graph(world, events)

    result = graph.invoke(
        {
            "agent_name": "Germany",
            "turn": 0,
            "reflections": [],
            "messages": [],
            "llm_calls": 0,
            "observation": "current situation",
            "memory_context": [],
            "plan": "COMMANDER'S ORDER: attack Paris",
            "directive": True,
        }
    )

    assert result["plan"] == "COMMANDER'S ORDER: attack Paris"
    assert result.get("actions") == []
    nodes = [event.get("node") for event in events]
    assert "act" in nodes
    # The human path must not re-run planning or observation.
    assert "plan" not in nodes


def test_default_entry_still_runs_plan_and_act(world):
    events = []
    graph = _graph(world, events)

    result = graph.invoke(
        {
            "agent_name": "Germany",
            "turn": 0,
            "reflections": [],
            "messages": [],
            "llm_calls": 0,
        }
    )

    nodes = [event.get("node") for event in events]
    assert "plan" in nodes
    assert "act" in nodes
    assert result.get("observation")
