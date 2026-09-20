"""Lightweight test doubles so the whole pipeline can run without any model,
embedding service, or network access."""


class FakeMemory:
    def __init__(self):
        self.memory_items = []
        self.reflections = []

    def add(self, observation, action, result=""):
        text = f"Observation: {observation} | Action: {action} | Result: {result}"
        self.memory_items.append(text)
        return text

    def add_reflection(self, reflection):
        self.reflections.append(reflection)
        self.memory_items.append(f"Reflection: {reflection}")

    def recent(self, n=5):
        return self.memory_items[-n:]

    def retrieve(self, query):
        return self.recent(5)

    def count(self):
        return len(self.memory_items)

    def export(self):
        return {
            "items": list(self.memory_items),
            "reflections": list(self.reflections),
        }

    def import_data(self, data):
        self.memory_items.extend(data.get("items", []) or [])
        self.reflections.extend(data.get("reflections", []) or [])


from types import SimpleNamespace


class StreamingLLM:
    """Minimal fake chat model exposing a ``stream`` interface."""

    def __init__(self, chunks=None):
        self.chunks = list(chunks) if chunks else ["A reflection."]
        self.prompts = []

    def stream(self, prompt):
        self.prompts.append(prompt)
        for chunk in self.chunks:
            yield SimpleNamespace(content=chunk)


class ScriptedGraph:
    """A fake LangGraph agent that returns actions from a callback."""

    def __init__(self, actions=None):
        self.actions = actions
        self.states = []

    def invoke(self, state):
        self.states.append(dict(state))
        name = state.get("agent_name", "?")
        turn = state.get("turn", 0)
        actions = self.actions(name, turn) if self.actions else []
        return {
            "observation": state.get("observation", f"observation for {name} at turn {turn}"),
            "actions": actions or [],
            "plan": state.get("plan", f"plan for {name}"),
            "reflections": list(state.get("reflections", [])),
        }


class FakeHuman:
    """Non-blocking stand-in for HumanController used in loop tests."""

    def __init__(self, faction, intent=None, confirm=None, suggest_count=3, timeout=1.0):
        self.faction = faction
        self.suggest_count = suggest_count
        self.timeout = timeout
        self._intent = intent if intent is not None else {"choice": 0}
        self._confirm = confirm if confirm is not None else {"accept": True}
        self.prompts = []
        self.plans = []

    def is_human(self, name):
        return self.faction == name

    def request_intent(self, prompt, timeout=None):
        self.prompts.append(prompt)
        return dict(self._intent)

    def request_confirm(self, plan, timeout=None):
        self.plans.append(plan)
        return dict(self._confirm)


def end_turn_action(name):
    return {"action": "end_turn", "args": {}, "result": f"{name} ended its turn."}


def make_fake_agent_factory(actions=None):
    def factory(world, embeddings, model_config=None, verbose=False, on_event=None, lang="en"):
        agents = {}
        for name in world.agents:
            agents[name] = {
                "graph": ScriptedGraph(actions),
                "memory": FakeMemory(),
                "provider": "ollama",
                "model": "fake-model",
                "llm": None,
            }
        return agents

    return factory
