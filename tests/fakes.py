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


class ScriptedGraph:
    """A fake LangGraph agent that returns actions from a callback."""

    def __init__(self, actions=None):
        self.actions = actions

    def invoke(self, state):
        name = state.get("agent_name", "?")
        turn = state.get("turn", 0)
        actions = self.actions(name, turn) if self.actions else []
        return {
            "observation": f"observation for {name} at turn {turn}",
            "actions": actions or [],
            "plan": f"plan for {name}",
            "reflections": list(state.get("reflections", [])),
        }


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
