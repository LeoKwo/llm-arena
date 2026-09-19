from agent.persona import get_persona
from agent.goals import get_goals
from agent.llm_factory import PROVIDERS, build_chat_model
from agent_graph.graph import build_agent_graph
from agent_memory.memory_store import FAISSMemory

DEFAULT_LOCAL_MODELS = {
    "Germany": "qwen3.5:latest",
    "France": "qwen3.5:latest",
    "United Kingdom": "qwen3.5:latest",
}

DEFAULT_API_PROVIDERS = {
    "Germany": "zhipu",
    "France": "deepseek",
    "United Kingdom": "qwen",
}


def default_model_config():
    return {
        name: {"provider": "ollama", "model": DEFAULT_LOCAL_MODELS[name]}
        for name in DEFAULT_LOCAL_MODELS
    }


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _germany_checks(world, name):
    return {
        "primary": lambda w, n: _clamp(
            (len(w.territories_of("Germany")) - w.initial_territories["Germany"]) / 2.0
        ),
        "secondary": lambda w, n: 1.0 if w.war_count("Germany") < 2 else 0.0,
        "tertiary": lambda w, n: _clamp(w.resources("Germany") / 100.0),
    }


def _france_checks(world, name):
    return {
        "primary": lambda w, n: 1.0 if w.locations.get("Paris") == "France" else 0.0,
        "secondary": lambda w, n: 1.0
        - _clamp(
            (len(w.territories_of("Germany")) - w.initial_territories["Germany"]) / 2.0
        ),
        "tertiary": lambda w, n: _clamp(
            (w.relation("France", "United Kingdom") + 100.0) / 200.0
        ),
    }


def _uk_checks(world, name):
    def domination_ratio(w):
        counts = [len(w.territories_of(a)) for a in w.agents]
        maximum = max(counts) if counts else 0
        average = sum(counts) / len(counts) if counts else 0
        return maximum, average

    return {
        "primary": lambda w, n: 1.0 - _clamp((domination_ratio(w)[0] - domination_ratio(w)[1]) / 3.0),
        "secondary": lambda w, n: 1.0 - _clamp(domination_ratio(w)[0] / 4.0),
        "tertiary": lambda w, n: _clamp(w.resources("United Kingdom") / 90.0),
    }


NATIONS = {
    "Germany": {
        "persona": get_persona(
            brief_description="Expansionist aggressive state",
            strength="Strong domestic support for war",
            weakness="Can't sustain long-term war because of limited resources",
        ),
        "goals": get_goals(
            primary="Expand territory",
            secondary="Avoid multi front war",
            tertiary="Maintain strong economy",
        ),
        "checks": _germany_checks,
    },
    "France": {
        "persona": get_persona(
            brief_description="Cautious defensive European power",
            strength="Strong defensive military and fortified borders",
            weakness="Political divisions and reluctance to start offensive wars",
        ),
        "goals": get_goals(
            primary="Preserve national sovereignty",
            secondary="Prevent German expansion",
            tertiary="Maintain alliance stability",
        ),
        "checks": _france_checks,
    },
    "United Kingdom": {
        "persona": get_persona(
            brief_description="Island naval power focused on balance of power in Europe",
            strength="Powerful navy and strong economy",
            weakness="Limited willingness for large land wars in Europe",
        ),
        "goals": get_goals(
            primary="Maintain balance of power in Europe",
            secondary="Prevent domination of the continent by any single power",
            tertiary="Protect global trade and colonies",
        ),
        "checks": _uk_checks,
    },
}


def build_all_agents(world, embeddings, model_config=None, verbose=False, on_event=None):
    config_map = model_config or default_model_config()
    agents = {}
    for name, config in NATIONS.items():
        spec = config_map.get(name) or {
            "provider": "ollama",
            "model": DEFAULT_LOCAL_MODELS[name],
        }
        provider = spec.get("provider", "ollama")
        model = spec.get("model") or PROVIDERS.get(provider, {}).get("default_model")
        llm = build_chat_model(provider=provider, model=model)
        memory = FAISSMemory(embeddings=embeddings, max_results=5)
        world.register_goals(name, config["checks"](world, name))
        graph = build_agent_graph(
            name=name,
            persona=config["persona"],
            goals=config["goals"],
            llm=llm,
            world=world,
            memory=memory,
            verbose=verbose,
            on_event=on_event,
        )
        agents[name] = {
            "graph": graph,
            "memory": memory,
            "provider": provider,
            "model": model,
        }
    return agents
