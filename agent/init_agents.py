import os

from agent.persona import get_persona
from agent.goals import get_goals
from agent.llm_factory import PROVIDERS, build_chat_model
from agent_graph.graph import build_agent_graph
from agent_memory.memory_store import FAISSMemory

DEFAULT_LOCAL_MODELS = {
    "Germany": "qwen3.5:latest",
    "France": "qwen3.5:latest",
    "United Kingdom": "qwen3.5:latest",
    "Soviet Union": "qwen3.5:latest",
}

DEFAULT_API_PROVIDERS = {
    "Germany": "zhipu",
    "France": "deepseek",
    "United Kingdom": "qwen",
    "Soviet Union": "deepseek",
}


# Prefixes used for per-nation overrides in .env, e.g. GERMANY_PROVIDER.
ENV_NATION_PREFIX = {
    "Germany": "GERMANY",
    "France": "FRANCE",
    "United Kingdom": "UK",
    "Soviet Union": "USSR",
}


def _env_value(*names):
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def default_model_config():
    """Default chat model per nation, read from .env.

    ``DEFAULT_PROVIDER`` / ``DEFAULT_MODEL`` set the baseline; per-nation
    ``<GERMANY|FRANCE|UK>_PROVIDER`` / ``_MODEL`` override it. Missing values
    fall back to the local Ollama defaults.
    """
    default_provider = _env_value("DEFAULT_PROVIDER") or "ollama"
    default_model = _env_value("DEFAULT_MODEL")
    config = {}
    for name in DEFAULT_LOCAL_MODELS:
        prefix = ENV_NATION_PREFIX[name]
        nation_provider = _env_value(f"{prefix}_PROVIDER")
        nation_model = _env_value(f"{prefix}_MODEL")
        provider = nation_provider or default_provider

        if nation_model:
            model = nation_model
        elif nation_provider and nation_provider != default_provider:
            # Provider overridden without a model: don't inherit DEFAULT_MODEL,
            # which belongs to the global provider.
            model = None
        else:
            model = default_model

        if not model:
            if provider == "ollama":
                model = DEFAULT_LOCAL_MODELS[name]
            else:
                model = PROVIDERS.get(provider, {}).get("default_model")
        config[name] = {"provider": provider, "model": model}
    return config


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _germany_checks(world, name):
    return {
        "primary": lambda w, n: _clamp(w.city_count("Germany") / 4.0),
        "secondary": lambda w, n: 1.0 if w.war_count("Germany") < 2 else 0.0,
        "tertiary": lambda w, n: _clamp(w.resources_total("Germany") / 250.0),
    }


def _france_checks(world, name):
    return {
        "primary": lambda w, n: 1.0 if w.cities["Paris"].owner == "France" else 0.0,
        "secondary": lambda w, n: 1.0 - _clamp(w.city_count("Germany") / 5.0),
        "tertiary": lambda w, n: _clamp(w.resources_total("France") / 250.0),
    }


def _uk_checks(world, name):
    def domination(w):
        counts = [w.city_count(a) for a in w.agents]
        maximum = max(counts) if counts else 0
        average = sum(counts) / len(counts) if counts else 0
        return maximum, average

    return {
        "primary": lambda w, n: 1.0
        - _clamp((domination(w)[0] - domination(w)[1]) / 4.0),
        "secondary": lambda w, n: _clamp(w.resources_total("United Kingdom") / 250.0),
        "tertiary": lambda w, n: 1.0
        if w.cities["London"].owner == "United Kingdom"
        else 0.0,
    }


def _ussr_checks(world, name):
    return {
        "primary": lambda w, n: _clamp(w.city_count("Soviet Union") / 6.0),
        "secondary": lambda w, n: 0.5
        * (1.0 if w.cities["Moscow"].owner == "Soviet Union" else 0.0)
        + 0.5 * (1.0 if w.cities["Leningrad"].owner == "Soviet Union" else 0.0),
        "tertiary": lambda w, n: _clamp(w.resources_total("Soviet Union") / 300.0),
    }


NATIONS = {
    "Germany": {
        "persona": get_persona(
            brief_description="Expansionist aggressive state",
            strength="Strong domestic support for war",
            weakness="Can't sustain long-term war because of limited resources",
        ),
        "goals": get_goals(
            primary="Control the most cities in Europe",
            secondary="Avoid fighting on two fronts",
            tertiary="Build a strong economy",
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
            primary="Keep Paris under French control",
            secondary="Prevent German expansion",
            tertiary="Accumulate resources",
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
            primary="Maintain the balance of power in Europe",
            secondary="Build a strong economy",
            tertiary="Keep London secure",
        ),
        "checks": _uk_checks,
    },
    "Soviet Union": {
        "persona": get_persona(
            brief_description="Vast collectivist superpower",
            strength="Enormous manpower and resources",
            weakness="Slow to mobilise across a huge frontier",
        ),
        "goals": get_goals(
            primary="Spread Soviet control across Eastern Europe",
            secondary="Keep Moscow and Leningrad secure",
            tertiary="Build a strong economy",
        ),
        "checks": _ussr_checks,
    },
}


def build_all_agents(
    world,
    embeddings,
    model_config=None,
    verbose=False,
    on_event=None,
    lang="en",
):
    config_map = model_config or default_model_config()
    agents = {}
    for name, config in NATIONS.items():
        spec = config_map.get(name) or {
            "provider": "ollama",
            "model": DEFAULT_LOCAL_MODELS[name],
        }
        provider = spec.get("provider", "ollama")
        model = spec.get("model") or PROVIDERS.get(provider, {}).get("default_model")
        llm = build_chat_model(
            provider=provider, model=model, api_key=spec.get("api_key")
        )
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
            lang=lang,
        )
        agents[name] = {
            "graph": graph,
            "memory": memory,
            "provider": provider,
            "model": model,
            "llm": llm,
        }
    return agents
