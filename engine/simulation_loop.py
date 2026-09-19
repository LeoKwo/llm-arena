from langchain_ollama.embeddings import OllamaEmbeddings

from agent.init_agents import (
    DEFAULT_LOCAL_MODELS,
    NATIONS,
    build_all_agents,
)
from world.environment import build_default_world

EMBED_MODEL = "qwen3-embedding:0.6b"
MAX_TURNS = 20


def _format_action(action):
    if not action:
        return "none"
    name = action.get("action", "unknown")
    args = action.get("args", {})
    if args:
        return f"{name}({args})"
    return name


def _print_event(event):
    kind = event.get("type")
    if kind in {"llm_start", "llm_token", "llm_end"}:
        return
    if kind == "turn_start":
        print(f"\n===== Turn {event['turn']} =====", flush=True)
    elif kind == "agent_thinking":
        print(f"  {event['agent']} is thinking...", flush=True)
    elif kind == "agent_action":
        print(f"  {event['agent']} -> {_format_action(event.get('action'))}", flush=True)
        print(f"      {event.get('result')}", flush=True)
    elif kind == "standings":
        standings = ", ".join(
            f"{name}={score:.0f}" for name, score in event.get("rankings", [])
        )
        print(f"  [standings] {standings}", flush=True)
    elif kind == "end":
        print(f"\nSimulation ended. Winner: {event.get('winner')}", flush=True)
    elif kind == "error":
        print(f"\nERROR: {event.get('message')}", flush=True)


def nation_metadata(model_config=None):
    config_map = model_config or {}
    metadata = {}
    for name, config in NATIONS.items():
        spec = config_map.get(name) or {
            "provider": "ollama",
            "model": DEFAULT_LOCAL_MODELS[name],
        }
        metadata[name] = {
            "provider": spec.get("provider", "ollama"),
            "model": spec.get("model") or DEFAULT_LOCAL_MODELS[name],
            "persona": config["persona"],
            "goals": config["goals"],
        }
    return metadata


def run(max_turns=MAX_TURNS, verbose=True, on_event=None, model_config=None, should_stop=None):
    def emit(event):
        if on_event is not None:
            on_event(event)
        if verbose:
            _print_event(event)

    def stop_requested():
        return bool(should_stop and should_stop())

    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        world = build_default_world()
        agents = build_all_agents(
            world,
            embeddings,
            model_config=model_config,
            verbose=False,
            on_event=on_event,
        )
        carry = {name: [] for name in agents}

        emit(
            {
                "type": "init",
                "nations": nation_metadata(model_config),
                "world": world.snapshot(),
                "max_turns": max_turns,
            }
        )

        for turn in range(max_turns):
            if stop_requested():
                break
            world.turn = turn
            emit({"type": "turn_start", "turn": turn})

            for name, bundle in agents.items():
                if not world.agents[name]["alive"]:
                    continue
                if stop_requested():
                    emit(
                        {
                            "type": "interrupted",
                            "world": world.snapshot(),
                            "rankings": world.rankings(),
                        }
                    )
                    return None
                emit({"type": "agent_thinking", "agent": name, "turn": turn})
                previous_reflections = len(carry[name])
                state = {
                    "agent_name": name,
                    "turn": turn,
                    "reflections": carry[name],
                    "messages": [],
                    "llm_calls": 0,
                }
                result = bundle["graph"].invoke(state)
                reflections = result.get("reflections", carry[name])
                carry[name] = reflections
                action = result.get("action")
                outcome = result.get("result")
                bundle["memory"].add(
                    result.get("observation", ""), action, outcome or ""
                )
                emit(
                    {
                        "type": "agent_action",
                        "turn": turn,
                        "agent": name,
                        "action": action,
                        "result": outcome,
                        "plan": result.get("plan", ""),
                        "observation": result.get("observation", ""),
                        "memory_context": result.get("memory_context", []),
                        "reflections": reflections[previous_reflections:],
                    }
                )
                emit({"type": "world_state", "world": world.snapshot()})

            world.advance()
            emit({"type": "world_state", "world": world.snapshot()})
            emit({"type": "standings", "rankings": world.rankings()})

            ended, winner = world.check_end_conditions()
            if ended:
                emit(
                    {
                        "type": "end",
                        "winner": winner,
                        "world": world.snapshot(),
                        "rankings": world.rankings(),
                    }
                )
                return winner

        if stop_requested():
            emit(
                {
                    "type": "interrupted",
                    "world": world.snapshot(),
                    "rankings": world.rankings(),
                }
            )
            return None

        winner = world.rankings()[0][0]
        emit(
            {
                "type": "end",
                "winner": winner,
                "world": world.snapshot(),
                "rankings": world.rankings(),
            }
        )
        return winner
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        raise


if __name__ == "__main__":
    run()
