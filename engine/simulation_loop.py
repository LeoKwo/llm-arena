from agent.init_agents import (
    DEFAULT_LOCAL_MODELS,
    NATIONS,
    build_all_agents,
)
from agent.llm_factory import build_embeddings
from engine.report import generate_report
from engine.savegame import (
    build_save_payload,
    deserialize_world,
    load_agents_memory,
)
from world.environment import build_default_world

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
    elif kind == "reporting":
        print("\nWriting the end-of-game report...", flush=True)
    elif kind in {"end", "interrupted"}:
        winner = event.get("winner")
        prefix = "Simulation ended" if kind == "end" else "Simulation interrupted"
        print(f"\n{prefix}. Winner: {winner}", flush=True)
        _print_report(event.get("report"))
    elif kind == "error":
        print(f"\nERROR: {event.get('message')}", flush=True)


def _print_report(report):
    if not report:
        return
    print("\n" + "=" * 60, flush=True)
    print(report.get("headline", ""), flush=True)
    print(report.get("dateline", ""), flush=True)
    print("-" * 60, flush=True)
    if report.get("lead"):
        print(report["lead"], flush=True)
    for paragraph in report.get("paragraphs", []):
        print(paragraph, flush=True)
    timeline = report.get("timeline", [])
    if timeline:
        print("-" * 60, flush=True)
        for entry in timeline:
            print(entry.get("text", ""), flush=True)
    if report.get("outcome"):
        print("-" * 60, flush=True)
        print(report["outcome"], flush=True)
    print("=" * 60, flush=True)


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


def _resolve_max_turns(max_turns, resume):
    if max_turns is not None:
        return int(max_turns)
    if resume and resume.get("max_turns"):
        return int(resume["max_turns"])
    return MAX_TURNS


def _resolve_model_config(model_config, resume):
    if model_config:
        return model_config
    if not resume:
        return model_config
    saved = resume.get("models") or {}
    resolved = {
        name: {"provider": spec.get("provider"), "model": spec.get("model")}
        for name, spec in saved.items()
        if spec.get("provider")
    }
    return resolved or model_config


def _report_llm(agents, winner):
    """Pick a model to write the news report (winner first, then any)."""
    order = []
    if winner and winner in agents:
        order.append(winner)
    order.extend(name for name in agents if name not in order)
    for name in order:
        bundle = agents.get(name, {})
        llm = bundle.get("llm")
        if llm is not None:
            label = f"{bundle.get('provider') or '?'}/{bundle.get('model') or '?'}"
            return llm, label
    return None, None


def run(
    max_turns=MAX_TURNS,
    verbose=True,
    on_event=None,
    model_config=None,
    should_stop=None,
    lang="en",
    resume=None,
    on_checkpoint=None,
    embeddings=None,
    world_factory=None,
    agent_factory=None,
    report_llm=None,
    report_label=None,
    report_enabled=True,
):
    def emit(event):
        if on_event is not None:
            on_event(event)
        if verbose:
            _print_event(event)

    def stop_requested():
        return bool(should_stop and should_stop())

    world_factory = world_factory or build_default_world
    agent_factory = agent_factory or build_all_agents
    max_turns = _resolve_max_turns(max_turns, resume)
    model_config = _resolve_model_config(model_config, resume)

    def end_event(kind, winner):
        if kind == "end":
            emit({"type": "reporting", "winner": winner})
            if report_llm is not None:
                llm, label = report_llm, report_label
            elif report_enabled:
                llm, label = _report_llm(agents, winner)
            else:
                llm, label = None, None
            report = generate_report(world, winner, llm=llm, lang=lang)
            if report.get("source") == "llm" and label:
                report["model"] = label
        else:
            report = generate_report(world, None, llm=None, lang=lang)
        return {
            "type": kind,
            "winner": winner,
            "world": world.snapshot(),
            "rankings": world.rankings(),
            "timeline": world.timeline(),
            "report": report,
        }

    try:
        if resume:
            world = deserialize_world(resume["world"])
        else:
            world = world_factory()
        if resume and world.turn > max_turns:
            # Keep an explicit (possibly smaller) request sane: never truncate
            # past the point the save reached.
            max_turns = world.turn
        world.max_turns = max_turns

        if embeddings is None:
            embeddings = build_embeddings()

        agents = agent_factory(
            world,
            embeddings,
            model_config=model_config,
            verbose=False,
            on_event=on_event,
            lang=lang,
        )
        if resume:
            load_agents_memory(agents, resume.get("agents", {}))

        carry = {
            name: list(bundle.get("memory").reflections)
            if bundle.get("memory") is not None
            else []
            for name, bundle in agents.items()
        }

        start_turn = world.turn if resume else 0

        init_event = {
            "type": "init",
            "nations": nation_metadata(model_config),
            "world": world.snapshot(),
            "max_turns": max_turns,
            "resumed": bool(resume),
        }
        if resume:
            init_event["timeline"] = world.timeline()
        emit(init_event)

        for turn in range(start_turn, max_turns):
            if stop_requested():
                break
            world.turn = turn
            world.start_round()
            for note in world.take_broadcasts():
                emit({"type": "broadcast", "event": note})
            emit({"type": "world_state", "world": world.snapshot()})
            emit({"type": "turn_start", "turn": turn})

            order = list(agents.keys())
            if order:
                offset = turn % len(order)
                order = order[offset:] + order[:offset]
            for name in order:
                bundle = agents[name]
                if not world.agents[name]["alive"]:
                    continue
                if stop_requested():
                    emit(end_event("interrupted", None))
                    return None
                emit({"type": "agent_thinking", "agent": name, "turn": turn})
                world.begin_turn(name)
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
                observation = result.get("observation", "")
                actions = result.get("actions") or [result.get("action")]
                for entry in actions:
                    if not entry:
                        continue
                    outcome = entry.get("result", "") if isinstance(entry, dict) else ""
                    bundle["memory"].add(observation, entry, outcome or "")
                    emit(
                        {
                            "type": "agent_action",
                            "turn": turn,
                            "agent": name,
                            "action": entry,
                            "result": outcome,
                            "plan": result.get("plan", ""),
                            "reflections": reflections[previous_reflections:],
                        }
                    )
                world.end_faction_turn(name)
                for note in world.take_broadcasts():
                    emit({"type": "broadcast", "event": note})
                emit({"type": "world_state", "world": world.snapshot()})

            world.advance()
            emit({"type": "world_state", "world": world.snapshot()})
            emit({"type": "standings", "rankings": world.rankings()})

            ended, winner = world.check_end_conditions()
            if on_checkpoint is not None:
                on_checkpoint(
                    build_save_payload(
                        world,
                        agents,
                        {"lang": lang, "ended": ended, "winner": winner},
                    )
                )
            if ended:
                emit(end_event("end", winner))
                return winner

        if stop_requested():
            emit(end_event("interrupted", None))
            return None

        winner = world.rankings()[0][0]
        emit(end_event("end", winner))
        return winner
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        raise


if __name__ == "__main__":
    run()
