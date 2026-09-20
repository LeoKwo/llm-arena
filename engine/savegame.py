"""Save / load support for LLM Arena games.

A save file is a JSON document capturing everything needed to resume a game:
the physical world (tiles, cities, units, standings, history) plus each
faction's memory. It deliberately does **not** store API keys or the LangGraph
objects (graphs are rebuilt on load); ``goal_checks`` are re-registered by
``build_all_agents`` because they are functions and cannot be serialised.

Saves are always taken at a turn boundary (``world.turn`` is the next turn to
play), so resuming simply continues the loop from ``world.turn``.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from world.environment import MAX_AP, Tile, Unit, World

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAVE_DIR = PROJECT_ROOT / "saves"
SAVE_VERSION = 1

_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


# --------------------------------------------------------------------- world
def serialize_world(world: World) -> dict:
    return {
        "turn": world.turn,
        "max_turns": world.max_turns,
        "tiles": [
            {
                "q": tile.q,
                "r": tile.r,
                "resources": tile.resources,
                "owner": tile.owner,
                "city_name": tile.city_name,
            }
            for tile in world.tiles.values()
        ],
        "agents": {
            name: {"home": info.get("home"), "alive": info.get("alive", True)}
            for name, info in world.agents.items()
        },
        "units": [
            {
                "id": unit.id,
                "owner": unit.owner,
                "q": unit.tile.q,
                "r": unit.tile.r,
                "resources": unit.resources,
                "ap": unit.ap,
                "acted": unit.acted,
            }
            for unit in world.units.values()
        ],
        "relations": {
            f"{a}|{b}": value for (a, b), value in world.relations.items()
        },
        "eliminated": list(world.eliminated),
        "spawns_this_turn": dict(world.spawns_this_turn),
        "capital_hold": dict(world.capital_hold),
        "last_score_delta": dict(world.last_score_delta),
        "score_mark": dict(world._score_mark),
        "unit_seq": world._unit_seq,
        "history": [dict(entry) for entry in world.history],
        "event_log": [dict(event) for event in world.event_log],
        "score_history": [dict(point) for point in world.score_history],
    }


def deserialize_world(data: dict) -> World:
    tiles: dict[tuple[int, int], Tile] = {}
    for raw in data.get("tiles", []):
        tiles[(raw["q"], raw["r"])] = Tile(
            raw["q"],
            raw["r"],
            raw["resources"],
            raw.get("owner"),
            raw.get("city_name"),
        )
    cities = {tile.city_name: tile for tile in tiles.values() if tile.city_name}

    agent_data = data.get("agents", {})
    agents = {name: {"home": info.get("home")} for name, info in agent_data.items()}
    world = World(tiles=tiles, cities=cities, agents=agents)
    world.turn = data.get("turn", 0)
    world.max_turns = data.get("max_turns")
    for name, info in agent_data.items():
        world.agents[name]["alive"] = info.get("alive", True)

    world._unit_seq = data.get("unit_seq", 0)
    for raw in data.get("units", []):
        tile = tiles[(raw["q"], raw["r"])]
        unit = Unit(
            raw["id"],
            raw["owner"],
            tile,
            raw["resources"],
            ap=raw.get("ap", MAX_AP),
        )
        unit.acted = bool(raw.get("acted", False))
        world.units[unit.id] = unit

    world.relations = {
        tuple(key.split("|", 1)): value
        for key, value in data.get("relations", {}).items()
    }
    world.eliminated = list(data.get("eliminated", []))
    world.spawns_this_turn = {name: 0 for name in world.agents}
    world.spawns_this_turn.update(data.get("spawns_this_turn", {}))
    world.capital_hold = dict(data.get("capital_hold", {}))
    world.last_score_delta = {name: 0.0 for name in world.agents}
    world.last_score_delta.update(data.get("last_score_delta", {}))
    world._score_mark = {name: 0.0 for name in world.agents}
    world._score_mark.update(data.get("score_mark", {}))
    world.history = [dict(entry) for entry in data.get("history", [])]
    world.event_log = [dict(event) for event in data.get("event_log", [])]
    world.score_history = [dict(point) for point in data.get("score_history", [])]
    # goal_checks are (re)registered by build_all_agents after loading.
    return world


# -------------------------------------------------------------------- agents
def serialize_agents(agents: dict) -> dict:
    out = {}
    for name, bundle in agents.items():
        memory = bundle.get("memory")
        out[name] = {
            "provider": bundle.get("provider"),
            "model": bundle.get("model"),
            "memory": memory.export() if memory is not None else {"items": [], "reflections": []},
        }
    return out


def load_agents_memory(agents: dict, data: dict) -> None:
    for name, payload in (data or {}).items():
        bundle = agents.get(name)
        if bundle is None:
            continue
        memory = bundle.get("memory")
        if memory is not None:
            memory.import_data(payload.get("memory", {}))


# --------------------------------------------------------------- save files
def default_save_dir() -> Path:
    return DEFAULT_SAVE_DIR


def _resolve_dir(save_dir=None) -> Path:
    path = Path(save_dir) if save_dir else default_save_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip()).strip("-")
    return slug[:48]


def new_save_id(name: Optional[str] = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = _slug(name) if name else ""
    return f"{stamp}-{slug}" if slug else stamp


def save_path(save_id: str, save_dir=None) -> Path:
    if not _ID_RE.match(save_id or ""):
        raise ValueError(f"Invalid save id '{save_id}'.")
    return _resolve_dir(save_dir) / f"{save_id}.json"


def build_save_payload(world: World, agents: dict, meta: Optional[dict] = None) -> dict:
    meta = meta or {}
    models = {
        name: {
            "provider": bundle.get("provider"),
            "model": bundle.get("model"),
        }
        for name, bundle in agents.items()
    }
    return {
        "version": SAVE_VERSION,
        "created": time.time(),
        "name": meta.get("name") or "autosave",
        "lang": meta.get("lang", "en"),
        "ended": bool(meta.get("ended", False)),
        "winner": meta.get("winner"),
        "human_faction": meta.get("human_faction"),
        "max_turns": world.max_turns,
        "next_turn": world.turn,
        "models": models,
        "world": serialize_world(world),
        "agents": serialize_agents(agents),
    }


def write_save_payload(
    payload: dict, save_id: Optional[str] = None, save_dir=None
) -> dict:
    """Atomically write an already-built payload (used by checkpoints)."""
    save_id = save_id or new_save_id(payload.get("name"))
    if not _ID_RE.match(save_id or ""):
        raise ValueError(f"Invalid save id '{save_id}'.")
    data = dict(payload)
    data["id"] = save_id
    path = _resolve_dir(save_dir) / f"{save_id}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    os.replace(tmp, path)
    return _metadata_from_payload(data, path)


def save_game(
    world: World,
    agents: dict,
    meta: Optional[dict] = None,
    save_dir=None,
    save_id: Optional[str] = None,
) -> dict:
    """Persist a game atomically. Returns the save metadata (with ``path``)."""
    meta = dict(meta or {})
    payload = build_save_payload(world, agents, meta)
    payload["name"] = meta.get("name") or payload.get("name") or "autosave"
    return write_save_payload(payload, save_id, save_dir)



def _metadata_from_payload(payload: dict, path: Path) -> dict:
    world = payload.get("world", {})
    return {
        "id": payload.get("id") or path.stem,
        "name": payload.get("name"),
        "created": payload.get("created"),
        "lang": payload.get("lang", "en"),
        "ended": bool(payload.get("ended", False)),
        "winner": payload.get("winner"),
        "human_faction": payload.get("human_faction"),
        "turn": world.get("turn", 0),
        "max_turns": payload.get("max_turns"),
        "models": payload.get("models", {}),
        "path": str(path),
    }


def load_game(save_id_or_path: str, save_dir=None) -> dict:
    """Load a save by id or by explicit path. Returns the full payload."""
    candidate = Path(save_id_or_path)
    if candidate.suffix == ".json" or candidate.is_absolute():
        path = candidate
    else:
        path = save_path(save_id_or_path, save_dir)
    if not path.exists():
        raise FileNotFoundError(f"Save '{save_id_or_path}' not found.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Save file '{path.name}' is corrupted: {exc}") from exc
    if not isinstance(payload, dict) or "world" not in payload:
        raise ValueError(f"Save file '{path.name}' is not a valid LLM Arena save.")
    if payload.get("version", SAVE_VERSION) > SAVE_VERSION:
        raise ValueError(
            f"Save version {payload['version']} is newer than supported "
            f"({SAVE_VERSION})."
        )
    if not payload.get("id"):
        payload["id"] = path.stem
    return payload


def list_saves(save_dir=None) -> list[dict]:
    directory = _resolve_dir(save_dir)
    saves = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "world" not in payload:
                continue
            payload.setdefault("id", path.stem)
            saves.append(_metadata_from_payload(payload, path))
        except Exception:
            continue
    saves.sort(key=lambda item: item.get("created") or 0, reverse=True)
    return saves


def delete_save(save_id: str, save_dir=None) -> bool:
    path = save_path(save_id, save_dir)
    if path.exists():
        path.unlink()
        return True
    return False
