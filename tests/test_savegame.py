import json

import pytest

from engine import savegame
from tests.fakes import FakeMemory


def _prepared_world(world):
    world.max_turns = 12
    world.turn = 3
    world.spawn_unit("Germany", "Berlin", 40, "test")
    world.relations[("France", "Germany")] = -30.0
    world._broadcast("city_captured", actor="Germany", city="Paris", previous="France")
    world.advance()
    return world


def test_world_roundtrip_is_lossless(world):
    world = _prepared_world(world)
    data = savegame.serialize_world(world)
    restored = savegame.deserialize_world(data)
    assert savegame.serialize_world(restored) == data


def test_roundtrip_preserves_units_and_relations(world):
    world = _prepared_world(world)
    restored = savegame.deserialize_world(savegame.serialize_world(world))
    assert restored.turn == world.turn
    assert restored.max_turns == world.max_turns
    assert set(restored.units) == set(world.units)
    original = world.units["U1"]
    copy = restored.units["U1"]
    assert (copy.tile.q, copy.tile.r) == (original.tile.q, original.tile.r)
    assert copy.resources == original.resources
    assert copy.ap == original.ap
    assert restored.relations == world.relations
    assert len(restored.event_log) == len(world.event_log)
    assert restored.event_log[0]["turn"] == world.event_log[0]["turn"]


def test_save_load_list_delete(world, save_dir):
    world = _prepared_world(world)
    agents = {"Germany": {"provider": "ollama", "model": "x", "memory": FakeMemory()}}
    agents["Germany"]["memory"].add("obs", {"action": "end_turn"}, "ok")

    meta = savegame.save_game(
        world, agents, {"name": "my game", "lang": "zh"}, save_dir=save_dir
    )
    assert meta["turn"] == world.turn
    assert meta["lang"] == "zh"

    listed = savegame.list_saves(save_dir=save_dir)
    assert [s["id"] for s in listed] == [meta["id"]]

    payload = savegame.load_game(meta["id"], save_dir=save_dir)
    assert payload["name"] == "my game"
    assert payload["agents"]["Germany"]["memory"]["items"]

    assert savegame.delete_save(meta["id"], save_dir=save_dir) is True
    assert savegame.list_saves(save_dir=save_dir) == []
    assert savegame.delete_save("missing", save_dir=save_dir) is False


def test_write_payload_overwrites_autosave(world, save_dir):
    world = _prepared_world(world)
    payload = savegame.build_save_payload(world, {}, {"name": "autosave"})
    first = savegame.write_save_payload(payload, save_id="autosave", save_dir=save_dir)
    second = savegame.write_save_payload(payload, save_id="autosave", save_dir=save_dir)
    assert first["id"] == second["id"] == "autosave"
    assert len(list(save_dir.glob("*.json"))) == 1
    assert not list(save_dir.glob("*.tmp"))


def test_corrupted_and_invalid_saves(world, save_dir):
    (save_dir / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        savegame.load_game("broken", save_dir=save_dir)

    (save_dir / "other.json").write_text(json.dumps({"hello": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        savegame.load_game("other", save_dir=save_dir)

    with pytest.raises(ValueError):
        savegame.save_path("bad/../id", save_dir=save_dir)

    with pytest.raises(FileNotFoundError):
        savegame.load_game("does-not-exist", save_dir=save_dir)

    # list_saves skips garbage instead of raising.
    assert savegame.list_saves(save_dir=save_dir) == []
