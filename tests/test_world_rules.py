from world.environment import (
    CAPITAL_ANNEX_PROTECT_TURNS,
    CITY_SCORE,
    GARRISON_DEFENSE_BONUS,
    TILE_SCORE,
    Unit,
)


def test_base_score_formula(world):
    # Germany starts with only Berlin (100 resources, 1 city, 0 controlled tiles).
    assert world.base_score("Germany") == 100 + CITY_SCORE * 1 + TILE_SCORE * 0
    assert world.score("Germany") == world.base_score("Germany")


def test_spawn_and_capital_income_and_upkeep(world):
    result = world.spawn_unit("Germany", "Berlin", 40, "test")
    assert "U1" in result
    berlin = world.cities["Berlin"]
    assert berlin.resources == 60
    assert world.units["U1"].resources == 40

    world.begin_turn("Germany")
    # Capital income +10, then 1 upkeep for the single unit.
    assert berlin.resources == 69
    assert world.units["U1"].ap == 2


def test_occupation_depends_on_unit_resources(world):
    berlin = world.cities["Berlin"]
    tile = next(t for t in world.tiles.values() if not t.is_city)
    unit = Unit("U1", "Germany", tile, 10)
    world.units["U1"] = unit
    assert world.controlled_tiles("Germany") == 1
    unit.resources = 3  # below OCCUPATION_MIN_RESOURCES
    assert world.controlled_tiles("Germany") == 0
    assert berlin.owner == "Germany"


def test_attack_city_captures_and_broadcasts(world):
    target = world.cities["Warsaw"]
    assert target.owner is None
    unit = Unit("U9", "Germany", target, 999)
    world.units["U9"] = unit
    result = world.attack_city("Germany", "U9", "Warsaw", "test")
    assert "captured" in result
    assert target.owner == "Germany"
    assert any(e["kind"] == "city_captured" and e["city"] == "Warsaw" for e in world.event_log)


def test_attack_city_defeat_destroys_unit(world):
    target = world.cities["Warsaw"]
    target.resources = 100
    unit = Unit("U9", "Germany", target, 5)
    world.units["U9"] = unit
    result = world.attack_city("Germany", "U9", "Warsaw", "test")
    assert "failed" in result
    assert "U9" not in world.units
    assert target.owner is None


def test_elimination_when_no_cities_or_units(world):
    # Strip France of everything.
    world.cities["Paris"].owner = None
    world._check_elimination("France")
    assert world.agents["France"]["alive"] is False
    assert "France" in world.eliminated
    assert any(e["kind"] == "faction_eliminated" for e in world.event_log)


def test_capital_annex_transfers_all_cities(world):
    berlin = world.cities["Berlin"]
    munich = world.cities["Munich"]
    # France has stormed Berlin and holds it; Germany still holds Munich.
    berlin.owner = "France"
    munich.owner = "Germany"
    world.capital_hold["Berlin"] = 0
    world.turn = CAPITAL_ANNEX_PROTECT_TURNS + 1

    world.start_round()

    assert munich.owner == "France"
    events = [e for e in world.event_log if e["kind"] == "capital_annexed"]
    assert events and events[-1]["actor"] == "France" and events[-1]["owner"] == "Germany"


def test_city_defense_includes_garrison_bonus(world):
    paris = world.cities["Paris"]
    paris.resources = 30
    assert world.city_defense(paris) == 30
    world.units["U1"] = Unit("U1", "France", paris, 20)
    assert world.city_defense(paris) == 30 + 20 + GARRISON_DEFENSE_BONUS
