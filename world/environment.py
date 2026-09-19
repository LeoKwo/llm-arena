from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from world import hexmap
from world.hexmap import DIRECTIONS, hex_distance, direction_toward, neighbors

# Make sure .env is loaded even if this module is imported before agent.llm_factory.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, default))


# --- Fixed rules ---
CITY_INCOME = 5.0
CAPITAL_INCOME = 10.0
CITY_RESOURCE_FLOOR = 5.0
GARRISON_DEFENSE_BONUS = 5.0
RECOVERY_PER_TURN = 1.0

MOVE_AP = 1
ATTACK_AP = 1
LOOT_AP = 2
SUPPLY_AP = 1
MERGE_AP = 0
MAX_AP = 2

GOAL_WEIGHTS = {"primary": 0.5, "secondary": 0.3, "tertiary": 0.2}
WAR_RELATION_THRESHOLD = -20.0

# --- Tunable via .env ---
ATTACK_ATTRITION = _env_float("ATTACK_ATTRITION", 0.3)
CITY_LOOT_RATE = _env_float("CITY_LOOT_RATE", 0.3)
UPKEEP_PER_UNIT = _env_float("UPKEEP_PER_UNIT", 1.0)
SUPPLY_RANGE = _env_int("SUPPLY_RANGE", 3)
RELATION_DECAY = _env_float("RELATION_DECAY", 3.0)
GOAL_SCORE_BONUS = _env_float("GOAL_SCORE_BONUS", 0.5)

CAPITAL_OF = {capital: nation for nation, capital in hexmap.CAPITALS.items()}


class Tile:
    __slots__ = ("q", "r", "resources", "owner", "city_name")

    def __init__(
        self,
        q: int,
        r: int,
        resources: float,
        owner: Optional[str] = None,
        city_name: Optional[str] = None,
    ) -> None:
        self.q = q
        self.r = r
        self.resources = float(resources)
        self.owner = owner
        self.city_name = city_name

    @property
    def is_city(self) -> bool:
        return self.city_name is not None

    @property
    def coords(self) -> tuple[int, int]:
        return (self.q, self.r)


class Unit:
    __slots__ = ("id", "owner", "tile", "resources", "ap")

    def __init__(
        self, unit_id: str, owner: str, tile: Tile, resources: float, ap: int = MAX_AP
    ) -> None:
        self.id = unit_id
        self.owner = owner
        self.tile = tile
        self.resources = float(resources)
        self.ap = ap


def build_default_world(seed=None) -> "World":
    if seed is None:
        raw = (os.environ.get("MAP_SEED") or "").strip()
        seed = raw or None
    rng = random.Random(seed)

    tiles: dict[tuple[int, int], Tile] = {}
    for (q, r) in hexmap.board_coords():
        tiles[(q, r)] = Tile(q, r, rng.uniform(*hexmap.TILE_RESOURCE_RANGE))

    cities: dict[str, Tile] = {}
    for name, (q, r) in hexmap.CITY_COORDS.items():
        tile = tiles[(q, r)]
        tile.city_name = name
        cities[name] = tile

    for nation, capital in hexmap.CAPITALS.items():
        cities[capital].owner = nation
        cities[capital].resources = hexmap.CAPITAL_RESOURCES
    for name, tile in cities.items():
        if tile.owner is None:
            tile.resources = rng.uniform(*hexmap.CITY_RESOURCE_RANGE)

    agents = {nation: {"home": hexmap.CAPITALS[nation]} for nation in hexmap.CAPITALS}
    return World(tiles=tiles, cities=cities, agents=agents)


class World:
    def __init__(
        self,
        tiles: dict[tuple[int, int], Tile],
        cities: dict[str, Tile],
        agents: dict[str, dict],
    ) -> None:
        self.turn = 0
        self.max_turns: Optional[int] = None
        self.tiles = tiles
        self.cities = cities
        self.agents: dict[str, dict] = {
            name: {"home": info.get("home"), "alive": True}
            for name, info in agents.items()
        }
        self.units: dict[str, Unit] = {}
        self._unit_seq = 0
        self.history: list[dict] = []
        self.relations: dict[tuple[str, str], float] = {}
        self.goal_checks: dict[str, dict[str, Callable]] = {}
        self.eliminated: list[str] = []
        self.spawns_this_turn: dict[str, int] = {name: 0 for name in agents}
        self.broadcasts: list[dict] = []

    def _broadcast(self, kind: str, **data) -> None:
        self.broadcasts.append({"kind": kind, **data})

    def take_broadcasts(self) -> list[dict]:
        items = self.broadcasts
        self.broadcasts = []
        return items

    # ------------------------------------------------------------------ lookup
    def living_agents(self) -> list[str]:
        return [name for name, a in self.agents.items() if a["alive"]]

    def cities_of(self, agent_name: str) -> list[Tile]:
        return [
            tile for _, tile in sorted(self.cities.items()) if tile.owner == agent_name
        ]

    def units_of(self, agent_name: str) -> list[Unit]:
        return [unit for unit in self.units.values() if unit.owner == agent_name]

    def units_on(self, tile: Tile) -> list[Unit]:
        return [unit for unit in self.units.values() if unit.tile is tile]

    def city_count(self, agent_name: str) -> int:
        return len(self.cities_of(agent_name))

    def unit_count(self, agent_name: str) -> int:
        return len(self.units_of(agent_name))

    def resources_total(self, agent_name: str) -> float:
        cities = sum(tile.resources for tile in self.cities_of(agent_name))
        units = sum(unit.resources for unit in self.units_of(agent_name))
        return cities + units

    def controller_of(self, tile: Tile) -> Optional[str]:
        """Cities are owned persistently; land is controlled only while a unit
        with at least 1 resource stands on it."""
        if tile.is_city:
            return tile.owner
        for unit in self.units_on(tile):
            if unit.resources >= 1:
                return unit.owner
        return None

    def relation(self, a: str, b: str) -> float:
        if a == b:
            return 100.0
        key = (a, b) if a < b else (b, a)
        return self.relations.get(key, 0.0)

    def war_count(self, agent_name: str) -> int:
        return sum(
            1
            for other in self.living_agents()
            if other != agent_name
            and self.relation(agent_name, other) <= WAR_RELATION_THRESHOLD
        )

    def _set_relation(self, a: str, b: str, delta: float) -> None:
        key = (a, b) if a < b else (b, a)
        value = self.relations.get(key, 0.0) + delta
        self.relations[key] = max(-100.0, min(100.0, value))

    def tile_at(self, q: int, r: int) -> Optional[Tile]:
        return self.tiles.get((q, r))

    def adjacent(self, tile: Tile) -> dict[str, Tile]:
        found = {}
        for name, (q, r) in neighbors(tile.q, tile.r).items():
            neighbor = self.tiles.get((q, r))
            if neighbor is not None:
                found[name] = neighbor
        return found

    def _reduce_city(self, city: Tile, amount: float) -> float:
        """Reduce a city's resources without going below the floor."""
        reducible = max(0.0, city.resources - CITY_RESOURCE_FLOOR)
        taken = min(amount, reducible)
        city.resources -= taken
        return taken

    # ------------------------------------------------------------- turn cycle
    def begin_turn(self, agent_name: str) -> None:
        if not self.agents[agent_name]["alive"]:
            return
        capital = hexmap.CAPITALS.get(agent_name)
        for tile in self.cities_of(agent_name):
            tile.resources += CAPITAL_INCOME if tile.city_name == capital else CITY_INCOME
        self._pay_upkeep(agent_name)
        for unit in self.units_of(agent_name):
            unit.ap = MAX_AP
        self.spawns_this_turn[agent_name] = 0

    def _pay_upkeep(self, agent_name: str) -> None:
        due = len(self.units_of(agent_name)) * UPKEEP_PER_UNIT
        if due <= 0:
            return
        for city in sorted(
            self.cities_of(agent_name), key=lambda t: t.resources, reverse=True
        ):
            if due <= 0:
                break
            due -= self._reduce_city(city, due)
        if due > 0:
            self._apply_unit_attrition(self.units_of(agent_name), due)
            self._check_elimination(agent_name)

    def end_faction_turn(self, agent_name: str) -> None:
        """Units that spent no action points recover a little strength."""
        if not self.agents[agent_name]["alive"]:
            return
        for unit in self.units_of(agent_name):
            if unit.ap >= MAX_AP:
                unit.resources += RECOVERY_PER_TURN

    def advance(self) -> None:
        self.turn += 1
        for key in list(self.relations):
            value = self.relations[key]
            if value < 0:
                value = min(0.0, value + RELATION_DECAY)
            elif value > 0:
                value = max(0.0, value - RELATION_DECAY)
            self.relations[key] = value

    # ------------------------------------------------------------- unit setup
    def _new_unit_id(self) -> str:
        self._unit_seq += 1
        return f"U{self._unit_seq}"

    def spawn_unit(self, owner: str, city_name: str, amount, reason: str = "") -> str:
        if not self.agents[owner]["alive"]:
            return f"{owner} has been eliminated and cannot act."
        city = self.cities.get(city_name)
        if city is None:
            return f"Unknown city '{city_name}'."
        if city.owner != owner:
            return f"{owner} does not control {city_name}."
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return f"amount must be a number, got '{amount}'."
        if amount <= 0:
            return "amount must be greater than 0."
        if amount > city.resources:
            return (
                f"Cannot allocate {int(amount)} from {city_name}; it only has "
                f"{int(city.resources)} resources."
            )
        cap = self.city_count(owner)
        if self.spawns_this_turn.get(owner, 0) >= cap:
            return (
                f"Spawn limit reached: {owner} controls {cap} cities and may "
                f"create at most {cap} units per turn."
            )
        city.resources -= amount
        unit = Unit(self._new_unit_id(), owner, city, amount, ap=MAX_AP)
        self.units[unit.id] = unit
        self.spawns_this_turn[owner] = self.spawns_this_turn.get(owner, 0) + 1
        result = (
            f"{owner} created unit {unit.id} at {city_name} with {int(amount)} "
            f"resources (AP {unit.ap})."
        )
        self._record(owner, "spawn_unit", city_name, result, reason)
        return result

    def disband_unit(self, owner: str, unit_id: str, reason: str = "") -> str:
        unit = self.units.get(unit_id)
        if unit is None or unit.owner != owner:
            return f"{owner} has no unit '{unit_id}'."
        tile = unit.tile
        if not (tile.is_city and tile.owner == owner):
            return (
                f"{unit.id} can only be disbanded while standing on a city you "
                f"control."
            )
        returned = unit.resources
        tile.resources += returned
        self._remove_unit(unit.id)
        result = (
            f"{unit.id} disbanded at {tile.city_name}: {int(returned)} resources "
            f"returned to the city (city now {int(tile.resources)})."
        )
        self._record(owner, "disband_unit", tile.city_name, result, reason, unit_id=unit_id)
        return result

    def merge_units(
        self, owner: str, unit_id: str, other_id: str, reason: str = ""
    ) -> str:
        unit = self.units.get(unit_id)
        other = self.units.get(other_id)
        if unit is None or unit.owner != owner:
            return f"{owner} has no unit '{unit_id}'."
        if other is None or other.owner != owner:
            return f"{owner} has no unit '{other_id}'."
        if unit_id == other_id:
            return "Cannot merge a unit with itself."
        if unit.tile is not other.tile:
            return f"{unit_id} and {other_id} are not on the same tile."
        if unit.ap < MERGE_AP:
            return f"{unit_id} cannot merge right now."
        unit.resources += other.resources
        unit.ap = min(unit.ap, other.ap)
        self._remove_unit(other.id)
        result = (
            f"{unit_id} absorbed {other_id}: now {int(unit.resources)} resources "
            f"(AP {unit.ap})."
        )
        self._record(owner, "merge_units", unit.tile.city_name or f"({unit.tile.q},{unit.tile.r})", result, reason, unit_id=unit_id)
        return result

    # --------------------------------------------------------------- movement
    def move_unit(
        self, owner: str, unit_id: str, direction: str, reason: str = ""
    ) -> str:
        unit = self.units.get(unit_id)
        if unit is None or unit.owner != owner:
            return f"{owner} has no unit '{unit_id}'."
        if unit.ap < MOVE_AP:
            return f"{unit.id} has no action points left to move."
        direction = (direction or "").strip().upper()
        delta = DIRECTIONS.get(direction)
        if delta is None:
            return (
                f"Unknown direction '{direction}'. Use one of: "
                f"{', '.join(DIRECTIONS)}."
            )
        target = self.tiles.get((unit.tile.q + delta[0], unit.tile.r + delta[1]))
        if target is None:
            return f"There is no tile to the {direction} of {unit.id}."
        if target.is_city and target.owner != owner:
            return (
                f"{unit.id} cannot enter {target.city_name}; enemy or neutral "
                f"cities can only be taken with attack_city."
            )

        origin = unit.tile
        location = target.city_name or f"({target.q},{target.r})"
        unit.ap -= MOVE_AP
        defenders = [u for u in self.units_on(target) if u.owner != owner]

        if defenders:
            defense = sum(u.resources for u in defenders) + GARRISON_DEFENSE_BONUS
            if unit.resources > defense:
                for enemy in list(defenders):
                    self._remove_unit(enemy.id)
                unit.resources = max(
                    0.0, unit.resources - _floor(defense * ATTACK_ATTRITION)
                )
                unit.tile = target
                result = (
                    f"{unit.id} defeated the garrison at {location} "
                    f"(defense {int(defense)}) and took the tile "
                    f"(now {int(unit.resources)} resources)."
                )
                self._record(owner, "battle", location, result, reason, unit_id=unit.id)
                for enemy in defenders:
                    self._broadcast(
                        "unit_destroyed",
                        actor=owner,
                        victim=enemy.owner,
                        unit=enemy.id,
                        tile=location,
                    )
                    self._check_elimination(enemy.owner)
                return result

            loss = _floor(unit.resources * ATTACK_ATTRITION)
            self._remove_unit(unit.id)
            self._apply_unit_attrition(defenders, loss)
            result = (
                f"{unit.id} ({int(unit.resources)}) was destroyed attacking the "
                f"garrison at {location} (defense {int(defense)})."
            )
            self._record(owner, "battle", location, result, reason, unit_id=unit.id)
            self._broadcast(
                "unit_destroyed", actor=defenders[0].owner, victim=owner,
                unit=unit_id, tile=location,
            )
            self._check_elimination(owner)
            return result

        unit.tile = target
        result = (
            f"{unit.id} moved {direction} from "
            f"{origin.city_name or f'({origin.q},{origin.r})'} to {location} "
            f"(AP {unit.ap})."
        )
        self._record(owner, "move_unit", location, result, reason, unit_id=unit_id)
        return result

    def supply_unit(
        self, owner: str, unit_id: str, city_name: str, amount, reason: str = ""
    ) -> str:
        unit = self.units.get(unit_id)
        if unit is None or unit.owner != owner:
            return f"{owner} has no unit '{unit_id}'."
        city = self.cities.get(city_name)
        if city is None:
            return f"Unknown city '{city_name}'."
        if city.owner != owner:
            return f"{owner} does not control {city_name}."
        if unit.ap < SUPPLY_AP:
            return f"{unit.id} has no action points left to receive supply."
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return f"amount must be a number, got '{amount}'."
        if amount <= 0:
            return "amount must be greater than 0."
        if hex_distance(unit.tile.coords, city.coords) > SUPPLY_RANGE:
            return (
                f"{unit.id} is out of supply range from {city_name} "
                f"(max {SUPPLY_RANGE} tiles)."
            )
        available = max(0.0, city.resources - CITY_RESOURCE_FLOOR)
        if amount > available:
            return (
                f"{city_name} can only spare {int(available)} resources "
                f"(keeping {int(CITY_RESOURCE_FLOOR)} in reserve)."
            )
        city.resources -= amount
        unit.resources += amount
        unit.ap -= SUPPLY_AP
        result = (
            f"{city_name} supplied {unit.id} with {int(amount)} resources "
            f"(unit now {int(unit.resources)}, city {int(city.resources)}, "
            f"AP {unit.ap})."
        )
        self._record(owner, "supply_unit", city_name, result, reason, unit_id=unit_id)
        return result

    # ---------------------------------------------------------------- combat
    def attack_city(
        self, owner: str, unit_id: str, city_name: str, reason: str = ""
    ) -> str:
        unit = self.units.get(unit_id)
        if unit is None or unit.owner != owner:
            return f"{owner} has no unit '{unit_id}'."
        city = self.cities.get(city_name)
        if city is None:
            return f"Unknown city '{city_name}'."
        if city.owner == owner:
            return f"{owner} already controls {city_name}."
        if unit.ap < ATTACK_AP:
            return f"{unit.id} has no action points left to attack."
        if hex_distance(unit.tile.coords, city.coords) > 1:
            return f"{unit.id} is not within 1 tile of {city_name}; move closer first."

        unit.ap -= ATTACK_AP
        defender = city.owner
        garrison = [u for u in self.units_on(city) if u.owner == defender]
        defense = city.resources + sum(u.resources for u in garrison)
        if garrison:
            defense += GARRISON_DEFENSE_BONUS
        if defender is not None:
            self._set_relation(owner, defender, -30.0)

        if unit.resources > defense:
            loot = _floor(city.resources * CITY_LOOT_RATE)
            unit.resources += loot
            city.resources = max(CITY_RESOURCE_FLOOR, city.resources * 0.5)
            for guard in list(garrison):
                self._remove_unit(guard.id)
            unit.resources = max(
                0.0, unit.resources - _floor(defense * ATTACK_ATTRITION)
            )
            city.owner = owner
            unit.tile = city
            result = (
                f"{owner}'s {unit.id} captured {city_name} from "
                f"{defender or 'neutral forces'} (defense {int(defense)}): looted "
                f"{int(loot)}, city now {int(city.resources)}, unit now "
                f"{int(unit.resources)}."
            )
            self._record(owner, "attack_city", city_name, result, reason, unit_id=unit_id)
            self._broadcast(
                "city_captured", actor=owner, city=city_name, previous=defender
            )
            self._on_capital_change(city, defender, owner)
            if defender is not None:
                self._check_elimination(defender)
            return result

        # Failed assault: attacker dies, defenders take attrition.
        spent = int(unit.resources)
        loss = _floor(unit.resources * ATTACK_ATTRITION)
        self._remove_unit(unit.id)
        self._damage_defenders(garrison, city, loss)
        result = (
            f"{owner}'s {unit_id} ({spent}) failed to take {city_name} "
            f"(defense {int(defense)}); the unit was destroyed."
        )
        self._record(owner, "attack_city", city_name, result, reason, unit_id=unit_id)
        self._check_elimination(owner)
        return result

    # ------------------------------------------------------------ land actions
    def loot_tile(self, owner: str, unit_id: str, reason: str = "") -> str:
        unit = self.units.get(unit_id)
        if unit is None or unit.owner != owner:
            return f"{owner} has no unit '{unit_id}'."
        if unit.ap < LOOT_AP:
            return f"{unit.id} does not have enough action points to loot."
        tile = unit.tile
        if tile.is_city:
            return f"{unit.id} is standing on a city; there is nothing to loot."
        if tile.resources <= 0:
            return f"Tile ({tile.q},{tile.r}) has already been stripped bare."
        gained = int(tile.resources)
        unit.resources += tile.resources
        tile.resources = 0.0
        unit.ap = max(0, unit.ap - LOOT_AP)
        result = (
            f"{unit.id} looted ({tile.q},{tile.r}) for {gained} resources "
            f"(unit now {int(unit.resources)}); it cannot move again this turn."
        )
        self._record(owner, "loot_tile", f"({tile.q},{tile.r})", result, reason, unit_id=unit_id)
        return result

    def end_turn(self, owner: str, reason: str = "") -> str:
        result = f"{owner} ended its turn."
        self._record(owner, "end_turn", "", result, reason)
        return result

    # ----------------------------------------------------------------- history
    def _record(
        self,
        actor: str,
        action: str,
        target: str,
        result: str,
        reason: str = "",
        unit_id: str = "",
    ) -> None:
        self.history.append(
            {
                "turn": self.turn,
                "actor": actor,
                "action": action,
                "unit": unit_id,
                "target": target,
                "reason": reason,
                "result": result,
            }
        )

    def _remove_unit(self, unit_id: str) -> None:
        self.units.pop(unit_id, None)

    def _apply_unit_attrition(self, units: list[Unit], total: float) -> None:
        """Reduce the given units by ``total`` resources, largest first."""
        remaining = total
        for unit in sorted(units, key=lambda u: u.resources, reverse=True):
            if remaining <= 0:
                break
            take = min(unit.resources, remaining)
            unit.resources -= take
            remaining -= take
            if unit.resources <= 0:
                self._remove_unit(unit.id)

    def _damage_defenders(
        self, garrison: list[Unit], city: Tile, amount: float
    ) -> None:
        unit_total = sum(u.resources for u in garrison)
        absorbed = min(unit_total, amount)
        self._apply_unit_attrition(garrison, absorbed)
        leftover = amount - absorbed
        if leftover > 0:
            self._reduce_city(city, leftover)

    def _check_elimination(self, agent_name: str) -> None:
        if not self.agents[agent_name]["alive"]:
            return
        if self.city_count(agent_name) > 0 or self.unit_count(agent_name) > 0:
            return
        self.agents[agent_name]["alive"] = False
        self.eliminated.append(agent_name)
        self._broadcast("faction_eliminated", actor=agent_name)

    def _on_capital_change(
        self, city: Tile, previous_owner: Optional[str], new_owner: Optional[str]
    ) -> None:
        original = CAPITAL_OF.get(city.city_name)
        if original is None or previous_owner == new_owner:
            return
        if previous_owner == original:
            for tile in self.cities_of(previous_owner):
                self._reduce_city(tile, 2.0)
            for unit in self.units_of(previous_owner):
                unit.resources = max(0.0, unit.resources - 1.0)
            self._broadcast(
                "capital_fallen",
                owner=previous_owner,
                city=city.city_name,
                by=new_owner,
            )
        if new_owner == original:
            for tile in self.cities_of(new_owner):
                tile.resources += 5.0
            for unit in self.units_of(new_owner):
                unit.resources += 2.0
            self._broadcast("capital_retaken", owner=new_owner, city=city.city_name)

    # -------------------------------------------------------------- objectives
    def register_goals(self, agent_name: str, checks: dict[str, Callable]) -> None:
        self.goal_checks[agent_name] = checks

    def goal_completion(self, agent_name: str) -> float:
        checks = self.goal_checks.get(agent_name, {})
        total = 0.0
        for level, weight in GOAL_WEIGHTS.items():
            fn = checks.get(level)
            if fn is None:
                continue
            try:
                progress = float(fn(self, agent_name))
            except Exception:
                progress = 0.0
            progress = max(0.0, min(1.0, progress))
            total += weight * progress
        return total

    def score(self, agent_name: str) -> float:
        if not self.agents[agent_name]["alive"]:
            return 0.0
        return self.resources_total(agent_name) * (
            1.0 + GOAL_SCORE_BONUS * self.goal_completion(agent_name)
        )

    def rankings(self) -> list[tuple[str, float]]:
        return sorted(
            ((name, self.score(name)) for name in self.agents),
            key=lambda item: item[1],
            reverse=True,
        )

    def check_end_conditions(self) -> tuple[bool, Optional[str]]:
        alive = self.living_agents()
        if len(alive) <= 1:
            winner = alive[0] if alive else None
            return True, winner
        return False, None

    # ------------------------------------------------------------------ render
    def city_defense(self, city: Tile) -> float:
        garrison = [u for u in self.units_on(city) if u.owner == city.owner]
        defense = city.resources + sum(u.resources for u in garrison)
        if garrison:
            defense += GARRISON_DEFENSE_BONUS
        return defense

    def _describe_tile(self, tile: Tile) -> str:
        if tile.is_city:
            desc = (
                f"{tile.city_name} (city, owner={tile.owner or 'neutral'}, "
                f"res={int(tile.resources)}, defense={int(self.city_defense(tile))})"
            )
        else:
            controller = self.controller_of(tile) or "unowned"
            desc = (
                f"({tile.q},{tile.r}) land res={int(tile.resources)} "
                f"controller={controller}"
            )
        units = self.units_on(tile)
        if units:
            desc += " units=" + ",".join(
                f"{u.owner}:{u.id}({int(u.resources)})" for u in units
            )
        return desc

    def _unit_neighbourhood(self, unit: Unit) -> str:
        parts = [
            f"{direction}: {self._describe_tile(tile)}"
            for direction, tile in self.adjacent(unit.tile).items()
        ]
        location = unit.tile.city_name or f"({unit.tile.q},{unit.tile.r})"
        return (
            f"{unit.id} at {location} (res={int(unit.resources)}, AP={unit.ap}) | "
            + " | ".join(parts)
        )

    def _route_hints(self, unit: Unit) -> str:
        targets = [
            (name, tile, hex_distance(unit.tile.coords, tile.coords))
            for name, tile in self.cities.items()
            if tile.owner != unit.owner
        ]
        targets.sort(key=lambda item: item[2])
        hints = [
            f"{name} ({direction_toward(unit.tile.coords, tile.coords)}, d{distance})"
            for name, tile, distance in targets[:3]
        ]
        return f"{unit.id} -> " + "; ".join(hints) if hints else ""

    def _unit_options(self, unit: Unit) -> str:
        options = []
        partners = [
            u.id for u in self.units_on(unit.tile) if u.owner == unit.owner and u.id != unit.id
        ]
        if partners:
            options.append("merge with " + ", ".join(partners))
        for city in self.cities_of(unit.owner):
            if hex_distance(unit.tile.coords, city.coords) <= SUPPLY_RANGE:
                spare = int(max(0.0, city.resources - CITY_RESOURCE_FLOOR))
                if spare > 0:
                    options.append(f"supply from {city.city_name} (up to {spare})")
        if unit.tile.is_city and unit.tile.owner == unit.owner:
            options.append("disband here")
        return "; ".join(options) if options else "none"

    def get_observation(self, agent_name: str) -> str:
        info = self.agents[agent_name]
        if self.max_turns:
            remaining = max(0, self.max_turns - (self.turn + 1))
            header = (
                f"Turn {self.turn + 1} of {self.max_turns} — {remaining} turns "
                f"remaining. You are {agent_name}."
            )
        else:
            header = f"Turn {self.turn}. You are {agent_name}."
        lines = [header]
        if not info["alive"]:
            lines.append("You have been eliminated.")
            return "\n".join(lines)

        cities = self.cities_of(agent_name)
        units = self.units_of(agent_name)
        lines.append(
            f"Controlled resources (used for scoring): "
            f"{int(self.resources_total(agent_name))} "
            f"({int(sum(c.resources for c in cities))} in cities, "
            f"{int(sum(u.resources for u in units))} in units)."
        )
        own_capital = hexmap.CAPITALS.get(agent_name)
        lines.append(f"Your cities ({len(cities)}):")
        for tile in cities:
            mark = " [capital, +10/turn]" if tile.city_name == own_capital else ""
            lines.append(
                f"- {tile.city_name} at ({tile.q},{tile.r}): "
                f"resources={int(tile.resources)}, "
                f"defense={int(self.city_defense(tile))}{mark}"
            )
        lines.append(f"Your units ({len(units)}), each gets 2 action points per turn:")
        if units:
            for unit in units:
                location = unit.tile.city_name or f"({unit.tile.q},{unit.tile.r})"
                lines.append(
                    f"- {unit.id} at {location}: resources={int(unit.resources)}, "
                    f"AP={unit.ap} | options: {self._unit_options(unit)}"
                )
        else:
            lines.append("- (none)")

        cap = len(cities)
        used = self.spawns_this_turn.get(agent_name, 0)
        lines.append(
            f"Unit creation this turn: {used}/{cap} used "
            f"(max = number of cities you control). New units act immediately."
        )
        lines.append(
            "Rules: cities +5/turn (your capital +10), never below 5 resources. "
            "A unit standing on a tile controls it until it leaves. Units that "
            "spend no action points recover 1 resource. A location defended by "
            "one or more units gets +5 defense. Attacks: the stronger side wins; "
            "the winner loses 30% of the loser's strength, the loser's units are "
            "destroyed. Capturing a city loots 30% of it, halves it, and the "
            "attacker moves in. Each unit costs 1 upkeep per turn, paid by your "
            "cities (unpaid upkeep drains the units)."
        )

        lines.append("All cities on the board:")
        for name, tile in sorted(self.cities.items()):
            original = CAPITAL_OF.get(name)
            mark = f" [capital of {original}]" if original else ""
            guard = [u for u in self.units_on(tile) if u.owner == tile.owner]
            gtxt = f", garrison={len(guard)}" if guard else ""
            lines.append(
                f"- {name} at ({tile.q},{tile.r}): owner={tile.owner or 'neutral'}, "
                f"resources={int(tile.resources)}, "
                f"defense={int(self.city_defense(tile))}{gtxt}{mark}"
            )

        if units:
            lines.append("Unit surroundings and routes (E/SE/SW/W/NW/NE):")
            for unit in units:
                lines.append("- " + self._unit_neighbourhood(unit))
                hint = self._route_hints(unit)
                if hint:
                    lines.append("  " + hint)

        if self.history:
            lines.append("Recent events:")
            for event in self.history[-8:]:
                target = f" {event['target']}" if event["target"] else ""
                lines.append(
                    f"- T{event['turn']} {event['actor']} {event['action']}{target}"
                )
        return "\n".join(lines)

    def snapshot(self) -> dict:
        return {
            "turn": self.turn,
            "max_turns": self.max_turns,
            "tiles": [
                {
                    "q": tile.q,
                    "r": tile.r,
                    "resources": int(tile.resources),
                    "owner": self.controller_of(tile),
                    "is_city": tile.is_city,
                    "city": tile.city_name,
                }
                for tile in self.tiles.values()
            ],
            "cities": [
                {
                    "name": tile.city_name,
                    "q": tile.q,
                    "r": tile.r,
                    "owner": tile.owner,
                    "resources": int(tile.resources),
                    "capital": CAPITAL_OF.get(tile.city_name),
                    "defense": int(self.city_defense(tile)),
                }
                for tile in sorted(self.cities.values(), key=lambda t: t.city_name)
            ],
            "units": [
                {
                    "id": unit.id,
                    "owner": unit.owner,
                    "q": unit.tile.q,
                    "r": unit.tile.r,
                    "resources": int(unit.resources),
                    "ap": unit.ap,
                }
                for unit in self.units.values()
            ],
            "agents": {
                name: {
                    "alive": info["alive"],
                    "home": info.get("home"),
                    "resources_total": int(self.resources_total(name)),
                    "cities": [c.city_name for c in self.cities_of(name)],
                    "unit_count": self.unit_count(name),
                }
                for name, info in self.agents.items()
            },
            "relations": {
                f"{a}|{b}": int(value) for (a, b), value in self.relations.items()
            },
            "scores": {name: round(self.score(name), 1) for name in self.agents},
            "goal_completion": {
                name: round(self.goal_completion(name), 3) for name in self.agents
            },
        }


def _floor(value: float) -> int:
    return int(value)
