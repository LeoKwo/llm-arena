from __future__ import annotations

import os
import random
from typing import Callable, Optional

from world import hexmap
from world.hexmap import DIRECTIONS, hex_distance, direction_toward, neighbors

TERRITORY_WEIGHT = 20.0
GOAL_WEIGHT = 100.0
GOAL_WEIGHTS = {"primary": 0.5, "secondary": 0.3, "tertiary": 0.2}

CITY_INCOME = 5.0
CAPITAL_INCOME = 10.0
DISBAND_BONUS = 5.0
GARRISON_GAIN = 2.0
CAPITAL_LOSS_CITY_PENALTY = 2.0
CAPITAL_LOSS_UNIT_PENALTY = 1.0
CAPITAL_REGAIN_CITY_BONUS = 5.0
CAPITAL_REGAIN_UNIT_BONUS = 2.0

CAPITAL_OF = {capital: nation for nation, capital in hexmap.CAPITALS.items()}
MOVE_AP = 1
ATTACK_AP = 1
CLAIM_AP = 2
GARRISON_AP = 2
MAX_AP = 2
WAR_RELATION_THRESHOLD = -20.0


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
            tile
            for _, tile in sorted(self.cities.items())
            if tile.owner == agent_name
        ]

    def units_of(self, agent_name: str) -> list[Unit]:
        return [unit for unit in self.units.values() if unit.owner == agent_name]

    def city_count(self, agent_name: str) -> int:
        return len(self.cities_of(agent_name))

    def unit_count(self, agent_name: str) -> int:
        return len(self.units_of(agent_name))

    def resources_total(self, agent_name: str) -> float:
        cities = sum(tile.resources for tile in self.cities_of(agent_name))
        units = sum(unit.resources for unit in self.units_of(agent_name))
        return cities + units

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

    # ------------------------------------------------------------- turn cycle
    def begin_turn(self, agent_name: str) -> None:
        if not self.agents[agent_name]["alive"]:
            return
        capital = hexmap.CAPITALS.get(agent_name)
        for tile in self.cities_of(agent_name):
            tile.resources += (
                CAPITAL_INCOME if tile.city_name == capital else CITY_INCOME
            )
        for unit in self.units_of(agent_name):
            unit.ap = MAX_AP
        self.spawns_this_turn[agent_name] = 0

    def advance(self) -> None:
        self.turn += 1

    # ------------------------------------------------------------- unit setup
    def _new_unit_id(self) -> str:
        self._unit_seq += 1
        return f"U{self._unit_seq}"

    def spawn_unit(
        self, owner: str, city_name: str, amount, reason: str = ""
    ) -> str:
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
                f"Cannot allocate {int(amount)} from {city_name}; "
                f"it only has {int(city.resources)} resources."
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
            f"{owner} created unit {unit.id} at {city_name} with "
            f"{int(amount)} resources (AP {unit.ap})."
        )
        self._record(owner, "spawn_unit", city_name, result, reason)
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
        enemies = [
            other
            for other in self.units.values()
            if other.owner != owner and other.tile is target
        ]
        unit.ap -= MOVE_AP

        if enemies:
            # Unit vs unit: the side with more resources wins and loses nothing;
            # the loser's unit is destroyed with all of its resources.
            for enemy in sorted(enemies, key=lambda u: u.resources, reverse=True):
                if unit.resources > enemy.resources:
                    self._remove_unit(enemy.id)
                    self._record(
                        owner,
                        "battle",
                        location,
                        f"{unit.id} ({int(unit.resources)}) destroyed "
                        f"{enemy.id} ({enemy.owner}, {int(enemy.resources)}) "
                        f"at {location}.",
                        reason,
                        unit_id=unit.id,
                    )
                    self._broadcast(
                        "unit_destroyed",
                        actor=owner,
                        victim=enemy.owner,
                        unit=enemy.id,
                        tile=location,
                    )
                else:
                    message = (
                        f"{unit.id} ({int(unit.resources)}) was destroyed by "
                        f"{enemy.id} ({enemy.owner}, {int(enemy.resources)}) "
                        f"at {location}."
                    )
                    self._remove_unit(unit.id)
                    self._record(
                        owner, "battle", location, message, reason, unit_id=unit.id
                    )
                    self._broadcast(
                        "unit_destroyed",
                        actor=enemy.owner,
                        victim=owner,
                        unit=unit.id,
                        tile=location,
                    )
                    self._check_elimination(owner)
                    return message

        unit.tile = target
        result = (
            f"{unit.id} moved {direction} from "
            f"{origin.city_name or f'({origin.q},{origin.r})'} to {location} "
            f"(AP {unit.ap})."
        )
        self._record(owner, "move_unit", location, result, reason, unit_id=unit_id)
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
        tile.resources += returned + DISBAND_BONUS
        self._remove_unit(unit.id)
        result = (
            f"{unit.id} disbanded at {tile.city_name}: {int(returned)} resources "
            f"returned to the city plus {int(DISBAND_BONUS)} bonus "
            f"(city now {int(tile.resources)})."
        )
        self._record(owner, "disband_unit", tile.city_name, result, reason, unit_id=unit_id)
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
            return (
                f"{unit.id} is not within 1 tile of {city_name}; move closer "
                f"first."
            )

        unit.ap -= ATTACK_AP
        defender = city.owner
        if defender is not None:
            self._set_relation(owner, defender, -30.0)

        if unit.resources > city.resources:
            captured = int(city.resources * 0.5)
            city.resources = float(captured)
            city.owner = owner
            result = (
                f"{owner}'s {unit.id} ({int(unit.resources)}) captured "
                f"{city_name} from {defender or 'neutral forces'}; the city now "
                f"holds {captured} resources (halved)."
            )
            self._record(owner, "attack_city", city_name, result, reason, unit_id=unit_id)
            self._broadcast("city_captured", actor=owner, city=city_name, previous=defender)
            self._on_capital_change(city, defender, owner)
            if defender is not None:
                self._check_elimination(defender)
            return result

        spent = int(unit.resources)
        unit.resources = 0.0
        self._remove_unit(unit.id)
        result = (
            f"{owner}'s {unit_id} ({spent}) failed to take {city_name} "
            f"({int(city.resources)}); the unit was destroyed."
        )
        self._record(owner, "attack_city", city_name, result, reason, unit_id=unit_id)
        self._check_elimination(owner)
        return result

    # ------------------------------------------------------------ land actions
    def claim_tile(self, owner: str, unit_id: str, reason: str = "") -> str:
        unit = self.units.get(unit_id)
        if unit is None or unit.owner != owner:
            return f"{owner} has no unit '{unit_id}'."
        if unit.ap < CLAIM_AP:
            return f"{unit.id} does not have enough action points to claim land."
        tile = unit.tile
        if tile.is_city:
            return f"{unit.id} is standing on a city; use attack_city instead."
        if tile.owner is not None:
            return f"Tile ({tile.q},{tile.r}) is already controlled by {tile.owner}."
        gained = int(tile.resources)
        unit.resources += tile.resources
        tile.resources = 0.0
        tile.owner = owner
        unit.ap = max(0, unit.ap - CLAIM_AP)
        result = (
            f"{unit.id} claimed ({tile.q},{tile.r}) and gained {gained} "
            f"resources (unit now has {int(unit.resources)}); it cannot move "
            f"again this turn."
        )
        self._record(owner, "claim_tile", f"({tile.q},{tile.r})", result, reason, unit_id=unit_id)
        return result

    def garrison(self, owner: str, unit_id: str, reason: str = "") -> str:
        unit = self.units.get(unit_id)
        if unit is None or unit.owner != owner:
            return f"{owner} has no unit '{unit_id}'."
        if unit.ap < GARRISON_AP:
            return f"{unit.id} does not have enough action points to garrison."
        tile = unit.tile
        unit.ap = max(0, unit.ap - GARRISON_AP)
        if tile.is_city or tile.owner is not None:
            result = (
                f"{unit.id} garrisoned at "
                f"({tile.q},{tile.r}) but gained nothing (the tile is not "
                f"unclaimed land)."
            )
        else:
            unit.resources += GARRISON_GAIN
            result = (
                f"{unit.id} garrisoned at ({tile.q},{tile.r}) and gained "
                f"{int(GARRISON_GAIN)} resources (now {int(unit.resources)})."
            )
        self._record(owner, "garrison", f"({tile.q},{tile.r})", result, reason, unit_id=unit_id)
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
            # Losing your capital hurts every remaining city and unit.
            for tile in self.cities_of(previous_owner):
                tile.resources = max(0.0, tile.resources - CAPITAL_LOSS_CITY_PENALTY)
            for unit in self.units_of(previous_owner):
                unit.resources = max(0.0, unit.resources - CAPITAL_LOSS_UNIT_PENALTY)
            self._broadcast(
                "capital_fallen",
                owner=previous_owner,
                city=city.city_name,
                by=new_owner,
            )
        if new_owner == original:
            # Recapturing your capital rallies the nation.
            for tile in self.cities_of(new_owner):
                tile.resources += CAPITAL_REGAIN_CITY_BONUS
            for unit in self.units_of(new_owner):
                unit.resources += CAPITAL_REGAIN_UNIT_BONUS
            self._broadcast(
                "capital_retaken", owner=new_owner, city=city.city_name
            )

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
        return self.resources_total(agent_name) + self.goal_completion(agent_name) * GOAL_WEIGHT

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
    def _describe_tile(self, tile: Tile) -> str:
        owner = tile.owner or "unowned"
        if tile.is_city:
            desc = (
                f"{tile.city_name} (city, owner={tile.owner or 'neutral'}, "
                f"res={int(tile.resources)})"
            )
        else:
            desc = f"({tile.q},{tile.r}) land res={int(tile.resources)} owner={owner}"
        units = [u for u in self.units.values() if u.tile is tile]
        if units:
            desc += " units=" + ",".join(
                f"{u.owner}:{u.id}({int(u.resources)})" for u in units
            )
        return desc

    def _unit_neighbourhood(self, unit: Unit) -> str:
        parts = []
        for direction, tile in self.adjacent(unit.tile).items():
            parts.append(f"{direction}: {self._describe_tile(tile)}")
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

    def get_observation(self, agent_name: str) -> str:
        info = self.agents[agent_name]
        lines = [f"Turn {self.turn}. You are {agent_name}."]
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
        capital_of = {
            nation: capital for nation, capital in hexmap.CAPITALS.items()
        }
        own_capital = hexmap.CAPITALS.get(agent_name)
        lines.append(f"Your cities ({len(cities)}):")
        for tile in cities:
            mark = " [capital, +10/turn]" if tile.city_name == own_capital else ""
            lines.append(
                f"- {tile.city_name} at ({tile.q},{tile.r}): "
                f"resources={int(tile.resources)}{mark}"
            )
        lines.append(f"Your units ({len(units)}), each gets 2 action points per turn:")
        if units:
            for unit in units:
                location = unit.tile.city_name or f"({unit.tile.q},{unit.tile.r})"
                can_disband = (
                    " (can disband here)" if unit.tile.owner == agent_name and unit.tile.is_city else ""
                )
                lines.append(
                    f"- {unit.id} at {location}: resources={int(unit.resources)}, "
                    f"AP={unit.ap}{can_disband}"
                )
        else:
            lines.append("- (none)")
        lines.append(
            "Capital rules: your capital earns +10/turn, other cities +5/turn. "
            "Losing your capital costs each of your cities 2 and each unit 1 "
            "resource; recapturing it grants each city 5 and each unit 2. "
            "Disband a unit on your city to return its resources to that city "
            "(plus a 5 bonus). Moving onto an enemy unit triggers a battle: the "
            "side with more resources wins and loses nothing, the loser is "
            "destroyed."
        )

        cap = len(cities)
        used = self.spawns_this_turn.get(agent_name, 0)
        lines.append(
            f"Unit creation this turn: {used}/{cap} used "
            f"(max = number of cities you control). New units act immediately."
        )

        lines.append("All cities on the board:")
        for name, tile in sorted(self.cities.items()):
            original = CAPITAL_OF.get(name)
            mark = f" [capital of {original}]" if original else ""
            lines.append(
                f"- {name} at ({tile.q},{tile.r}): owner={tile.owner or 'neutral'}, "
                f"resources={int(tile.resources)}{mark}"
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
            "tiles": [
                {
                    "q": tile.q,
                    "r": tile.r,
                    "resources": int(tile.resources),
                    "owner": tile.owner,
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
