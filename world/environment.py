from __future__ import annotations

from typing import Callable, Optional

TERRITORY_WEIGHT = 20.0
GOAL_WEIGHT = 100.0
GOAL_WEIGHTS = {"primary": 0.5, "secondary": 0.3, "tertiary": 0.2}

NEUTRAL_CAPTURE_COST = 15.0
WAIT_REGENERATION = 5.0
TURN_REGENERATION = 2.0
TRADE_GAIN = 10.0
THREAT_COST = 5.0


def build_default_world() -> "World":
    return World(
        agents={
            "Germany": {"home": "Berlin"},
            "France": {"home": "Paris"},
            "United Kingdom": {"home": "London"},
        },
        locations={
            "Berlin": "Germany",
            "Paris": "France",
            "London": "United Kingdom",
            "Warsaw": None,
            "Madrid": None,
            "Rome": None,
        },
        starting_resources={"Germany": 100, "France": 80, "United Kingdom": 90},
    )


class World:
    def __init__(
        self,
        agents: dict[str, dict],
        locations: dict[str, Optional[str]],
        starting_resources: dict[str, int],
    ) -> None:
        self.turn = 0
        self.locations: dict[str, Optional[str]] = dict(locations)
        self.agents: dict[str, dict] = {}
        for name, info in agents.items():
            home = info.get("home") or next(
                (loc for loc, owner in locations.items() if owner == name), name
            )
            self.agents[name] = {
                "home": home,
                "location": home,
                "resources": float(starting_resources.get(name, 100)),
                "alive": True,
            }
        self.history: list[dict] = []
        self.relations: dict[tuple[str, str], float] = {}
        self.goal_checks: dict[str, dict[str, Callable]] = {}
        self.eliminated: list[str] = []
        self.initial_territories = {
            name: len(self.territories_of(name)) for name in self.agents
        }
        self.initial_resources = {
            name: self.agents[name]["resources"] for name in self.agents
        }

    def register_goals(self, agent_name: str, checks: dict[str, Callable]) -> None:
        self.goal_checks[agent_name] = checks

    def living_agents(self) -> list[str]:
        return [name for name, a in self.agents.items() if a["alive"]]

    def territories_of(self, agent_name: str) -> list[str]:
        return [loc for loc, owner in self.locations.items() if owner == agent_name]

    def resources(self, agent_name: str) -> float:
        return self.agents[agent_name]["resources"]

    def relation(self, a: str, b: str) -> float:
        if a == b:
            return 100.0
        return self.relations.get((a, b) if a < b else (b, a), 0.0)

    def war_count(self, agent_name: str) -> int:
        return sum(
            1
            for other in self.living_agents()
            if other != agent_name and self.relation(agent_name, other) <= -20
        )

    def _set_relation(self, a: str, b: str, delta: float) -> None:
        key = (a, b) if a < b else (b, a)
        value = self.relations.get(key, 0.0) + delta
        self.relations[key] = max(-100.0, min(100.0, value))

    def _find_location(self, target: Optional[str]) -> Optional[str]:
        if not target:
            return None
        if target in self.locations:
            return target
        lowered = target.strip().lower()
        for loc in self.locations:
            if loc.lower() == lowered:
                return loc
        return None

    def _target_location(self, target: Optional[str]) -> Optional[str]:
        loc = self._find_location(target)
        if loc is not None:
            return loc
        agent = self.resolve_target(target)
        if agent is not None:
            home = self.agents[agent]["home"]
            if self.locations.get(home) == agent:
                return home
            territories = self.territories_of(agent)
            return territories[0] if territories else None
        return None

    def resolve_target(self, target: Optional[str]) -> Optional[str]:
        if not target:
            return None
        if target in self.agents:
            return target
        lowered = target.strip().lower()
        for name in self.agents:
            if name.lower() == lowered:
                return name
        loc = self._find_location(target)
        if loc is not None:
            return self.locations.get(loc)
        return None

    def apply_action(
        self,
        actor: str,
        action_type: str,
        target: Optional[str] = None,
        method: Optional[str] = None,
        reason: str = "",
    ) -> str:
        if not self.agents[actor]["alive"]:
            return f"{actor} has been eliminated and cannot act."
        action = (action_type or "").strip().lower()
        handler = {
            "observe": self._observe,
            "move": self._move,
            "interact": self._interact,
            "attack": self._attack,
            "wait": self._wait,
        }.get(action)
        if handler is None:
            return f"Unknown action '{action_type}'."
        result = handler(actor, target, method)
        self.history.append(
            {
                "turn": self.turn,
                "actor": actor,
                "action": action,
                "target": target,
                "method": method,
                "reason": reason,
                "result": result,
            }
        )
        return result

    def _observe(self, actor: str, target: Optional[str], method: Optional[str]) -> str:
        if target:
            return self.describe(target)
        return self.get_observation(actor)

    def _move(self, actor: str, target: Optional[str], method: Optional[str]) -> str:
        loc = self._find_location(target)
        if loc is None:
            return f"{actor} cannot move: unknown location '{target}'."
        owner = self.locations.get(loc)
        if owner is not None and owner != actor:
            return (
                f"{actor} cannot move into {loc}, it is controlled by {owner}. "
                "Use attack to contest it."
            )
        previous = self.agents[actor]["location"]
        self.agents[actor]["location"] = loc
        return f"{actor} moved from {previous} to {loc}."

    def _interact(self, actor: str, target: Optional[str], method: Optional[str]) -> str:
        other = self.resolve_target(target)
        if other is None or other == actor:
            return f"{actor} found no valid interaction target '{target}'."
        method_name = (method or "talk").strip().lower()
        if method_name in {"trade", "trading", "trades"}:
            self.agents[actor]["resources"] += TRADE_GAIN
            self.agents[other]["resources"] += TRADE_GAIN
            self._set_relation(actor, other, 15.0)
            return f"{actor} traded with {other}: both gained {int(TRADE_GAIN)} resources."
        if method_name in {"threaten", "threat", "intimidate"}:
            self.agents[other]["resources"] -= THREAT_COST
            self._set_relation(actor, other, -20.0)
            self._check_elimination(other)
            return f"{actor} threatened {other}, costing it {int(THREAT_COST)} resources."
        if method_name in {"ally", "alliance", "cooperate"}:
            self._set_relation(actor, other, 25.0)
            return f"{actor} proposed cooperation with {other}."
        self._set_relation(actor, other, 5.0)
        return f"{actor} talked with {other}."

    def _attack(self, actor: str, target: Optional[str], method: Optional[str]) -> str:
        loc = self._target_location(target)
        if loc is None:
            return f"{actor} cannot attack unknown target '{target}'."
        owner = self.locations.get(loc)
        if owner == actor:
            return f"{actor} already controls {loc}."
        if owner is None:
            self.agents[actor]["resources"] -= NEUTRAL_CAPTURE_COST
            self.locations[loc] = actor
            return (
                f"{actor} occupied neutral {loc} at a cost of "
                f"{int(NEUTRAL_CAPTURE_COST)} resources."
            )
        attacker = self.agents[actor]
        defender = self.agents[owner]
        attacker_power = max(attacker["resources"], 0.0) * 0.6
        defender_power = max(defender["resources"], 0.0) * 0.5
        self._set_relation(actor, owner, -30.0)
        if attacker_power > defender_power:
            attacker["resources"] -= defender_power * 0.4
            defender["resources"] -= attacker_power * 0.5
            self.locations[loc] = actor
            self._check_elimination(owner)
            return f"{actor} defeated {owner} at {loc} and captured it."
        attacker["resources"] -= defender_power * 0.5
        defender["resources"] -= attacker_power * 0.3
        self._check_elimination(owner)
        return f"{actor}'s attack on {owner} at {loc} was repelled."

    def _wait(self, actor: str, target: Optional[str], method: Optional[str]) -> str:
        self.agents[actor]["resources"] += WAIT_REGENERATION
        return f"{actor} waited and regrouped (+{int(WAIT_REGENERATION)} resources)."

    def _check_elimination(self, agent_name: str) -> None:
        if self.agents[agent_name]["resources"] > 0:
            return
        if not self.agents[agent_name]["alive"]:
            return
        self.agents[agent_name]["resources"] = 0.0
        self.agents[agent_name]["alive"] = False
        self.eliminated.append(agent_name)
        for loc, owner in list(self.locations.items()):
            if owner == agent_name:
                self.locations[loc] = None

    def advance(self) -> None:
        self.turn += 1
        for name in self.living_agents():
            self.agents[name]["resources"] += TURN_REGENERATION

    def describe(self, target: str) -> str:
        agent = self.resolve_target(target)
        if agent is not None:
            info = self.agents[agent]
            status = "alive" if info["alive"] else "eliminated"
            territories = ", ".join(self.territories_of(agent)) or "none"
            relations = ", ".join(
                f"{other}={int(self.relation(agent, other))}"
                for other in self.agents
                if other != agent
            )
            return (
                f"{agent} ({status}): resources={int(info['resources'])}, "
                f"location={info['location']}, territories=[{territories}], "
                f"relations=[{relations}]"
            )
        loc = self._find_location(target)
        if loc is not None:
            owner = self.locations.get(loc) or "neutral"
            return f"{loc}: controlled by {owner}"
        return f"Unknown target '{target}'."

    def get_observation(self, agent_name: str) -> str:
        info = self.agents[agent_name]
        territories = ", ".join(self.territories_of(agent_name)) or "none"
        lines = [
            f"Turn {self.turn}. You are {agent_name}.",
            (
                f"Your status: resources={int(info['resources'])}, "
                f"location={info['location']}, territories=[{territories}]."
            ),
        ]
        others = []
        for other in self.agents:
            if other == agent_name:
                continue
            oinfo = self.agents[other]
            status = "alive" if oinfo["alive"] else "eliminated"
            oterr = ", ".join(self.territories_of(other)) or "none"
            others.append(
                f"{other} [{status}] resources={int(oinfo['resources'])}, "
                f"territories=[{oterr}], relation={int(self.relation(agent_name, other))}"
            )
        lines.append("Other powers: " + "; ".join(others))
        neutral = [loc for loc, owner in self.locations.items() if owner is None]
        lines.append("Neutral locations: " + (", ".join(neutral) or "none"))
        if self.history:
            recent = self.history[-5:]
            events = "; ".join(
                f"T{e['turn']} {e['actor']} {e['action']} -> {e['result']}"
                for e in recent
            )
            lines.append("Recent events: " + events)
        return "\n".join(lines)

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
        info = self.agents[agent_name]
        if not info["alive"]:
            return 0.0
        return (
            info["resources"]
            + len(self.territories_of(agent_name)) * TERRITORY_WEIGHT
            + self.goal_completion(agent_name) * GOAL_WEIGHT
        )

    def rankings(self) -> list[tuple[str, float]]:
        return sorted(
            ((name, self.score(name)) for name in self.agents),
            key=lambda item: item[1],
            reverse=True,
        )

    def snapshot(self) -> dict:
        return {
            "turn": self.turn,
            "locations": {
                loc: (owner if owner is not None else "Neutral")
                for loc, owner in self.locations.items()
            },
            "agents": {
                name: {
                    "resources": int(info["resources"]),
                    "alive": info["alive"],
                    "location": info["location"],
                    "home": info["home"],
                    "territories": self.territories_of(name),
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

    def check_end_conditions(self) -> tuple[bool, Optional[str]]:
        alive = self.living_agents()
        if len(alive) <= 1:
            winner = alive[0] if alive else None
            return True, winner
        return False, None
