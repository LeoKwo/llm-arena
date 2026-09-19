from langchain.tools import tool

from world.hexmap import DIRECTIONS


def make_tools(world, agent_name):
    @tool
    def spawn_unit(city: str, amount: float, reason: str = "") -> str:
        """Create a new military unit by allocating resources from one of your cities.

        Each turn you may create at most as many units as the number of cities
        you control. New units act immediately and start with 2 action points.

        Args:
            city: Name of one of your cities to allocate resources from.
            amount: Resources to allocate (must not exceed that city's resources).
            reason: Why you are creating this unit.
        """
        return world.spawn_unit(agent_name, city, amount, reason)

    @tool
    def move_unit(unit_id: str, direction: str, reason: str = "") -> str:
        """Move one of your units to an adjacent hex tile. Costs 1 action point.

        You cannot enter a city you do not control (use attack_city instead).

        Args:
            unit_id: Id of your unit (for example "U1").
            direction: One of N, NE, SE, S, SW, NW.
            reason: Why you are moving.
        """
        return world.move_unit(agent_name, unit_id, direction, reason)

    @tool
    def attack_city(unit_id: str, city: str, reason: str = "") -> str:
        """Attack an enemy or neutral city within 1 tile. Costs 1 action point.

        You win if the unit's resources are greater than the city's resources;
        the city is then captured with half its resources. If the unit has fewer
        resources it is destroyed (its resources are lost).

        Args:
            unit_id: Id of your unit (for example "U1").
            city: Name of the city to attack (must be within 1 tile).
            reason: Why you are attacking.
        """
        return world.attack_city(agent_name, unit_id, city, reason)

    @tool
    def claim_tile(unit_id: str, reason: str = "") -> str:
        """Claim the unowned land tile your unit is standing on. Costs 2 action points.

        The unit gains the tile's resources and the tile becomes yours. The unit
        cannot move again this turn.

        Args:
            unit_id: Id of your unit (for example "U1").
            reason: Why you are claiming this tile.
        """
        return world.claim_tile(agent_name, unit_id, reason)

    @tool
    def garrison(unit_id: str, reason: str = "") -> str:
        """Garrison your unit in place. Costs 2 action points.

        If the tile is unclaimed land the unit gains 2 resources.

        Args:
            unit_id: Id of your unit (for example "U1").
            reason: Why you are holding position.
        """
        return world.garrison(agent_name, unit_id, reason)

    @tool
    def end_turn(reason: str = "") -> str:
        """End your turn. Call this when you have finished all your actions.

        Args:
            reason: Short summary of what you did this turn.
        """
        return world.end_turn(agent_name, reason)

    return [
        spawn_unit,
        move_unit,
        attack_city,
        claim_tile,
        garrison,
        end_turn,
    ]
