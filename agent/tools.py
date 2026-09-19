from langchain.tools import tool


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
        If the destination holds enemy units a battle happens: the side with the
        higher total resources (plus +5 if it has a garrison) wins; the loser's
        units are destroyed and the winner loses 30% of the loser's strength.
        A tile is controlled while a unit with at least 1 resource stands on it.

        Args:
            unit_id: Id of your unit (for example "U1").
            direction: One of E, SE, SW, W, NW, NE.
            reason: Why you are moving.
        """
        return world.move_unit(agent_name, unit_id, direction, reason)

    @tool
    def attack_city(unit_id: str, city: str, reason: str = "") -> str:
        """Attack an enemy or neutral city within 1 tile. Costs 1 action point.

        The city's defense is its resources plus any units stationed there, plus
        +5 if it is garrisoned. If your unit is stronger you capture the city
        (looting 30%, halving it) and move in; the winner loses 30% of the
        defense. If your unit is weaker it is destroyed and the defenders take
        30% of your strength as damage.

        Args:
            unit_id: Id of your unit (for example "U1").
            city: Name of the city to attack (must be within 1 tile).
            reason: Why you are attacking.
        """
        return world.attack_city(agent_name, unit_id, city, reason)

    @tool
    def loot_tile(unit_id: str, reason: str = "") -> str:
        """Strip the land tile your unit stands on for its resources. Costs 2 action points.

        The unit gains the tile's resources; the tile is then empty. Land is not
        permanently owned: a tile is controlled only while one of your units
        stands on it.

        Args:
            unit_id: Id of your unit (for example "U1").
            reason: Why you are looting this tile.
        """
        return world.loot_tile(agent_name, unit_id, reason)

    @tool
    def merge_units(unit_id: str, other_id: str, reason: str = "") -> str:
        """Merge two of your units on the same tile into one. Costs 0 action points.

        Resources are combined and the merged unit keeps the lower of the two
        action-point totals (so merging cannot refresh a spent unit).

        Args:
            unit_id: The unit that will absorb the other (keeps its id).
            other_id: The unit that will be absorbed and removed.
            reason: Why you are merging.
        """
        return world.merge_units(agent_name, unit_id, other_id, reason)

    @tool
    def supply_unit(unit_id: str, city: str, amount: float, reason: str = "") -> str:
        """Send resources from one of your cities to a nearby unit. Costs 1 action point.

        The unit must be within 3 tiles of the city. The city keeps at least 5
        resources in reserve.

        Args:
            unit_id: Id of the unit to reinforce (for example "U1").
            city: Name of one of your cities to draw from.
            amount: Resources to transfer (must not exceed the city's spare resources).
            reason: Why you are supplying this unit.
        """
        return world.supply_unit(agent_name, unit_id, city, amount, reason)

    @tool
    def disband_unit(unit_id: str, reason: str = "") -> str:
        """Disband one of your units while it stands on a city you control.

        The unit's resources are returned to that city to strengthen its defense.

        Args:
            unit_id: Id of your unit (for example "U1").
            reason: Why you are disbanding this unit.
        """
        return world.disband_unit(agent_name, unit_id, reason)

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
        loot_tile,
        merge_units,
        supply_unit,
        disband_unit,
        end_turn,
    ]
