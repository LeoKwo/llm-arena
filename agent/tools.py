from langchain.tools import tool


def make_tools(world, agent_name):
    @tool
    def observe(target: str = "", reason: str = "") -> str:
        """Observe another power, a location, or the whole world.

        Args:
            target: The power or location to observe (empty = whole world)
            reason: Why you are observing
        """
        return world.apply_action(
            agent_name, "observe", target=target or None, reason=reason
        )

    @tool
    def move(target: str, reason: str = "") -> str:
        """Move your forces to a friendly or neutral location.

        Args:
            target: Destination location name
            reason: Why you are moving
        """
        return world.apply_action(agent_name, "move", target=target, reason=reason)

    @tool
    def interact(target: str, method: str = "talk", reason: str = "") -> str:
        """Interact with another power.

        Args:
            target: The power to interact with
            method: One of talk, trade, threaten, ally
            reason: Why you are interacting
        """
        return world.apply_action(
            agent_name, "interact", target=target, method=method, reason=reason
        )

    @tool
    def attack(target: str, method: str = "direct", reason: str = "") -> str:
        """Attack an enemy power or contest a location.

        Args:
            target: The power or location to attack
            method: Attack method (direct, ambush, sabotage)
            reason: Strategic reason for attacking
        """
        return world.apply_action(
            agent_name, "attack", target=target, method=method, reason=reason
        )

    @tool
    def wait(reason: str = "") -> str:
        """Wait this turn to regroup and regenerate resources.

        Args:
            reason: Why you are waiting
        """
        return world.apply_action(agent_name, "wait", reason=reason)

    return [observe, move, interact, attack, wait]
