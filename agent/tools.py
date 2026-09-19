from langchain.tools import tool

@tool
def observe(state: dict):
    """
    Observe an agent, location, or object.

    Args:
        target: The object or agent to observe
        reason: Why the agent is observing
    """
    # Here you could query environment state
    return f"Observed {target}. Reason: {reason}"

@tool
def move(state: dict):
    """
    Move the agent to a location or position.

    Args:
        target: Destination location
        reason: Why the agent is moving
    """
    return f"Moved to {target}. Reason: {reason}"

@tool
def interact(state: dict):
    """
    Interact with another agent or object.

    Args:
        target: The agent or object to interact with
        method: The type of interaction (talk, trade, threaten, etc.)
        reason: Why this interaction is happening
    """
    return f"Interacted with {target} via {method}. Reason: {reason}"

@tool
def attack(state: dict):
    """
    Attack a target agent or location.

    Args:
        target: The target of the attack
        method: Attack method (direct, ambush, sabotage)
        reason: Strategic reason for attack
    """
    return f"Attacked {target} via {method}. Reason: {reason}"

@tool
def wait(state: dict):
    """
    Wait / skip turn.

    Args:
        reason: Why the agent is waiting
    """
    return f"Waiting. Reason: {reason}"