from langchain.tools import tool


# class BaseAction:
#     """
#     BaseAction is the base class of Action
#     """
#     def __init__(self, action, target=None, parameters=None, reason=None):
#         self.action = action
#         self.target = target
#         self.parameters = parameters or {}
#         self.reason = reason

#     def to_dict(self):
#         return {
#             "action": self.action,
#             "target": self.target,
#             "parameters": self.parameters,
#             "reason": self.reason
#         }

#     def __repr__(self):
#         return str(self.to_dict())
    

@tool
def observe(target: str, reason: str = "") -> str:
    """
    Observe an agent, location, or object.

    Args:
        target: The object or agent to observe
        reason: Why the agent is observing
    """
    # Here you could query environment state
    return f"Observed {target}. Reason: {reason}"

@tool
def move(target: str, reason: str = "") -> str:
    """
    Move the agent to a location or position.

    Args:
        target: Destination location
        reason: Why the agent is moving
    """
    return f"Moved to {target}. Reason: {reason}"

@tool
def interact(target: str, method: str, reason: str = "") -> str:
    """
    Interact with another agent or object.

    Args:
        target: The agent or object to interact with
        method: The type of interaction (talk, trade, threaten, etc.)
        reason: Why this interaction is happening
    """
    return f"Interacted with {target} via {method}. Reason: {reason}"

@tool
def attack(target: str, method: str = "direct", reason: str = "") -> str:
    """
    Attack a target agent or location.

    Args:
        target: The target of the attack
        method: Attack method (direct, ambush, sabotage)
        reason: Strategic reason for attack
    """
    return f"Attacked {target} via {method}. Reason: {reason}"

@tool
def wait(reason: str = "") -> str:
    """
    Wait / skip turn.

    Args:
        reason: Why the agent is waiting
    """
    return f"Waiting. Reason: {reason}"