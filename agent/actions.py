class BaseAction:
    """
    BaseAction is the base class of Action
    """
    def __init__(self, action, target=None, parameters=None, reason=None):
        self.action = action
        self.target = target
        self.parameters = parameters or {}
        self.reason = reason

    def to_dict(self):
        return {
            "action": self.action,
            "target": self.target,
            "parameters": self.parameters,
            "reason": self.reason
        }

    def __repr__(self):
        return str(self.to_dict())
    

class Observe(BaseAction):
    def __init__(self, target, reason=None):
        super().__init__(
            action="observe",
            target=target,
            reason=reason
        )

class Move(BaseAction):
    def __init__(self, target, reason=None):
        super().__init__(
            action="move",
            target=target,
            reason=reason
        )

class Interact(BaseAction):
    def __init__(self, target, method, reason=None):
        super().__init__(
            action="interact",
            target=target,
            parameters={
                "method": method
            },
            reason=reason
        )

class Attack(BaseAction):
    def __init__(self, target, method="direct", reason=None):
        super().__init__(
            action="attack",
            target=target,
            parameters={
                "method": method
            },
            reason=reason
        )

class Wait(BaseAction):
    def __init__(self, reason=None):
        super().__init__(
            action="wait",
            reason=reason
        )

