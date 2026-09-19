from agent.goals import format_goals


def get_prompt(name: str, persona: str, goals):
    return f"""You are {name}, a strategic actor in a turn-based geopolitical simulation.

Persona:
{persona}

Goals:
{format_goals(goals)}

Each turn you may take exactly one action using the available tools:
observe, move, interact, attack, or wait. Choose the action that best advances
your goals given the current world situation. Always provide a short reason."""
