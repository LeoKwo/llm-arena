from agent.goals import format_goals

LANGUAGE_INSTRUCTIONS = {
    "en": "",
    "zh": (
        "Always write your response in Simplified Chinese (简体中文). "
        "Keep tool names and JSON field names in English."
    ),
}


def language_instruction(lang):
    return LANGUAGE_INSTRUCTIONS.get((lang or "en").lower(), "")


def get_prompt(name: str, persona: str, goals, lang: str = "en"):
    prompt = f"""You are {name}, a strategic actor in a turn-based geopolitical simulation.

Persona:
{persona}

Goals:
{format_goals(goals)}

Each turn you may take exactly one action using the available tools:
observe, move, interact, attack, or wait. Choose the action that best advances
your goals given the current world situation. Always provide a short reason."""
    extra = language_instruction(lang)
    if extra:
        prompt += "\n\n" + extra
    return prompt
