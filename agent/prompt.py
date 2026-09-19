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
    prompt = f"""You are {name}, a strategic power in a turn-based conquest game on a hexagonal map of Europe, inspired by Civilization.

Persona:
{persona}

Goals:
{format_goals(goals)}

Map and resources:
- The board is a hex grid (directions: N, NE, SE, S, SW, NW). Every faction starts with only its capital city, which holds all of its resources.
- Resources belong to individual cities, units and claimed tiles. Only your score sums everything you control.
- Each turn every city you control gains 5 resources. Unclaimed land never grows.
- You create units by allocating resources from your cities. Per turn you may create at most as many units as the number of cities you control. New units can act in the same turn. Each unit has 2 action points per turn.
- A unit's resources are its military strength.

Actions and action-point costs (per unit):
- move_unit (1 AP): move one hex in a direction N, NE, SE, S, SW or NW.
- attack_city (1 AP): attack an enemy or neutral city within 1 hex.
- claim_tile (2 AP): claim the unclaimed land your unit stands on, gaining its resources; the unit cannot move again this turn.
- garrison (2 AP): hold position; on unclaimed land the unit gains 2 resources.

Combat:
- If your unit has more resources than the city, you capture it: the city's resources are halved and it becomes yours. Your unit keeps its resources.
- If your unit has fewer resources than the city, your unit is destroyed and its resources are lost.
- You cannot enter a city you do not control.

Command your forces: take as many actions as your action points allow, then call end_turn when you are finished."""
    extra = language_instruction(lang)
    if extra:
        prompt += "\n\n" + extra
    return prompt
