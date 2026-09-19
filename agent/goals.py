def get_goals(primary, secondary, tertiary):
    return {
        "primary": primary,
        "secondary": secondary,
        "tertiary": tertiary,
    }


def format_goals(goals):
    if not isinstance(goals, dict):
        return str(goals)
    lines = []
    for level in ("primary", "secondary", "tertiary"):
        value = goals.get(level)
        if value:
            lines.append(f"- {level.capitalize()}: {value}")
    return "\n".join(lines)
