def get_persona(
        brief_description: str,
        strength: str,
        weakness: str
    ):
    return f"""
        {brief_description}

        Strength:
        {strength}

        Weakness:
        {weakness}
    """