from agent.base_agent import BaseAgent
from agent.persona import get_persona
from agent.goals import get_goals

germany = BaseAgent(
    name="Germany",
    persona=get_persona(
        brief_description="Expansionist aggressive state",
        strength="Strong domestic support for war",
        weakness="Can't sustain long-term war because of limited resources"
    ),
    goals=get_goals(
        primary="Expand territory",
        secondary="Avoid multi front war",
        tertiary="Maintain strong economy"
    )
)

france = BaseAgent(
    name="France",
    persona=get_persona(
        brief_description="Cautious defensive European power",
        strength="Strong defensive military and fortified borders",
        weakness="Political divisions and reluctance to start offensive wars"
    ),
    goals=get_goals(
        primary="Preserve national sovereignty",
        secondary="Prevent German expansion",
        tertiary="Maintain alliance stability"
    )
)

united_kingdom = BaseAgent(
    name="United Kingdom",
    persona=get_persona(
        brief_description="Island naval power focused on balance of power in Europe",
        strength="Powerful navy and strong economy",
        weakness="Limited willingness for large land wars in Europe"
    ),
    goals=get_goals(
        primary="Maintain balance of power in Europe",
        secondary="Prevent domination of the continent by any single power",
        tertiary="Protect global trade and colonies"
    )
)

agents = [
    germany, france, united_kingdom
]