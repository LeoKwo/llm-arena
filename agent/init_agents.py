# from agent.base_agent import create_player_ds_r1, create_player_glm4, create_player_qwen3_5
from agent.base_agent import get_base_agent
from langchain_ollama.chat_models import ChatOllama
from agent.persona import get_persona
from agent.goals import get_goals
from agent.prompt import get_prompt
from langgraph.graph import StateGraph, START, END
from agent.agent_state import AgentState
from agent.tools import observe, move, interact, attack, wait
from langchain.tools import BaseTool
from typing import Tuple
from agent_memory.memory_store import FAISSMemory

tools = observe, move, interact, attack, wait

def agent_builder(tools: Tuple[BaseTool], baseAgent: ChatOllama):
    agent_executor = StateGraph(AgentState)
    agent_executor.add_node("Observe", tools[0])



# germany = create_player_ds_r1(
#     player_prompt=get_prompt(
#         name="Germany",
#         persona=get_persona(
#             brief_description="Expansionist aggressive state",
#             strength="Strong domestic support for war",
#             weakness="Can't sustain long-term war because of limited resources"
#         ),
#         goals=get_goals(
#             primary="Expand territory",
#             secondary="Avoid multi front war",
#             tertiary="Maintain strong economy"
#         )
#     )
# )


# france = create_player_glm4(
#     player_prompt=get_prompt(
#         name="France",
#         persona=get_persona(
#             brief_description="Cautious defensive European power",
#             strength="Strong defensive military and fortified borders",
#             weakness="Political divisions and reluctance to start offensive wars"
#         ),
#         goals=get_goals(
#             primary="Preserve national sovereignty",
#             secondary="Prevent German expansion",
#             tertiary="Maintain alliance stability"
#         )
#     )
# )

# united_kingdom = create_player_qwen3_5(
#     player_prompt=get_prompt(
#         name="United Kingdom",
#         persona=get_persona(
#             brief_description="Island naval power focused on balance of power in Europe",
#             strength="Powerful navy and strong economy",
#             weakness="Limited willingness for large land wars in Europe"
#         ),
#         goals=get_goals(
#             primary="Maintain balance of power in Europe",
#             secondary="Prevent domination of the continent by any single power",
#             tertiary="Protect global trade and colonies"
#         )
#     )
# )

# agents = [
#     germany, france, united_kingdom
# ]