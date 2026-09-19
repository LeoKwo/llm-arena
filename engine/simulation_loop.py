# from typing import List
# from agent.base_agent import BaseAgent

# class SimulationEngine:

#     def __init__(self, agents: List[BaseAgent], environment):
#         self.agents = agents
#         self.environment = environment
#         self.turn = 0

#     def run(self, max_turns=50):
#         for _ in range(max_turns):
#             print(f"\n--- Turn {self.turn} ---")
#             for agent in self.agents:
#                 self.step_agent(agent)
#             self.turn += 1

#     def step_agent(self, agent):
#         world_state = self.environment.get_world_state(agent)
#         action = agent.take_turn(world_state)
#         print(agent.name, "->", action)
#         self.environment.apply_action(agent, action)

from agent.init_agents import agents
from langchain_ollama.embeddings import OllamaEmbeddings
from agent_memory.memory_store import FAISSMemory

MAX_TURNS = 20

EMBED_MODEL = "herald/dmeta-embedding-zh"
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

for agent in agents:
    agent.memory = FAISSMemory(embeddings=embeddings, max_results=5)

for turn in range(MAX_TURNS):
    world_state["turn"] = turn
    print(f"\n--- Turn {turn} ---\n")

    for agent in agents:
        action = agent.take_turn(world_state)
        print(f"{agent.name} -> {action}")

        # Update world state based on action
        world_state = apply_action(world_state, agent, action)

    # Optionally: global events, check end conditions
    if check_end_conditions(world_state):
        break