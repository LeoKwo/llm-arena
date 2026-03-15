from typing import List
from agent.base_agent import BaseAgent

class SimulationEngine:

    def __init__(self, agents: List[BaseAgent], environment):
        self.agents = agents
        self.environment = environment
        self.turn = 0

    def run(self, max_turns=50):
        for _ in range(max_turns):
            print(f"\n--- Turn {self.turn} ---")
            for agent in self.agents:
                self.step_agent(agent)
            self.turn += 1

    def step_agent(self, agent):
        world_state = self.environment.get_world_state(agent)
        action = agent.take_turn(world_state)
        print(agent.name, "->", action)
        self.environment.apply_action(agent, action)