from langchain_ollama.chat_models import ChatOllama
from langchain.agents import create_agent
from agent.tools import observe, move, interact, attack, wait


def get_base_agent(model):
    return ChatOllama(model=model, temperature=0.1)



# def create_player_qwen3_5(player_prompt):
#     return create_agent(
#         model=qwen3_5,
#         tools=tools,
#         system_prompt=player_prompt
#     )

# def create_player_glm4(player_prompt):
#     return create_agent(
#         model=glm4,
#         tools=tools,
#         system_prompt=player_prompt
#     )

# def create_player_ds_r1(player_prompt):
#     return create_agent(
#         model=ds_r1,
#         tools=tools,
#         system_prompt=player_prompt
#     )



# class BaseAgent:
#     """
#     BaseAgent defines a LLM-powered agentic player in the game.
#     Each player has a name, a perona, goals and memory.
#     """
#     def __init__(self, name, persona, goals, llm=LLM):
#         self.name = name
#         self.persona = persona
#         self.goals = goals
#         self.llm = llm

#         self.memory = []
#         self.reflections = []

#     def observe(self, world_state):
#         observation = self._extract_relevant(world_state)
#         self.memory.append(observation)
#         return observation

#     def reflect(self):
#         reflection_prompt = self._build_reflection_prompt()
#         reflection = self.llm(reflection_prompt)
#         self.reflections.append(reflection)
#         return reflection

#     def plan(self):
#         plan_prompt = self._build_plan_prompt()
#         plan = self.llm(plan_prompt)
#         return plan

#     def act(self, world_state):
#         action_prompt = self._build_action_prompt(world_state)
#         action = self.llm(action_prompt)
#         return action
    
#     def take_turn(self, world_state):
#         observation = self.observe(world_state)
#         action = self.decide(observation)
#         self.memory.append({
#             "observation": observation,
#             "action": action
#         })
#         return action
    
#     def decide(self, observation):
#         prompt = f"""
#             Agent: {self.name}

#             Persona:
#             {self.persona}

#             Goals:
#             {self.goals}

#             Observation:
#             {observation}

#             Recent Memory:
#             {self.memory[-5:]}

#             Choose ONE action.

#             Available actions:
#             1. observe
#             2. move
#             3. interact
#             4. attack
#             5. wait

#             Respond with JSON.
#         """
#         response = self.llm(prompt)

#         action = parse_action(response)

#         # return action