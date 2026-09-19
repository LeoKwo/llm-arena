# from langgraph.graph import StateGraph, START, END
# from agent.agent_state import AgentState
# from agent.init_agents import 

# def llm_call(state: dict):
#     """LLM decides whether to call a tool or not"""

#     return {
#         "messages": [
#             model_with_tools.invoke(
#                 [
#                     SystemMessage(
#                         content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
#                     )
#                 ]
#                 + state["messages"]
#             )
#         ],
#         "llm_calls": state.get('llm_calls', 0) + 1
#     }

# agent_builder = StateGraph(AgentState)

# agent_builder.add_node("Observe", )

# observe, reflect, plan, act