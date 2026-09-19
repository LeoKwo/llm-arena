from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from typing import List
from agent_memory.memory_store import FAISSMemory
import operator


class AgentState(TypedDict):
    # action_taken: Annotated[List[AnyMessage], operator.add]
    memory: FAISSMemory
    goals: dict