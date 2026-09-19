from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from typing import List, Optional


class AgentState(TypedDict, total=False):
    agent_name: str
    turn: int
    observation: str
    memory_context: List[str]
    reflections: List[str]
    plan: str
    action: Optional[dict]
    result: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int
