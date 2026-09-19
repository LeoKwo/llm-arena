import json
import re

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agent.agent_state import AgentState
from agent.goals import format_goals
from agent.prompt import get_prompt
from agent.tools import make_tools

MAX_LLM_CALLS = 4
REFLECT_EVERY = 3
REFLECT_MIN_MEMORIES = 6


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _parse_action_from_text(content):
    text = _text(content)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except Exception:
        return None
    action = data.get("action") or data.get("name")
    if not action:
        return None
    args = data.get("args") or data.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    return {"action": str(action).lower(), "args": args}


def build_agent_graph(
    name, persona, goals, llm, world, memory, verbose=False, on_event=None
):
    tools = make_tools(world, name)
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    def emit_think(node, phase, text, turn):
        if on_event is not None:
            on_event(
                {
                    "type": "llm_" + phase,
                    "agent": name,
                    "node": node,
                    "turn": turn,
                    "text": text,
                }
            )

    def stream_text(prompt, node, turn):
        emit_think(node, "start", "", turn)
        parts = []
        for chunk in llm.stream(prompt):
            piece = _text(chunk.content)
            if piece:
                parts.append(piece)
                emit_think(node, "token", piece, turn)
        emit_think(node, "end", "", turn)
        return "".join(parts).strip()

    def observe_node(state):
        observation = world.get_observation(name)
        query = f"{observation}\nPrimary goal: {goals.get('primary', '')}"
        memory_context = memory.retrieve(query)
        return {
            "agent_name": name,
            "observation": observation,
            "memory_context": memory_context,
            "action": None,
            "result": None,
            "llm_calls": 0,
        }

    def reflect_node(state):
        turn = state.get("turn", 0)
        recent = memory.recent(8)
        prompt = (
            f"You are {name}. Persona: {persona}\n"
            f"Goals:\n{format_goals(goals)}\n\n"
            f"Recent memories:\n" + "\n".join(recent) + "\n\n"
            "Write a concise strategic reflection (3-4 sentences) about your "
            "position, mistakes, and what you should do next."
        )
        reflection = stream_text(prompt, "reflect", turn)
        memory.add_reflection(reflection)
        reflections = list(state.get("reflections", []))
        reflections.append(reflection)
        return {"reflections": reflections}

    def plan_node(state):
        turn = state.get("turn", 0)
        prompt = (
            f"You are {name}. Persona: {persona}\n"
            f"Goals:\n{format_goals(goals)}\n\n"
            f"Current situation:\n{state.get('observation', '')}\n\n"
            "Reflections:\n" + "\n".join(state.get("reflections", [])) + "\n\n"
            "Relevant memories:\n" + "\n".join(state.get("memory_context", [])) + "\n\n"
            "State a short plan (1-2 sentences) for this turn."
        )
        return {"plan": stream_text(prompt, "plan", turn)}

    def act_node(state):
        turn = state.get("turn", 0)
        system = get_prompt(name, persona, goals)
        context = (
            f"Situation:\n{state.get('observation', '')}\n\n"
            f"Plan for this turn: {state.get('plan', '')}\n\n"
            "Relevant memories:\n" + "\n".join(state.get("memory_context", [])) + "\n\n"
            "Choose exactly ONE action using the available tools. "
            "Provide a short reason."
        )
        messages = [SystemMessage(content=system), HumanMessage(content=context)]
        emit_think("act", "start", "", turn)
        full = None
        for chunk in llm_with_tools.stream(messages):
            piece = _text(chunk.content)
            if piece:
                emit_think("act", "token", piece, turn)
            full = chunk if full is None else full + chunk
        emit_think("act", "end", "", turn)
        if full is None:
            full = AIMessage(content="")
        return {"messages": [full], "llm_calls": state.get("llm_calls", 0) + 1}

    def route_after_act(state):
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None)
        if calls and state.get("llm_calls", 0) <= MAX_LLM_CALLS:
            return "tools"
        return "collect"

    def route_after_observe(state):
        turn = state.get("turn", 0)
        if (
            turn > 0
            and turn % REFLECT_EVERY == 0
            and memory.count() >= REFLECT_MIN_MEMORIES
        ):
            return "reflect"
        return "plan"

    def collect_node(state):
        messages = state.get("messages", [])
        action = None
        result = None
        for message in reversed(messages):
            if isinstance(message, ToolMessage) and result is None:
                result = _text(message.content)
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                calls = getattr(message, "tool_calls", None)
                if calls:
                    call = calls[0]
                    action = {
                        "action": call.get("name"),
                        "args": call.get("args", {}),
                        "reason": call.get("args", {}).get("reason", ""),
                    }
                    break
                parsed = _parse_action_from_text(message.content)
                if parsed is not None:
                    action = parsed
                    action["reason"] = action.get("args", {}).get("reason", "")
                    break
        if action is None:
            action = {
                "action": "wait",
                "args": {"reason": "no valid action"},
                "reason": "no valid action",
            }
        if verbose:
            print(f"  [{name}] action={action} result={result}", flush=True)
        return {"action": action, "result": result}

    builder = StateGraph(AgentState)
    builder.add_node("observe", observe_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("plan", plan_node)
    builder.add_node("act", act_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("collect", collect_node)

    builder.add_edge(START, "observe")
    builder.add_conditional_edges(
        "observe", route_after_observe, {"reflect": "reflect", "plan": "plan"}
    )
    builder.add_edge("reflect", "plan")
    builder.add_edge("plan", "act")
    builder.add_conditional_edges(
        "act", route_after_act, {"tools": "tools", "collect": "collect"}
    )
    builder.add_edge("tools", "collect")
    builder.add_edge("collect", END)

    return builder.compile()
