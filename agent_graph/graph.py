from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agent.agent_state import AgentState
from agent.goals import format_goals
from agent.prompt import get_prompt, language_instruction
from agent.tools import make_tools

MAX_LLM_CALLS = 24
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


def build_agent_graph(
    name, persona, goals, llm, world, memory, verbose=False, on_event=None, lang="en"
):
    tools = make_tools(world, name)
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    _lang_extra = language_instruction(lang)
    lang_suffix = ("\n\n" + _lang_extra) if _lang_extra else ""

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
        max_turns = getattr(world, "max_turns", None)
        time_note = ""
        if max_turns:
            remaining = max(0, max_turns - (turn + 1))
            time_note = f" Only {remaining} turns remain after this one."
        prompt = (
            f"You are {name}. Persona: {persona}\n"
            f"Goals:\n{format_goals(goals)}\n\n"
            f"Recent memories:\n" + "\n".join(recent) + "\n\n"
            "Write a concise strategic reflection (3-4 sentences) about your "
            "position, mistakes, and what you should do next." + time_note + lang_suffix
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
            "State a short plan (1-2 sentences) for this turn." + lang_suffix
        )
        return {"plan": stream_text(prompt, "plan", turn)}

    def act_node(state):
        turn = state.get("turn", 0)
        system = get_prompt(name, persona, goals, lang)
        context = (
            f"Situation:\n{state.get('observation', '')}\n\n"
            f"Plan for this turn: {state.get('plan', '')}\n\n"
            "Relevant memories:\n" + "\n".join(state.get("memory_context", [])) + "\n\n"
            "Issue one or more actions using the available tools, then call "
            "end_turn when you are done. Units have 2 action points each; you "
            "may create new units and command them in the same turn." + lang_suffix
        )
        history = list(state.get("messages", []))
        messages = [SystemMessage(content=system), HumanMessage(content=context)] + history
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
        messages = state.get("messages", [])
        if not messages:
            return "collect"
        last = messages[-1]
        calls = getattr(last, "tool_calls", None)
        if not calls:
            return "collect"
        if any(call.get("name") == "end_turn" for call in calls):
            return "collect"
        if state.get("llm_calls", 0) >= MAX_LLM_CALLS:
            return "collect"
        return "tools"

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
        results_by_id = {}
        for message in messages:
            if isinstance(message, ToolMessage):
                results_by_id[message.tool_call_id] = _text(message.content)

        actions = []
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            for call in getattr(message, "tool_calls", None) or []:
                tool_name = call.get("name")
                if tool_name == "end_turn":
                    continue
                args = call.get("args", {}) or {}
                actions.append(
                    {
                        "action": tool_name,
                        "args": args,
                        "reason": args.get("reason", ""),
                        "result": results_by_id.get(call.get("id"), ""),
                    }
                )
        if not actions:
            return {"actions": [], "action": None, "result": None}
        last = actions[-1]
        return {"actions": actions, "action": last, "result": last.get("result")}

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
    builder.add_edge("tools", "act")
    builder.add_edge("collect", END)

    return builder.compile()
