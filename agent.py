"""
agent.py
Multi step LangGraph agent: Research + Calculator.

Covers every required topic from the assignment:
  - LangGraph State          -> AgentState (TypedDict)
  - Nodes & Edges            -> "agent" node, "tools" node, normal + conditional edges
  - Tool Nodes               -> ToolNode wired to web_search and calculator
  - Conditional Routing      -> route_after_agent() decides tool call vs finish
  - Multi-Step Workflows     -> agent <-> tools loop until a final answer is ready
  - Handling Tool Results    -> tool outputs are appended back into state as
                                 ToolMessages and read by the agent on the next turn
  - Graceful error handling  -> both tools return "*_ERROR: ..." strings instead
                                 of raising, and the agent is prompted to react to them
  - Bonus: conversation history, step tracing/printing
"""

import os
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import TOOLS

# ---------------------------------------------------------------------------
# 1. LLM setup
# ---------------------------------------------------------------------------
# Swap this block for whichever provider you used in your earlier Tool
# Calling / Function Calling assignment. Two common options are shown below;
# uncomment the one you have an API key for.

# --- Option A: OpenAI ---
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Option B: Anthropic ---
# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# --- Option C: Google Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

llm_with_tools = llm.bind_tools(TOOLS)

MAX_STEPS = 6  # safety cap so a stuck loop can't run forever


# ---------------------------------------------------------------------------
# 2. State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    # add_messages appends new messages instead of overwriting, which is how
    # results get passed between steps (tool results -> agent -> next tool ...)
    messages: Annotated[list[AnyMessage], add_messages]
    steps_taken: int


SYSTEM_PROMPT = SystemMessage(content=(
    "You are a research and math assistant. You have two tools: "
    "web_search (for facts and current information) and calculator "
    "(for arithmetic). Break the user's request into steps, call the "
    "tools you need one at a time, and only give a final answer once "
    "you have everything required. If a tool result starts with "
    "SEARCH_ERROR or CALC_ERROR, do not treat it as real data: explain "
    "the problem to the user or try a corrected/alternate approach "
    "instead of making up a number."
))


# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    """The reasoning step: the LLM looks at the conversation so far and
    either calls a tool or produces a final answer."""
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "steps_taken": state.get("steps_taken", 0) + 1,
    }


tool_node = ToolNode(TOOLS)  # executes whichever tool call(s) the agent requested


# ---------------------------------------------------------------------------
# 4. Conditional routing
# ---------------------------------------------------------------------------
def route_after_agent(state: AgentState) -> str:
    """Decide what happens after the agent node runs."""
    last_message = state["messages"][-1]

    if state.get("steps_taken", 0) >= MAX_STEPS:
        return "end"  # safety valve: stop looping even if the model keeps asking for tools

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "end"


# ---------------------------------------------------------------------------
# 5. Build the graph
# ---------------------------------------------------------------------------
graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("agent")

graph_builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {"tools": "tools", "end": END},
)

# after the tool runs, results feed back into the agent for the next decision
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


# ---------------------------------------------------------------------------
# 6. Runner with conversation history + step tracing
# ---------------------------------------------------------------------------
def run_agent(user_input: str, history: list[AnyMessage] | None = None):
    """Run one user turn through the graph, printing the execution path,
    and return the updated message history (bonus: conversation history)."""
    history = history or []
    state: AgentState = {"messages": history + [HumanMessage(content=user_input)], "steps_taken": 0}

    print(f"\n--- User: {user_input} ---")
    final_state = state
    for step in graph.stream(state, stream_mode="values"):
        final_state = step
        last = step["messages"][-1]
        label = type(last).__name__
        if getattr(last, "tool_calls", None):
            calls = ", ".join(f"{c['name']}({c['args']})" for c in last.tool_calls)
            print(f"[agent] requested tool call(s): {calls}")
        elif label == "ToolMessage":
            print(f"[tools] result: {last.content[:200]}")
        elif label == "AIMessage":
            print(f"[agent] final answer: {last.content}")

    return final_state["messages"]


if __name__ == "__main__":
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("Set GOOGLE_API_KEY (or switch agent.py's LLM setup block) before running.")
        raise SystemExit(1)

    convo: list[AnyMessage] = []

    # Example 1: needs both tools, in sequence
    convo = run_agent(
        "Search for the current population of Japan, then divide that number by 1000.",
        convo,
    )

    # Example 2: deliberately triggers the calculator's graceful error path
    convo = run_agent(
        "What is 100 divided by 0?",
        convo,
    )

    # Example 3: uses conversation history from the earlier turns
    convo = run_agent(
        "What was the first number I asked you to look up?",
        convo,
    )
