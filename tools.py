"""
tools.py
Defines the two tools used by the multi step agent:

1. web_search   - looks up current information on the internet (DuckDuckGo, no API key needed)
2. calculator   - evaluates arithmetic expressions safely

Both tools are decorated with @tool so LangGraph / LangChain can bind them
to the LLM and call them automatically. Both tools catch their own errors
and return a readable error string instead of raising, so the graph can
keep running and the agent can react to the failure (this satisfies the
"handle at least one failure gracefully" requirement).
"""

import ast
import operator

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Tool 1: Web Search
# ---------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the web for up to date information on a topic.

    Use this when the user asks about facts, current events, prices,
    people, or anything you are not confident about from memory.

    Args:
        query: The search query string.
    """
    try:
        try:
            from ddgs import DDGS  # new, maintained package
        except ImportError:
            from duckduckgo_search import DDGS  # fallback for older installs

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return f"No search results found for '{query}'."

        formatted = "\n\n".join(
            f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}"
            for r in results
        )
        return formatted

    except Exception as exc:
        # Graceful failure: return a clear error string rather than crashing
        # the graph. The agent node will see this in the tool result and can
        # decide to retry, try a different query, or apologize to the user.
        return f"SEARCH_ERROR: could not complete web search for '{query}'. Reason: {exc}"


# ---------------------------------------------------------------------------
# Tool 2: Calculator
# ---------------------------------------------------------------------------
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the numeric result.

    Supports +, -, *, /, %, ** and parentheses. Use this for any math the
    user needs done, especially after a web_search returns numbers that
    need to be combined or compared.

    Args:
        expression: A math expression, e.g. "12 * (3 + 4) / 2".
    """
    try:
        parsed = ast.parse(expression, mode="eval").body
        result = _safe_eval(parsed)
        return str(result)
    except ZeroDivisionError:
        return "CALC_ERROR: division by zero is undefined."
    except Exception as exc:
        return f"CALC_ERROR: could not evaluate '{expression}'. Reason: {exc}"


TOOLS = [web_search, calculator]
