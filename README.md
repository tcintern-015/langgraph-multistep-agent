# Multi-Step AI Agent with LangGraph (Research + Calculator)

This extends the earlier Tool Calling / Function Calling agent into a real
multi-step workflow using **LangGraph**. Instead of one LLM call that
optionally invokes a single tool, the agent now runs a loop: think, act,
observe, think again, until it has enough information to answer.

## What changed from the previous assignment

The earlier project called a tool once and returned the result. This
version wraps that same idea in a graph so the agent can:

- call more than one tool in the same turn (search, then calculate)
- see the result of one tool call before deciding on the next step
- keep looping until it decides it's actually done, not just after one call
- recover when a tool fails instead of crashing

## Workflow

```
User Request -> [agent] -> needs a tool? -> [tools] -> back to [agent] -> ... -> final answer
```

- **State** (`AgentState` in `agent.py`): holds the running `messages` list
  (so tool results and prior turns are visible to every step) and a
  `steps_taken` counter used as a safety cap.
- **Nodes**:
  - `agent` - calls the LLM with the two tools bound to it; it either
    requests a tool call or writes a final answer.
  - `tools` - a LangGraph `ToolNode` that actually executes whichever
    tool(s) the agent asked for.
- **Edges**:
  - Entry point is `agent`.
  - `agent -> tools` or `agent -> END`, decided by `route_after_agent`
    (conditional routing based on whether the last message contains a
    tool call, and on the step cap).
  - `tools -> agent` always, so results feed back into the next reasoning
    step.
- **Tools** (`tools.py`):
  1. `web_search` - DuckDuckGo search, no API key required.
  2. `calculator` - safe arithmetic evaluator (AST-based, no `eval()`).

## Error handling (required: at least one graceful failure)

Both tools catch their own exceptions and return a string prefixed with
`SEARCH_ERROR:` or `CALC_ERROR:` instead of raising. The system prompt
tells the agent to treat those prefixes as a failed lookup, not as real
data, so it explains the problem to the user or adjusts its approach
instead of hallucinating a number. Try asking it `"What is 100 divided by
0?"` to see this path (included as an example in `agent.py`).

## Bonus items included

- **Conversation history**: `run_agent()` takes and returns the message
  list, so you can pass prior turns back in and the agent remembers
  earlier answers (see Example 3 in `agent.py`, which asks about a value
  from a previous turn).
- **Execution tracing**: `run_agent()` streams the graph with
  `stream_mode="values"` and prints each step (tool call requested, tool
  result, final answer) so you can see the path the graph actually took.
- **LangSmith**: set the `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` /
  `LANGCHAIN_PROJECT` variables in `.env` (see `.env.example`) and every
  run is automatically traced in LangSmith, no code changes needed.

## Setup (local, command line)

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your API key
python agent.py
```

By default `agent.py` uses `ChatGoogleGenerativeAI` (Gemini) since it has a
generous free tier, good for a class project. Get a key at
https://aistudio.google.com/apikey and put it in `.env` as `GOOGLE_API_KEY`.
If you were using OpenAI or Anthropic in your earlier assignment, swap the
"LLM setup" block at the top of `agent.py` to Option A or Option B instead.

## Run the web UI locally

```bash
streamlit run app.py
```

## Get a live link (Streamlit Community Cloud, free)

1. Push this whole folder to a public GitHub repo.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", pick your repo and branch, set the main file path to
   `app.py`.
4. Under "Advanced settings -> Secrets", add:
   ```
   GOOGLE_API_KEY = "your-gemini-key-here"
   ```
5. Click Deploy. Streamlit builds it from `requirements.txt` and gives you
   a public `https://<your-app-name>.streamlit.app` link, that's your live
   link for the submission.

## Files

| File | Purpose |
|---|---|
| `agent.py` | Graph definition: state, nodes, edges, conditional routing, runner |
| `tools.py` | `web_search` and `calculator` tool implementations |
| `app.py` | Streamlit chat UI for a live, deployable version of the agent |
| `requirements.txt` | Dependencies |
| `.env.example` | Template for API keys / LangSmith config |

## Example run

```
--- User: Search for the current population of Japan, then divide that number by 1000. ---
[agent] requested tool call(s): web_search({'query': 'current population of Japan'})
[tools] result: Title: Japan Population 2026...
[agent] requested tool call(s): calculator({'expression': '123000000 / 1000'})
[tools] result: 123000.0
[agent] final answer: Japan's current population is approximately 123 million...

--- User: What is 100 divided by 0? ---
[agent] requested tool call(s): calculator({'expression': '100 / 0'})
[tools] result: CALC_ERROR: division by zero is undefined.
[agent] final answer: That expression is undefined, division by zero has no numeric result...

--- User: What was the first number I asked you to look up? ---
[agent] final answer: You asked me to look up the population of Japan...
```
