"""
app.py
Streamlit chat UI wrapping the LangGraph agent so it can be deployed as a
live link (e.g. Streamlit Community Cloud).

Run locally:
    streamlit run app.py

Deploy (free):
    1. Push this repo to GitHub.
    2. Go to https://share.streamlit.io, sign in with GitHub.
    3. "New app" -> pick this repo -> main file path: app.py
    4. In "Advanced settings -> Secrets", add:
         GOOGLE_API_KEY = "your-gemini-key-here"
    5. Deploy. Streamlit gives you a public https link, that's your live link.
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import graph, SYSTEM_PROMPT

st.set_page_config(page_title="Research + Calculator Agent", page_icon="🧠")
st.title("🧠 Research + Calculator Agent (LangGraph)")
st.caption("Multi-step agent: web_search + calculator tools, powered by LangGraph.")

if "messages" not in st.session_state:
    st.session_state.messages = []  # conversation history (bonus requirement)

# Render prior conversation (only human/final-AI turns, not internal tool chatter)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
        st.chat_message("assistant").write(msg.content)

user_input = st.chat_input("Ask something that needs a search and/or some math...")

if user_input:
    st.chat_message("user").write(user_input)
    state = {
        "messages": st.session_state.messages + [HumanMessage(content=user_input)],
        "steps_taken": 0,
    }

    with st.status("Working through the steps...", expanded=True) as status:
        final_state = state
        for step in graph.stream(state, stream_mode="values"):
            final_state = step
            last = step["messages"][-1]
            if getattr(last, "tool_calls", None):
                for c in last.tool_calls:
                    st.write(f"🔧 Calling `{c['name']}` with `{c['args']}`")
            elif isinstance(last, ToolMessage):
                preview = str(last.content)[:300]
                st.write(f"📥 Tool result: {preview}")
        status.update(label="Done", state="complete", expanded=False)

    st.session_state.messages = final_state["messages"]
    final_answer = final_state["messages"][-1].content
    st.chat_message("assistant").write(final_answer)

with st.sidebar:
    st.subheader("About")
    st.write(
        "This agent decides between two tools, a web search and a "
        "calculator, chaining them together across multiple steps using "
        "a LangGraph state machine. Errors from either tool (like divide "
        "by zero) are caught and explained instead of crashing the app."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
