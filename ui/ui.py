import streamlit as st
from langchain_core.messages import HumanMessage
from app.agent import app as agent_app   # reuse LangGraph agent (single brain)


# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Agentic HR Assistant",
    layout="centered"
)

st.title("🤖 Agentic HR Assistant")
st.caption("Powered by LangGraph, RAG, Tools & Guardrails")


# -------------------------------------------------
# Example prompts (helps interviewer/demo users)
# -------------------------------------------------
with st.expander("💡 Try example questions"):
    st.markdown(
        """
        - **India annual leave**  
        - **Netherlands sick leave**  
        - **Check my leave balance for E001**
        """
    )


# -------------------------------------------------
# Session state (store chat history)
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------------------------
# Display chat history
# -------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

        # show source if available
        if "source" in msg:
            st.caption(f"📄 Source: {msg['source']}")


# -------------------------------------------------
# Chat input
# -------------------------------------------------
user_input = st.chat_input("Ask HR related questions...")

if user_input:

    # display user message
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # invoke agent (LangGraph)
    result = agent_app.invoke({
        "messages": [HumanMessage(content=user_input)],
        "decision": "",
        "retries": 0,
        "tool_result": None
    })

    answer = result["messages"][-1].content

    # infer source (simple & effective for demo)
    source = None
    question_lower = user_input.lower()

    if "india" in question_lower:
        source = "HR Leave Policy – India"
    elif "netherlands" in question_lower:
        source = "HR Leave Policy – Netherlands"
    elif "leave balance" in question_lower:
        source = "HR Backend System (Tool)"

    # display assistant message
    st.chat_message("assistant").write(answer)

    if source:
        st.caption(f"📄 Source: {source}")

    # store assistant message
    msg_data = {
        "role": "assistant",
        "content": answer
    }

    if source:
        msg_data["source"] = source

    st.session_state.messages.append(msg_data)


# -------------------------------------------------
# Footer (simple & professional)
# -------------------------------------------------
st.markdown("---")
st.caption("Built by Raja | GenAI Engineer")

# Diagram
st.markdown("## 🏗️ System Architecture")

st.components.v1.html(
    """
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <div class="mermaid">
    flowchart LR
        U[👤 HR User] --> UI[🎨 Streamlit UI]
        U --> API[⚙️ FastAPI]

        UI --> AGENT[🧠 LangGraph Agent]
        API --> AGENT

        AGENT -->|Need Context| RET[🔎 RAG Retriever]
        RET --> VDB[(📦 FAISS Vector DB)]
        KB[📚 HR Policies] --> VDB

        RET --> AGENT
        AGENT --> LLM[🤖 Groq LLM]
        LLM --> AGENT

        AGENT --> UI
        AGENT --> API
    </div>

    <script>
        mermaid.initialize({ startOnLoad: true });
    </script>
    """,
    height=450,
)
