import os
import logging
from typing import TypedDict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from app.rag_retriever import retrieve_documents
from app.hr_tools import get_leave_balance
from app.config import MODEL_NAME, TEMPERATURE, MAX_RETRIES

# -------------------------------------------------
# Logging setup (production friendly)
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# Agent State
# -------------------------------------------------
class AgentState(TypedDict):
    messages: List[BaseMessage]
    decision: str
    retries: int
    tool_result: Optional[dict]


# -------------------------------------------------
# LLM setup
# -------------------------------------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model=MODEL_NAME,
    temperature=TEMPERATURE
)


# -------------------------------------------------
# Decision Node (agent brain)
# -------------------------------------------------
def decision_node(state: AgentState):

    conversation = " ".join(
        msg.content.lower()
        for msg in state["messages"]
        if isinstance(msg, HumanMessage)
    )

    # Tool intent
    if "leave balance" in conversation or "remaining leave" in conversation:
        logger.info("Decision: TOOL")
        return {"decision": "tool"}

    # RAG intent
    has_country = "india" in conversation or "netherlands" in conversation
    has_leave_type = "annual" in conversation or "sick" in conversation

    if has_country and has_leave_type:
        logger.info("Decision: ANSWER (RAG)")
        return {"decision": "answer"}

    # Ask / fallback
    if state["retries"] < MAX_RETRIES:
        logger.info("Decision: ASK")
        return {"decision": "ask"}

    logger.info("Decision: FALLBACK")
    return {"decision": "fallback"}


# -------------------------------------------------
# Ask Follow-up Node
# -------------------------------------------------
def ask_followup_node(state: AgentState):

    followup = AIMessage(
        content="I still need the country (India / Netherlands) and leave type (annual / sick) to help you."
    )

    return {
        "messages": state["messages"] + [followup],
        "retries": state["retries"] + 1
    }


# -------------------------------------------------
# Answer Node (RAG + Guardrails)
# -------------------------------------------------
def answer_node(state: AgentState):

    user_question = state["messages"][-1].content

    docs = retrieve_documents(user_question)

    # Guardrail: no docs → no LLM
    if not docs:
        logger.info("No documents retrieved → I don't know")
        return {
            "messages": state["messages"] + [
                AIMessage(content="I don't know based on the available HR policy.")
            ]
        }

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an HR assistant.
Answer ONLY using the HR policy below.
If the answer is NOT found in the policy, reply exactly:
"I don't know based on the available HR policy."

HR Policy:
{context}

Question:
{user_question}
"""

    response = llm.invoke(prompt)

    return {
        "messages": state["messages"] + [response]
    }


# -------------------------------------------------
# Tool Node (backend action)
# -------------------------------------------------
def tool_node(state: AgentState):

    last_msg = state["messages"][-1].content.lower()

    if "e001" in last_msg:
        emp_id = "E001"
    elif "e002" in last_msg:
        emp_id = "E002"
    else:
        emp_id = "E001"

    result = get_leave_balance(emp_id)

    logger.info(f"Tool called for employee {emp_id}")

    return {
        "tool_result": result,
        "messages": state["messages"] + [
            AIMessage(content=f"Your leave balance is: {result}")
        ]
    }


# -------------------------------------------------
# Fallback Node
# -------------------------------------------------
def fallback_node(state: AgentState):

    fallback = AIMessage(
        content="I’m unable to proceed without the required details. Please contact HR for further assistance."
    )

    return {
        "messages": state["messages"] + [fallback]
    }


# -------------------------------------------------
# Build LangGraph
# -------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("decision", decision_node)
graph.add_node("ask", ask_followup_node)
graph.add_node("answer", answer_node)
graph.add_node("tool", tool_node)
graph.add_node("fallback", fallback_node)

graph.set_entry_point("decision")


def route_decision(state: AgentState):
    return state["decision"]


graph.add_conditional_edges(
    "decision",
    route_decision,
    {
        "ask": "ask",
        "answer": "answer",
        "tool": "tool",
        "fallback": "fallback"
    }
)

graph.add_edge("ask", END)
graph.add_edge("answer", END)
graph.add_edge("tool", END)
graph.add_edge("fallback", END)

app = graph.compile()


# -------------------------------------------------
# Interactive Chat Loop
# -------------------------------------------------
if __name__ == "__main__":

    messages = []
    retries = 0

    print("\nHR Bot started. Type 'exit' to quit.\n")

    while True:
        user_text = input("You: ")

        if user_text.lower() == "exit":
            print("Bot: Goodbye 👋")
            break

        messages.append(HumanMessage(content=user_text))

        result = app.invoke({
            "messages": messages,
            "decision": "",
            "retries": retries,
            "tool_result": None
        })

        last_msg = result["messages"][-1]
        print("Bot:", last_msg.content)

        messages = result["messages"]
        retries = result.get("retries", retries)

        if "unable to proceed" in last_msg.content.lower():
            break
