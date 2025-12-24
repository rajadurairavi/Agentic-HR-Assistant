from fastapi import FastAPI
from pydantic import BaseModel
import logging

from langchain_core.messages import HumanMessage

# Import LangGraph agent
from app.agent import app as agent_app

# ----------------------------------
# Logging
# ----------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------
# FastAPI App
# ----------------------------------
app = FastAPI(
    title="Agentic HR Assistant",
    description="GenAI-powered HR chatbot using LangGraph, RAG, and Tools",
    version="1.0.0",
)

# ----------------------------------
# Request / Response Models
# ----------------------------------
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


# ----------------------------------
# Root Endpoint
# ----------------------------------
@app.get("/")
def root():
    return {
        "message": "✅ Agentic HR Assistant is running",
        "tech": ["LangGraph", "RAG", "FastAPI", "Streamlit"],
        "status": "OK",
    }


# ----------------------------------
# Health Check
# ----------------------------------
@app.get("/health")
def health():
    return {
        "service": "Agentic HR Assistant",
        "status": "healthy",
        "llm": "Groq",
        "vector_db": "FAISS",
    }


# ----------------------------------
# Ask Endpoint (FIXED)
# ----------------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask_hr_bot(request: QuestionRequest):
    logger.info(f"User question: {request.question}")

    result = agent_app.invoke(
        {
            "messages": [
                HumanMessage(content=request.question)  # ✅ IMPORTANT FIX
            ],
            "decision": "",
            "retries": 0,
            "tool_result": None,
        }
    )

    answer = result["messages"][-1].content
    return {"answer": answer}
