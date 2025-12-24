# 🤖 Agentic HR Assistant
A production-style GenAI application using Agentic AI + RAG

---
## 📌 What is this project?

Agentic HR Assistant is a **GenAI-powered HR chatbot** that answers HR policy questions
(like annual leave, sick leave, country-specific rules) using:

- **Agentic AI (LangGraph)** for decision making
- **RAG (FAISS + embeddings)** for grounded answers
- **FastAPI** for backend APIs
- **Streamlit** for interactive UI

The same **single agent (brain)** is reused across UI and API.
---

## 🧠 Why Agentic + RAG?

- LLM alone can hallucinate
- Agent decides **what to do**
- RAG provides **verified knowledge**
- LLM only generates the final response

> LLM is NOT the brain.  
> Agent is the brain.

---

## 🧰 Tech Stack

### Core GenAI
- **LangGraph** – Agentic workflow
- **LangChain Core** – Message handling
- **Groq LLM** – Text generation
- **FAISS** – Vector database
- **HuggingFace Embeddings** – Text embeddings

### Backend & UI
- **FastAPI** – REST API
- **Streamlit** – Web UI

### Language & Runtime
- **Python 3.10+**

---

## 📦 Python Packages Used

Create a `requirements.txt` file with the following content:

```txt
fastapi
uvicorn
streamlit
langgraph
langchain
langchain-core
langchain-community
langchain-huggingface
langchain-groq
faiss-cpu
pydantic
python-dotenv

⚙️ Project Setup (Step-by-Step)
1️⃣ Clone Repository
git clone <your-repo-url>
cd Agentic01

2️⃣ Create Virtual Environment
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Mac / Linux

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Environment Variables

Create a .env file in project root:

GROQ_API_KEY=your_groq_api_key_here

📂 Project Structure & File Purpose
Agentic01/
│
├── api/
│   └── api.py
│       # FastAPI backend
│       # Exposes /ask, /health endpoints
│
├── app/
│   ├── agent.py
│   │   # LangGraph agent (core brain)
│   │   # Handles decision-making and flow
│   │
│   ├── rag_store.py
│   │   # Builds FAISS index (run once or when data changes)
│   │
│   ├── rag_retriever.py
│   │   # Loads FAISS index and retrieves documents
│   │
│   ├── hr_tools.py
│   │   # Tool functions (example: leave balance lookup)
│   │
│   └── config.py
│       # Central configuration
│       # Project root, paths, model configs
│
├── ui/
│   └── ui.py
│       # Streamlit UI
│       # Uses the same agent as FastAPI
│
├── knowledgebase/
│   ├── data/
│   │   # Raw HR documents (txt files)
│   │
│   └── faiss_index/
│       # Vector DB files (generated)
│
├── requirements.txt
├── README.md
└── venv/

📚 Knowledge Base
Documents Location
knowledgebase/data/


Example files:

hr_leave_policy_india.txt

hr_leave_policy_netherlands.txt

hr_general_policy.txt

Each document is:

Chunked

Embedded

Stored in FAISS with metadata

🔁 RAG Lifecycle
1️⃣ Build FAISS Index (Run Once)

Run this only when documents change:

python app/rag_store.py


This will:

Load documents

Generate embeddings

Store FAISS index in knowledgebase/faiss_index

2️⃣ Runtime Retrieval

At runtime:

FAISS index is loaded

Relevant documents are retrieved

Context is passed to the agent

Used by:

FastAPI

Streamlit

🧠 Agent Workflow (LangGraph)
Agent Decisions

The agent decides between:

ASK → Missing information

ANSWER → Use RAG

TOOL → Call backend tool

Agent State Includes:

Messages

Decision

Retry count

Tool output

This makes the system:

Deterministic

Explainable

Safe for enterprise usage

🚀 Running the Application
1️⃣ Start FastAPI Backend
uvicorn api.api:app --reload


Check:

API Docs → http://127.0.0.1:8000/docs

Health → http://127.0.0.1:8000/health

2️⃣ Start Streamlit UI

⚠️ Important (PYTHONPATH fix):

set PYTHONPATH=.
streamlit run ui/ui.py


UI:

http://localhost:8501

🔌 FastAPI Usage
Ask Endpoint
POST /ask


Payload:

{
  "question": "India annual leave"
}


Response:

{
  "answer": "Employees in India are entitled to 24 days of annual leave per year."
}

🧪 Common Issues & Fixes
❌ Streamlit works but FastAPI does not

✔ Ensure FastAPI sends HumanMessage, not raw dicts

❌ FAISS not retrieving documents

✔ Rebuild index after moving knowledgebase

python app/rag_store.py

🎯 Why This Project Is Interview-Ready

Uses Agentic AI (LangGraph)

Proper RAG implementation

Clean project architecture

Same agent reused across UI & API

Real-world debugging experience

🏁 Final Note

This is not a demo project.

It is a real GenAI system with:

Decision-making

Knowledge grounding

Production-style structure

Agent decides.
Knowledge grounds.
LLM speaks.



👑 You are a GenAI Engineer.


---


