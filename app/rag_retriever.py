from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import FAISS_DIR


# -----------------------------------------
# Detect country from user question
# -----------------------------------------
def detect_country(question: str):

    q = question.lower()

    if "india" in q or "indian" in q:
        return "India"

    if "netherlands" in q or "dutch" in q:
        return "Netherlands"

    return None   # no country detected


# -----------------------------------------
# Load retriever with metadata filtering
# -----------------------------------------
def load_retriever():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


# -----------------------------------------
# Retrieve documents using metadata filter
# -----------------------------------------
def retrieve_documents(question: str):

    vector_store = load_retriever()

    country = detect_country(question)

    if country:
        # apply metadata filter
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": {"country": country}
            }
        )
    else:
        # fallback: no filter
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

    return retriever.invoke(question)
