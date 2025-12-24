import os                                           # file handling
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import DATA_DIR, FAISS_DIR

"""
This script is executed ONLY when documents change.
It builds and saves the FAISS vector index to disk.
"""

# ----------------------------
# Load HR documents
# ----------------------------
def load_documents():
    documents = []

    # India HR policy
    loader = TextLoader(DATA_DIR/"hr_leave_policy_india.txt")
    docs = loader.load()
    for doc in docs:
        doc.metadata["country"] = "India"
        doc.metadata["policy_type"] = "leave"
    documents.extend(docs)

    # Netherlands HR policy
    loader = TextLoader(DATA_DIR/"hr_leave_policy_netherlands.txt")
    docs = loader.load()
    for doc in docs:
        doc.metadata["country"] = "Netherlands"
        doc.metadata["policy_type"] = "leave"
    documents.extend(docs)

    # General HR policy
    loader = TextLoader(DATA_DIR/"hr_general_policy.txt")
    docs = loader.load()
    for doc in docs:
        doc.metadata["country"] = "General"
        doc.metadata["policy_type"] = "general"
    documents.extend(docs)

    return documents


# ----------------------------
# Create Vector Store
# ----------------------------
def create_vector_store():

    docs = load_documents()                         # load HR docs

    # split documents into small chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,                             # size of each chunk
        chunk_overlap=50                            # overlap for context
    )

    chunks = splitter.split_documents(docs)

    # create embeddings (numbers for text)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # store embeddings in FAISS (vector DB)
    vector_store = FAISS.from_documents(chunks, embeddings)

    # save vector DB to disk
    vector_store.save_local(FAISS_DIR)

    print("✅ HR documents indexed and stored successfully.")


# ----------------------------
# Run once to build index
# ----------------------------
if __name__ == "__main__":
    create_vector_store()
