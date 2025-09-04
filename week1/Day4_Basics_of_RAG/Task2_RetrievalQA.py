import os
from dotenv import load_dotenv

# Load .env file if you store GROQ_API_KEY there
load_dotenv()

# ---- LangChain imports ----
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document

# -------------------------------
# 1. Sample Company Policy
# -------------------------------
policy_text = """
Company Policy Document

Refund Policy:
Customers may request a refund within 30 days of purchase if they are not satisfied with the product.
Refunds will be processed within 7 business days after approval.
Digital products are non-refundable once downloaded.
"""
documents = [Document(page_content=policy_text, metadata={"source": "policy"})]

# -------------------------------
# 2. Split text into chunks
# -------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# -------------------------------
# 3. Embeddings with HuggingFace
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS index
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# 4. Groq LLM
# -------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # You can also try "mixtral-8x7b-32768"
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------------
# 5. RetrievalQA Chain
# -------------------------------
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# -------------------------------
# 6. Ask a Question
# -------------------------------
query = "What is the refund policy?"
answer = qa.run(query)

print("Q:", query)
print("A:", answer)


