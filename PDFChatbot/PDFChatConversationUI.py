__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import google.generativeai as genai
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain


# 🔑 Load API Key

genai.configure(api_key="AIzaSyBE-lpNACAu_b6RV5UMLbm2PGXj73wqniw")

st.title("📄 AskMyPDF – Chat with Any PDF Using LangChain 1.x + Gemini")


# ----------------------
# PDF UPLOAD
# ----------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    # Remove old Chroma DB to avoid mixing PDFs
    if os.path.exists("chromadb"):
        shutil.rmtree("chromadb")

    # Save uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
    )
    docs = text_splitter.split_documents(data)

    # Create ChromaDB VectorStore
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory="chromadb"
    )

    retriever = vectorstore.as_retriever()

    # LLM (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # Prompt Template
    prompt = ChatPromptTemplate.from_template("""
You are an intelligent assistant. Use ONLY the provided PDF context to answer the question.
If answer is not in the PDF, say "Sorry, I could not find that in the document."

Context:
{context}

Question: {question}
""")

    # Documents → Answer chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # RAG chain using RunnablePassthrough
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | question_answer_chain
    )

    # ----------------------
    # Chat Input
    # ----------------------
    user_query = st.chat_input("Ask me anything about the PDF:")

    if user_query:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_query)
            st.write(response["output_text"])

else:
    st.info("👆 Upload a PDF to get started")




