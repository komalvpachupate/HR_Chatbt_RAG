import os
import streamlit as st
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Function to load data from CSV
def load_csv_data():
    df = pd.read_csv("hr_onboarding.csv")
    if 'question' not in df.columns or 'answer' not in df.columns:
        st.error("CSV must contain 'question' and 'answer' columns")
        return None
    return df

# Function to create FAISS vector store
def create_vector_store(data):
    if data is None or data.empty:
        st.warning("⚠️ No data found in CSV!")
        return None
    
    st.info("🔄 Creating FAISS vector store from CSV data...")
    
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = [f"Q: {q}\nA: {a}" for q, a in zip(data['question'], data['answer'])]
    split_docs = text_splitter.create_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    
    vector_store.save_local("faiss_qna_index")
    st.success("✅ FAISS model saved successfully!")
    return vector_store

# Load FAISS vector store or recreate if missing
@st.cache_resource
def get_vector_store():
    if os.path.exists("faiss_qna_index"):
        st.info("📂 Loading existing FAISS index...")
        return FAISS.load_local("faiss_qna_index", 
                                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), 
                                allow_dangerous_deserialization=True)
    
    data = load_csv_data()
    return create_vector_store(data)

st.title("🤖 AI-Powered Q&A HR Chatbot")

vector_store = get_vector_store()
if vector_store:
    retriever = vector_store.as_retriever()
    llm = Ollama(model="llama2:latest")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
else:
    qa_chain = None

# User input for question
question = st.text_input("Ask a question:", "")
if st.button("Get Answer"):
    if question and qa_chain:
        with st.spinner("Finding the best answer... ⏳"):
            response = qa_chain.run(question)
            st.success("✅ Answer Found!")
            st.write(response)
    elif not qa_chain:
        st.error("❌ FAISS vector store not initialized. Please check the CSV file.")
    else:
        st.warning("⚠️ Please enter a question before asking.")
