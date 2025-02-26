import os
import psycopg2
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.llms import Ollama





# Function to create FAISS vector store
def create_vector_store(summaries):
    if not summaries:
        st.warning("⚠️ No data found in database!")
        return None

    st.info("🔄 Creating FAISS vector store from database...")

    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    split_docs = text_splitter.create_documents(summaries)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embeddings)

    vector_store.save_local("faiss_index")  # Save to local directory
    st.success("✅ FAISS model saved successfully!")

    return vector_store


# Load FAISS vector store, or recreate if missing
@st.cache_resource
def get_vector_store():
    """Loads FAISS vector store. If missing, recreates it from database."""
    if os.path.exists("faiss_index"):
        st.info("📂 Loading existing FAISS index...")
        return FAISS.load_local("faiss_index",
                                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

    # If index does not exist, recreate it
    summaries = fetch_news_summaries()
    return create_vector_store(summaries)

model_name = "llama2:latest"
# Initialize FAISS vector store
vector_store = get_vector_store()
if vector_store:
    retriever = vector_store.as_retriever()
    llm = Ollama(model=model_name)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
else:
    qa_chain = None

# Streamlit UI
st.title("📰 AI-Powered News Blog Generator")
st.write("Enter a blog topic, and AI will generate a blog using retrieved news summaries!")

# User input
topic = st.text_input("Enter blog topic:", "")

if st.button("Generate Blog"):
    if topic and qa_chain:
        with st.spinner("Generating your blog... ⏳"):
            response = qa_chain.run(topic)
            st.success("✅ Blog generated successfully!")
            st.write(response)
    elif not qa_chain:
        st.error("❌ FAISS vector store not initialized. Check database connection.")
    else:
        st.warning("⚠️ Please enter a topic before generating.")


