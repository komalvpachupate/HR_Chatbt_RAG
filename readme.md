# Flow of the Code
# 1. Imports
# The necessary libraries are imported, including:
# os for file operations.
# psycopg2 for PostgreSQL database connections.
# streamlit for creating the web interface.
# langchain components for text processing, embeddings, vector storage, and language model integration.
# 2. Database Connection
# Function: connect_to_db()
# Tries to establish a connection to a PostgreSQL database.
# Returns a connection object and a cursor for executing SQL commands.
# Handles exceptions and displays an error message if the connection fails.
# 3. Fetching Data from Database
# Function: fetch_news_summaries()
# Calls connect_to_db() to get the database connection.
# Executes a SQL query to fetch non-null summaries from the news table.
# Returns a list of summaries or an empty list if there are errors.
# 4. Creating FAISS Vector Store
# Function: create_vector_store(summaries)
# Checks if summaries are available; if not, it warns the user.
# Uses CharacterTextSplitter to split summaries into smaller chunks for processing.
# Creates embeddings using HuggingFace's model (sentence-transformers/all-MiniLM-L6-v2).
# Initializes a FAISS vector store with these embeddings and saves it locally.
# Returns the created vector store.
# 5. Loading or Creating Vector Store
# Function: get_vector_store()
# Checks if a local FAISS index exists.
# If it exists, loads it; otherwise, fetches summaries from the database and creates a new vector store using create_vector_store().
# Uses Streamlit's caching mechanism to optimize performance.
# 6. Setting Up Language Model
# After loading or creating the vector store:
# Initializes an instance of the Ollama model (e.g., llama2) using LangChain's interface.
# Creates a retrieval-based question-answering chain (RetrievalQA) using the loaded vector store and Ollama model.
# 7. Streamlit User Interface
# Sets up a simple web interface with Streamlit:
# Displays a title and description for the app.
# Provides an input box for users to enter a blog topic.
# Includes a button to generate the blog based on user input.
# 8. Generating Blog Content
# On clicking the "Generate Blog" button:
# Checks if a topic is entered and if the QA chain is initialized.
# If valid, it runs the QA chain with the provided topic to generate blog content and displays it.
# If there are issues (e.g., no topic entered or QA chain not initialized), appropriate messages are shown to guide the user.