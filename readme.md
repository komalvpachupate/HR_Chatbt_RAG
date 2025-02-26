# Q&A Chatbot using RAG, OLLAMA, FIASS and Streamlit 

This project is a simple **Q&A Chatbot** built using **Streamlit**, **FAISS**, and **LangChain**. The chatbot allows users to upload a CSV file containing questions and answers, which is then used for **retrieval-augmented generation (RAG)** to provide intelligent responses.

## 🚀 Setup and Installation


### **1. Create and Activate Virtual Environment**

```
python -m venv qna_env
qna_env\Scripts\activate
```

### **3. Install Dependencies**
```
pip install -r requirements.txt
```
> **Note:** Ensure `requirements.txt` contains:
> ```
> streamlit
> pandas
> langchain
> langchain-community
> faiss-cpu
> sentence-transformers
> ```

## 🎯 Usage

### **1. Run the Streamlit App**
```sh
streamlit run app.py
```

### **2. Upload CSV File**
- The CSV file must have **'question'** and **'answer'** columns.
- The system will process the file and create a FAISS vector store.

### **3. Ask Questions**
- Type a question in the text box and get an AI-generated answer based on the uploaded data.

## 🛠 Troubleshooting

- If FAISS index does not load, delete `faiss_qna_index` and restart the app.

## 📌 Features
✔ Upload custom Q&A dataset (CSV format)
✔ AI-powered search using FAISS, OLLAMA
✔ Lightweight and easy to set up

## Output image 
![alt text](<Final_Output.png>)

