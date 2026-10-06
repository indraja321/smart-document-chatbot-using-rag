# 🤖 Smart Document Chatbot Using RAG

## 📌 Project Overview

The **Smart Document Chatbot Using RAG** is an AI-based application that allows users to upload a PDF document and ask questions about its content.

The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from the uploaded document and generate accurate, context-based answers.

Instead of asking the AI to answer from general knowledge, the chatbot first searches the uploaded document and then uses the retrieved information to generate the response.

---

## 🎯 Objectives

- Allow users to upload PDF documents.
- Extract useful information from documents.
- Divide documents into smaller chunks.
- Convert document chunks into embeddings.
- Store embeddings in ChromaDB.
- Retrieve relevant information based on the user's question.
- Generate answers using the Ollama language model.
- Provide a simple and interactive interface using Streamlit.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Main programming language |
| **Streamlit** | Web application interface |
| **LangChain** | RAG application framework |
| **PyPDF** | PDF document loading |
| **HuggingFace Embeddings** | Generate document embeddings |
| **ChromaDB** | Vector database |
| **Ollama** | Local Large Language Model |
| **Llama 3.2** | Language model used for generating answers |

---

## 🔄 RAG Workflow

```text
        📄 PDF Document
              ↓
        PDF Text Extraction
              ↓
       Document Chunking
              ↓
       Generate Embeddings
              ↓
          ChromaDB
              ↓
      User Asks a Question
              ↓
      Similarity Search
              ↓
      Retrieve Relevant Chunks
              ↓
          Ollama LLM
              ↓
        🤖 Final Answer
