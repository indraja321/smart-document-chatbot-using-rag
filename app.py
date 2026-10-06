import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Smart Document Chatbot Using RAG")

st.subheader("📚 RAG-Powered Document Assistant")
st.caption(
    "PDF → LangChain → Chunks → Embeddings → ChromaDB → Retrieval → Ollama"
)


# -----------------------------
# Embedding Model
# -----------------------------
@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embedding_model()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Settings")

ollama_model = st.sidebar.text_input(
    "Ollama Model",
    value="llama3.2"
)

chunk_size = st.sidebar.slider(
    "Chunk Size",
    200,
    1500,
    500,
    100
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    0,
    300,
    50,
    50
)

top_k = st.sidebar.slider(
    "Chunks to Retrieve",
    1,
    5,
    3
)


# -----------------------------
# Upload PDF
# -----------------------------
st.header("📄 Upload & Process Document")

uploaded_file = st.file_uploader(
    "Upload a text-based PDF",
    type=["pdf"]
)


# -----------------------------
# Process PDF
# -----------------------------
if uploaded_file and st.button("📥 Process & Store PDF"):

    # Save uploaded PDF temporarily
    pdf_path = "uploaded_document.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Reading PDF..."):

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

    if not documents:

        st.error("No readable text was found in the PDF.")

        st.stop()

    # -----------------------------
    # Split Documents
    # -----------------------------
    with st.spinner("Splitting document into chunks..."):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = text_splitter.split_documents(documents)

    # -----------------------------
    # Create Chroma Vector Store
    # -----------------------------
    with st.spinner("Generating embeddings and storing in ChromaDB..."):

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

    # Store vectorstore in session
    st.session_state.vectorstore = vectorstore

    st.success(
        f"Successfully stored {len(chunks)} chunks in ChromaDB."
    )

    # -----------------------------
    # Display Chunks
    # -----------------------------
    with st.expander("🔍 View Stored Chunks"):

        for i, chunk in enumerate(chunks[:5]):

            st.write(f"**Chunk {i + 1}**")

            st.write(chunk.page_content)

            st.divider()


# -----------------------------
# Chat Section
# -----------------------------
st.header("💬 Chat with Your Document")

question = st.text_input(
    "Ask a question about the uploaded document"
)


# -----------------------------
# Retrieve & Answer
# -----------------------------
if st.button("🔎 Retrieve & Answer"):

    if not question.strip():

        st.warning("Please enter a question.")

        st.stop()

    if "vectorstore" not in st.session_state:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()

    # -----------------------------
    # Retrieve Relevant Documents
    # -----------------------------
    with st.spinner("Searching the document..."):

        vectorstore = st.session_state.vectorstore

        retrieved_docs = vectorstore.similarity_search(
            question,
            k=top_k
        )

    # -----------------------------
    # Display Retrieved Information
    # -----------------------------
    st.subheader("🔍 Retrieved Information")

    for i, doc in enumerate(retrieved_docs):

        with st.expander(
            f"Retrieved Chunk {i + 1}"
        ):

            st.write(doc.page_content)

    # -----------------------------
    # Create Context
    # -----------------------------
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # -----------------------------
    # Create Prompt
    # -----------------------------
    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the context provided below.

If the answer is not present in the context, say:

"I could not find the answer in the uploaded document."

Context:

{context}

Question:

{question}

Answer:
"""

    # -----------------------------
    # Ollama
    # -----------------------------
    with st.spinner(
        f"Generating answer with {ollama_model}..."
    ):

        try:

            llm = OllamaLLM(
                model=ollama_model
            )

            answer = llm.invoke(prompt)

            st.subheader("🤖 Answer")

            st.write(answer)

        except Exception as e:

            st.error(
                f"Could not connect to Ollama.\n\n"
                f"Make sure Ollama is running and "
                f"'{ollama_model}' is installed.\n\n"
                f"Error: {e}"
            )