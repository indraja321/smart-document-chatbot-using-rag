import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

st.title("🤖 Smart Document Chatbot Using RAG")

st.subheader("📚 RAG-Powered Document Assistant")
st.caption("PDF → Chunks → Embeddings → ChromaDB → Retrieval → Ollama")


@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


model = load_embedding_model()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")


st.sidebar.header("Settings")

ollama_model = st.sidebar.text_input(
    "Ollama Model",
    value="llama3.2"
)

chunk_size = st.sidebar.slider(
    "Chunk size",
    200,
    1500,
    500,
    100
)

top_k = st.sidebar.slider(
    "Chunks to retrieve",
    1,
    5,
    3
)


st.header("📄 Upload & Process Document")

uploaded_file = st.file_uploader(
    "Upload a text-based PDF",
    type=["pdf"]
)


if uploaded_file and st.button("Process & Store PDF"):

    reader = PdfReader(uploaded_file)

    text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    if not text.strip():
        st.error("No readable text was found. Try a text-based PDF.")
        st.stop()

    # Create chunks
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    # Generate embeddings
    with st.spinner("Generating embeddings..."):
        embeddings = model.encode(chunks)

    # Delete existing documents
    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Store in ChromaDB
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": uploaded_file.name}
            for _ in range(len(chunks))
        ],
    )

    st.success(
        f"Stored {len(chunks)} chunks in ChromaDB."
    )

    # Show stored chunks
    with st.expander("View stored chunks"):

        for i, chunk in enumerate(chunks[:5]):

            st.write(f"**Chunk {i + 1}:**")
            st.write(chunk)
            st.divider()


st.header("💬 Chat with Your Document")

question = st.text_input(
    "Ask a question about the uploaded document"
)


if st.button("Retrieve & Answer"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    count = collection.count()

    if count == 0:
        st.warning("Please upload and process a PDF first.")
        st.stop()

    # Retrieve relevant chunks
    with st.spinner("Searching documents..."):

        question_embedding = model.encode([question])[0]

        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=min(top_k, count)
        )

    retrieved_chunks = results["documents"][0]

    st.subheader("### 🔍 Retrieved Information")

    for i, chunk in enumerate(retrieved_chunks):

        with st.expander(
            f"Retrieved Chunk {i + 1}"
        ):
            st.write(chunk)

    # Create context
    context = "\n\n".join(retrieved_chunks)

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

    # Ask Ollama
    with st.spinner("Generating answer with llama3.2..."):

        try:

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response["message"]["content"]

            st.subheader("Answer")
            st.write(answer)

        except Exception as e:

            st.error(
                f"Could not connect to Ollama. "
                f"Make sure Ollama is running and the model "
                f"'llama3.2' is installed.\n\n"
                f"Error: {e}"
            )