import streamlit as st
import tempfile
import os

from rag import extract_text, chunk_text, build_index, retrieve, generate_answer

st.set_page_config(page_title="PDF Q&A", layout="centered")
st.title("PDF Q&A with RAG")
st.caption("Upload a PDF, then ask questions about its contents.")

# Session setup 
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "filename" not in st.session_state:
    st.session_state.filename = None

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Only re process if this is a new file 
    if st.session_state.filename != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Extracting text from PDF..."):
            text = extract_text(tmp_path)

        os.remove(tmp_path)

        if not text.strip():
            st.error("No extractable text found in this PDF. It may be a scanned/image-only document.")
        else:
            with st.spinner("Splitting document into chunks..."):
                chunks = chunk_text(text, chunk_size=250, overlap=40)

            st.write(f"Document split into **{len(chunks)}** chunks. Building search index...")

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(done, total):
                progress_bar.progress(done / total)
                status_text.text(f"Embedding chunk {done}/{total}")

            index, embeddings = build_index(chunks, batch_size=64, progress_callback=update_progress)

            progress_bar.empty()
            status_text.empty()

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.filename = uploaded_file.name

            st.success(f"'{uploaded_file.name}' processed and ready for questions.")

# Answer Questions
if st.session_state.index is not None:
    st.divider()
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Retrieving relevant sections..."):
            retrieved_chunks = retrieve(query, st.session_state.chunks, st.session_state.index, top_k=2)

        with st.spinner("Generating answer..."):
            answer = generate_answer(query, retrieved_chunks)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Show retrieved context"):
            for i, chunk in enumerate(retrieved_chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)
                st.markdown("---")
else:
    st.info("Upload a PDF above to get started.")