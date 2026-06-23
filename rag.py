from pypdf import PdfReader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

GEN_MODEL_NAME = "google/flan-t5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)


gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(device)


def extract_text(pdf_path):
    # Extract text from a PDF, page by page
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)
    return "\n".join(pages_text)


def chunk_text(text, chunk_size=500, overlap=50):
    # Split text into overlapping chunks.
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_index(chunks, batch_size=64, progress_callback=None):
    # Embed chunks in batches and build a FAISS index.
    # progress_callback(done, total) is called after each batch for a progress bar on large documents.
    all_embeddings = []
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = embedder.encode(
            batch,
            convert_to_numpy=True,
            batch_size=batch_size,
        )
        all_embeddings.append(batch_embeddings)
        if progress_callback:
            progress_callback(min(i + batch_size, total), total)

    embeddings = np.vstack(all_embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings


def retrieve(query, chunks, index, top_k=3):
    # Find the top k most relevant chunks for user query.
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer(query, retrieved_chunks, max_new_tokens=200):
    context = "\n\n".join(retrieved_chunks)
    prompt = (
        "You will answer a question using ONLY the context provided. "
        "First find the sentence(s) in the context that answer the question, "
        "then answer using that information. "
        "If the context does not contain the answer, respond with: I don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    inputs = gen_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    outputs = gen_model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer