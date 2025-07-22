from sentence_transformers import SentenceTransformer
from typing import List
import time

print("🕒 Starting model loading...")
start_time = time.time()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

load_time = time.time() - start_time
print(f"✅ Model loaded in {load_time:.2f} seconds")

# Chunk configuration
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
BATCH_SIZE = 32  # Process 32 chunks at a time

def split_into_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    # Split text into overlapping chunks
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += size - overlap
    return chunks

def embed_chunks(chunks: List[str], model_override=None) -> List[List[float]]:
    start_time = time.time()
    all_embeddings = []
    used_model = model_override if model_override is not None else model
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        show_progress = len(batch) > 10
        batch_embeddings = used_model.encode(batch, show_progress_bar=show_progress)
        all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
    embed_time = time.time() - start_time
    print(f"⏱️ Embedding {len(chunks)} chunks took {embed_time:.2f} seconds")
    return all_embeddings
