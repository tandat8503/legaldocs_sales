import pickle
import numpy as np
from core.embedding import embed_chunks

def search_law_sections(contract_text, top_k=5):
    # Load law chunks & embeddings
    try:
        with open("chatbot/legal_data/ucc_article2_law_chunks.pkl", "rb") as f:
            data = pickle.load(f)
        law_chunks = data["chunks"]
        law_embeddings = np.array(data["embeddings"])
    except FileNotFoundError:
        raise RuntimeError("Chưa có file chatbot/legal_data/ucc_article2_law_chunks.pkl. Hãy chạy law_chunking.py trước!")
    # Embed contract text
    contract_vec = np.array(embed_chunks([contract_text])[0])
    # Tính cosine similarity
    sims = law_embeddings @ contract_vec / (np.linalg.norm(law_embeddings, axis=1) * np.linalg.norm(contract_vec) + 1e-8)
    idxs = np.argsort(sims)[::-1][:top_k]
    results = [{
        "section": law_chunks[i]["section"],
        "text": law_chunks[i]["text"],
        "score": float(sims[i])
    } for i in idxs]
    return results

# Test nhanh
if __name__ == "__main__":
    query = "delivery of goods"
    results = search_law_sections(query)
    for r in results:
        print(f"Section: {r['section']} (score: {r['score']:.3f})\n{r['text'][:200]}\n---\n")