import os
from embedding import split_into_chunks, embed_chunks
import pickle

def process_all_law_sections(section_dir):
    all_chunks = []
    for fname in os.listdir(section_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(section_dir, fname), 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = split_into_chunks(text)
            all_chunks.extend(chunks)
    embeddings = embed_chunks(all_chunks)
    return all_chunks, embeddings

# Lưu ra file npy để load nhanh
import numpy as np

def save_embeddings(chunks, embeddings, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "law_chunks.txt"), "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.replace('\n', ' ') + "\n")
    np.save(os.path.join(out_dir, "law_embeddings.npy"), embeddings)

def load_embeddings(out_dir):
    import numpy as np
    with open(os.path.join(out_dir, "law_chunks.txt"), "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f.readlines()]
    embeddings = np.load(os.path.join(out_dir, "law_embeddings.npy"))
    return chunks, embeddings

LAW_DIR = "chatbot/legal_data/ucc_article2_sections"
chunks = []
for fname in os.listdir(LAW_DIR):
    if not fname.endswith(".txt"):
        continue
    with open(os.path.join(LAW_DIR, fname), "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            continue
        # Mỗi file là một section, chunk theo section
        chunks.append({
            "section": fname.replace(".txt", ""),
            "text": text
        })

print(f"Tổng số chunk: {len(chunks)}")
for i, c in enumerate(chunks[:3]):
    print(f"Chunk {i+1} - Section: {c['section']}\n{c['text'][:300]}\n---\n")

# Lấy danh sách text để embedding
texts = [c["text"] for c in chunks]
print("Đang sinh embedding cho các chunk luật...")
embeddings = embed_chunks(texts)

# Lưu embedding và metadata ra file
out_path = "chatbot/legal_data/ucc_article2_law_chunks.pkl"
with open(out_path, "wb") as f:
    pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
print(f"Đã lưu embedding và metadata vào {out_path}")