import chromadb
from chromadb.config import Settings
from core.embedding import embed_chunks

chroma_client = chromadb.Client(Settings(
    persist_directory="chroma_db"  # Thư mục lưu trữ local
))

def get_or_create_collection(name):
    return chroma_client.get_or_create_collection(name)

def save_to_contracts(chunks, filename, vectors=None):
    collection = get_or_create_collection("contracts")
    if vectors is None:
        vectors = embed_chunks(chunks)
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "chunk_id": i} for i in range(len(chunks))]
    collection.add(
        embeddings=vectors,
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

def search_contracts(query, filename=None, top_k=5, model_override=None):
    collection = get_or_create_collection("contracts")
    query_vector = embed_chunks([query], model_override=model_override)[0]
    where = {"filename": filename} if filename else None
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where
    )
    matches = []
    for i in range(len(results["ids"][0])):
        matches.append({
            "score": results["distances"][0][i],
            "chunk": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"]
        })
    return matches

def save_to_laws(chunk_objs, filename, vectors=None):
    # chunk_objs là list dict, mỗi dict có 'text', 'article_code', 'section_id', 'filename', 'chunk_id'
    collection = get_or_create_collection("laws")
    chunks = [obj["text"] for obj in chunk_objs]
    if vectors is None:
        vectors = embed_chunks(chunks)
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "filename": obj["filename"],
            "chunk_id": obj["chunk_id"],
            "article_code": obj.get("article_code", ""),
            "section_id": obj.get("section_id", "")
        }
        for obj in chunk_objs
    ]
    collection.add(
        embeddings=vectors,
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

def search_laws(query, top_k=5, filename=None, model_override=None):
    collection = get_or_create_collection("laws")
    query_vector = embed_chunks([query], model_override=model_override)[0]
    where = {"filename": filename} if filename else None
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where
    )
    matches = []
    for i in range(len(results["ids"][0])):
        matches.append({
            "score": results["distances"][0][i],
            "chunk": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"]
        })
    return matches 