from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from core.embedding import embed_chunks
import uuid

connections.connect(host="localhost", port="19530")

def get_or_create_collection(name):
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="article_code", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="section_id", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description=f"{name} Chunks")
    try:
        collection = Collection(name, schema=schema)
    except Exception:
        collection = Collection(name)
    if not collection.has_index():
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_SQ8",
                "metric_type": "IP",
                "params": {"nlist": 1024}
            }
        )
    return collection

laws_collection = get_or_create_collection("laws")
contracts_collection = get_or_create_collection("contracts")

def save_to_laws(chunk_objs, filename, vectors=None):
    # chunk_objs là list dict, mỗi dict có 'text', 'article_code', 'section_id', 'filename', 'chunk_id'
    chunks = [obj["text"] for obj in chunk_objs]
    if vectors is None:
        vectors = embed_chunks(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    filenames = [obj["filename"] for obj in chunk_objs]
    article_codes = [obj["article_code"] for obj in chunk_objs]
    section_ids = [obj["section_id"] for obj in chunk_objs]
    chunk_ids = [obj["chunk_id"] for obj in chunk_objs]
    laws_collection.insert([ids, filenames, chunks, vectors, article_codes, section_ids, chunk_ids])
    laws_collection.flush()

def save_to_contracts(chunks, filename, vectors=None):
    if vectors is None:
        vectors = embed_chunks(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    filenames = [filename] * len(chunks)
    # Để tương thích, chèn các trường rỗng cho contracts (nếu muốn mở rộng schema cho contracts tương tự laws)
    article_codes = [""] * len(chunks)
    section_ids = [""] * len(chunks)
    chunk_ids = list(range(len(chunks)))
    contracts_collection.insert([ids, filenames, chunks, vectors, article_codes, section_ids, chunk_ids])
    contracts_collection.flush()

def search_laws(query, top_k=5, article_code=None, section_id=None, keyword=None, model_override=None):
    """
    Advanced semantic search for law sections with filtering.
    - article_code: filter by Article (e.g., '2', '2A', ...)
    - section_id: filter by Section (e.g., '2-201', ...)
    - keyword: filter by keyword in chunk text
    - model_override: specify embedding model if needed
    Returns a list of dicts: score, chunk, filename, article_code, section_id, chunk_id
    """
    query_vectors = embed_chunks([query], model_override=model_override)
    query_vector = query_vectors[0]
    laws_collection.load()
    exprs = []
    if article_code:
        exprs.append(f'article_code == "{article_code}"')
    if section_id:
        exprs.append(f'section_id == "{section_id}"')
    expr = " and ".join(exprs) if exprs else None
    results = laws_collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 32}},
        limit=top_k * 2,  # lấy nhiều hơn để lọc keyword nếu cần
        output_fields=["chunk", "filename", "article_code", "section_id", "chunk_id"],
        expr=expr
    )
    matches = []
    for hit in results[0]:
        chunk_text = hit.get("chunk", "")
        if keyword and keyword.lower() not in chunk_text.lower():
            continue
        matches.append({
            "score": hit.get("_score", 0),
            "chunk": chunk_text,
            "filename": hit.get("filename"),
            "article_code": hit.get("article_code"),
            "section_id": hit.get("section_id"),
            "chunk_id": hit.get("chunk_id"),
        })
        if len(matches) >= top_k:
            break
    return matches

def search_contracts(query, filename=None, top_k=5, model_override=None):
    query_vectors = embed_chunks([query], model_override=model_override)
    query_vector = query_vectors[0]
    contracts_collection.load()
    expr = None
    if filename:
        expr = f'filename == "{filename}"'
    results = contracts_collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 32}},
        limit=top_k,
        output_fields=["chunk", "filename"],
        expr=expr
    )
    matches = []
    for hit in results[0]:
        matches.append({
            "score": hit.get("_score", 0),
            "chunk": hit.get("chunk"),
            "filename": hit.get("filename")
        })
    return matches

def delete_contract(filename):
    contracts_collection.load()
    contracts_collection.delete(expr=f'filename == "{filename}"')
    contracts_collection.flush()