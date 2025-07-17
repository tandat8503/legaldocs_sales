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
    # Nếu schema Milvus chỉ có id, filename, chunk, embedding:
    laws_collection.insert([ids, filenames, chunks, vectors])
    laws_collection.flush()

def save_to_contracts(chunks, filename, vectors=None):
    if vectors is None:
        vectors = embed_chunks(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    filenames = [filename] * len(chunks)
    contracts_collection.insert([ids, filenames, chunks, vectors])
    contracts_collection.flush()

def search_laws(query, top_k=5, model_override=None):
    query_vectors = embed_chunks([query], model_override=model_override)
    query_vector = query_vectors[0]
    laws_collection.load()
    results = laws_collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 32}},
        limit=top_k,
        output_fields=["chunk", "filename"]
    )
    matches = []
    for hit in results[0]:
        matches.append({
            "score": hit.score,
            "chunk": hit.entity.get("chunk"),
            "filename": hit.entity.get("filename")
        })
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
            "score": hit.score,
            "chunk": hit.entity.get("chunk"),
            "filename": hit.entity.get("filename")
        })
    return matches

def delete_contract(filename):
    contracts_collection.load()
    contracts_collection.delete(expr=f'filename == "{filename}"')
    contracts_collection.flush()