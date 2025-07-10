from core.milvus_utilis import search_laws

def search_law_sections(query, top_k=5):
    return search_laws(query, top_k=top_k)

# Test nhanh
if __name__ == "__main__":
    query = "delivery of goods"
    results = search_law_sections(query)
    for r in results:
        print(f"Filename: {r['filename']} (score: {r['score']:.3f})\n{r['chunk'][:200]}\n---\n")