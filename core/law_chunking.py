import os
from embedding import split_into_chunks
from core.chroma_utilis import save_to_laws

def process_all_law_sections(section_dir):
    for root, dirs, files in os.walk(section_dir):
        for fname in files:
            if fname.endswith('.txt'):
                file_path = os.path.join(root, fname)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                # Lấy article_code và section_id từ đường dẫn
                rel_path = os.path.relpath(file_path, section_dir)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    article_code = parts[0]
                    section_id = os.path.splitext(parts[1])[0]
                else:
                    article_code = "unknown"
                    section_id = os.path.splitext(fname)[0]
                chunks = split_into_chunks(text)
                # Thêm metadata cho mỗi chunk
                chunk_objs = []
                for i, chunk in enumerate(chunks):
                    chunk_objs.append({
                        "text": chunk,
                        "article_code": article_code,
                        "section_id": section_id,
                        "filename": fname,
                        "chunk_id": i
                    })
                if chunk_objs:
                    save_to_laws(chunk_objs, filename=fname)

if __name__ == "__main__":
    LAW_DIR = "chatbot/ucc_articles"  # Đường dẫn mới tới thư mục chứa tất cả article
    process_all_law_sections(LAW_DIR)
    print("Đã lưu toàn bộ chunk luật vào Milvus (collection 'laws').")