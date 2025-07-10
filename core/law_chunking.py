import os
from core.embedding import split_into_chunks
from core.milvus_utilis import save_to_laws

def process_all_law_sections(section_dir):
    for fname in os.listdir(section_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(section_dir, fname), 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = split_into_chunks(text)
            if chunks:
                save_to_laws(chunks, filename=fname)

if __name__ == "__main__":
    LAW_DIR = "chatbot/legal_data/ucc_article2_sections"  # Đổi path nếu cần
    process_all_law_sections(LAW_DIR)
    print("Đã lưu toàn bộ chunk luật vào Milvus (collection 'laws').")