from pymilvus import connections, Collection

# Kết nối tới Milvus (sửa host/port nếu cần)
connections.connect(host="localhost", port="19530")

# Tên các collection cần xóa
collections_to_delete = ["laws", "contracts"]

for col_name in collections_to_delete:
    try:
        col = Collection(col_name)
        col.drop()
        print(f"✅ Dropped collection: {col_name}")
    except Exception as e:
        print(f"⚠️  Could not drop collection {col_name}: {e}")