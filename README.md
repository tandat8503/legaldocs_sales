# Legal Contract AI Agent

A powerful AI agent for reviewing, analyzing, and answering questions about US sales contracts (UCC) and related commercial law.  
Supports `.txt`, `.pdf`, `.docx` contracts, deep semantic search, and Retrieval Augmented Generation (RAG) for high-quality legal answers.

---

## Features

- **Contract Upload**: Accept `.txt`, `.pdf`, `.docx` files for contract review.
- **Ask Legal Questions**: Enter any question about the uploaded contract or general US contract law, liens, secured transactions, etc.
- **Deep Semantic Search**: Uses local SentenceTransformer embeddings and Milvus vector database to find relevant law and contract sections.
- **RAG (Retrieval Augmented Generation)**: Combines search results with LLM (OpenAI/EternalAI GPT) for professional legal memos.
- **Relevant Law Sections**: Suggests the most relevant UCC law sections for your question.
- **Session & History**: Remembers question history per contract.
- **Streamlit & Flask UI**: Modern web interface, easy to use.
- **Error Handling**: Detects non-contract uploads, out-of-scope questions, and provides clear feedback.

---

## Project Structure

```
legal_docs_sales/
│
├── core/
│   ├── rag_chain.py         # LLM prompt & legal QA logic (RAG)
│   ├── semantic_search.py   # Semantic search for law/contract sections
│   ├── embedding.py         # Embedding utilities (SentenceTransformer)
│   ├── law_chunking.py      # Chunk & embed UCC law text files
│   ├── milvus_utilis.py     # Milvus vector DB utilities
│
├── uploaded_contracts/      # Uploaded contract files (auto-created)
├── templates/
│   └── index.html           # Flask web UI template
├── config/
│   └── config.py            # LLM API key & endpoint config
├── app_streamlit.py         # Streamlit web app (recommended)
├── web_app.py               # Flask web app (legacy)
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Milvus, Minio, Etcd setup
└── README.md                # This file
```

---

## Setup Instructions

### 1. Clone & Setup Python Environment

```bash
git clone https://github.com/tandat8503/legaldocs_sales.git
cd legal_docs_sales
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure LLM API Keys

Tạo file `.env` trong thư mục `legal_docs_sales/config/` hoặc gốc project với nội dung:

```
OPENAI_API_KEY=sk-...         # Your OpenAI or EternalAI API key
LLM_API_URL=https://...       # LLM API endpoint (EternalAI or OpenAI)
LLM_MODEL_ID=gpt-4o-mini      # Model name (default: gpt-4o-mini)
```

### 3. Start Milvus Vector Database (Docker)

```bash
docker-compose up -d
```

- Milvus runs on `localhost:19530` by default.

### 4. Prepare UCC Law Data (First Time Only)

- Ensure all UCC law `.txt` files are in `chatbot/ucc_articles/`.
- Run the chunking & embedding script to index law sections:
  ```bash
  python -m core.law_chunking
  ```

### 5. Run the Web App

#### Option 1: Streamlit UI (Recommended)

```bash
streamlit run app_streamlit.py
```

- App runs at [http://localhost:8501](http://localhost:8501)

#### Option 2: Flask UI

```bash
python web_app.py
```

- App runs at [http://localhost:8050](http://localhost:8050)

---

## Usage Guide

1. **Upload a contract** (`.txt`, `.pdf`, `.docx`).
2. **Enter your legal question** (about the contract or general US contract law).
3. **Submit** to get a professional legal memo, risk analysis, and relevant UCC law sections.
4. **Review question history** for each contract.
5. **Finish session** or **upload new contract** as needed.

**Note:**

- You can ask general legal questions (e.g., about liens, UCC, secured transactions) without uploading a contract.
- The system will warn if the uploaded file is not a contract or if the question is out of scope.

---

## Advanced: Re-embedding Law Data

If you update UCC law files or Milvus schema:

1. (Optional) Run a script to delete old Milvus collections.
2. Re-run the chunking script:
   ```bash
   python -m core.law_chunking
   ```

---

## Troubleshooting

- **Milvus connection errors**: Ensure Docker containers are running.
- **LLM API errors**: Check `.env` for correct API key and endpoint.
- **Non-contract file warning**: Only upload valid contracts for analysis.
- **Out-of-scope question**: Only US contract law, UCC, liens, secured transactions, etc. are supported.

---

## License

MIT License (or your license here)

---

## Contact

For issues or feature requests, open an issue or pull request on your repository.
