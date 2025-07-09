# Legal Contract AI Agent

A web-based AI agent for reviewing, analyzing, and answering questions about US sales contracts (UCC) for legal compliance, risk, and improvement suggestions. Supports English-language contracts in `.txt`, `.pdf`, `.docx` formats.

---

## Features

- **Upload contract**: Accepts `.txt`, `.pdf`, `.docx` files.
- **Ask any legal question**: Enter your question about the uploaded contract (in English).
- **US Law compliance**: Detects violations of US law (UCC), suggests improvements, or identifies suitable jurisdictions.
- **Relevant law sections**: Suggests the most relevant UCC law sections for your contract.
- **LLM-powered**: Uses OpenAI GPT models for legal reasoning and summarization.
- **Input validation**: Only English, legal-related questions are accepted; answers are concise and safe.
- **No unnecessary code**: CLI, Milvus, and unrelated files have been removed for clarity and maintainability.

---

## Project Structure

```
chatbot/
│
├── core/
│   ├── rag_chain.py           # LLM prompts & legal QA logic
│   ├── semantic_search.py     # Find relevant law sections (UCC)
│   ├── embedding.py           # Text embedding utilities
│   ├── law_chunking.py        # Chunking and embedding UCC law
│   ├── crawl_ucc_article2.py  # Script to crawl UCC Article 2 from Cornell
│   └── __pycache__/           # Python cache (can be deleted)
│
├── uploaded_contracts/        # Uploaded contract files (auto-created)
│
├── templates/
│   └── index.html             # Web UI template
│
├── config/
│   └── config.py              # API key and endpoint config
│
├── web_app.py                 # Main Flask web app
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .env                       # (You create) API keys and secrets
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo>
cd chatbot
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the `chatbot/` directory with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

You can also adjust `LLM_API_URL` in `config/config.py` if needed.

---

## Usage

### 1. Start the web app

```bash
python web_app.py
```

The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 2. Upload a contract

- Click "Upload contract" and select a `.txt`, `.pdf`, or `.docx` file.

### 3. Ask your legal question

- Enter your question in English (e.g., "Does this contract violate any US law?", "Is this contract suitable for use in the US?").
- The AI will answer concisely, focusing on your question, and suggest improvements or identify legal issues.
- Only English, legal-related questions are accepted. Unsafe or off-topic questions will be refused.

---

## Advanced: Update or Re-crawl UCC Law

- To re-crawl UCC Article 2: run `python -m core.crawl_ucc_article2`
- To re-chunk and embed law sections: run `python -m core.law_chunking`
- To test semantic search: run `python -m core.semantic_search`

---

## Notes

- Only English questions are supported.
- The AI will refuse to answer non-legal or unsafe questions.
- Uploaded contracts are stored in `uploaded_contracts/` for session reuse.
- All code for CLI, Milvus, and unrelated features has been removed for clarity.
- Python cache files (`__pycache__`) can be deleted at any time.

---

## License

MIT License (or your license here)

---

**For any issues or contributions, please open an issue or pull request.**
