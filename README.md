# Legal Contract AI Agent

A web-based AI agent for reviewing, analyzing, and answering questions about US sales contracts (UCC) for legal compliance, risk, and improvement suggestions. Supports `.txt`, `.pdf`, `.docx` contracts.

---

## Features

- **Upload contract**: Accepts `.txt`, `.pdf`, `.docx` files.
- **Ask any legal question**: Enter your question about the uploaded contract (in English).
- **US Law compliance**: Detects violations of US law (UCC), suggests improvements, or identifies suitable jurisdictions.
- **Relevant law sections**: Suggests the most relevant UCC law sections for your contract.
- **LLM-powered**: Uses OpenAI/EternalAI GPT models for legal reasoning and summarization.

---

## Project Structure

```
legal_docs_sales/
│
├── core/
│   ├── rag_chain.py           # LLM prompts & legal QA logic
│   ├── semantic_search.py     # Find relevant law sections (UCC)
│   ├── embedding.py           # Text embedding utilities
│   ├── law_chunking.py        # Chunking and embedding UCC law
│   ├── crawl_ucc_article2.py  # Script to crawl UCC Article 2
│
├── uploaded_contracts/        # Uploaded contract files
├── templates/
│   └── index.html             # Web UI template
├── config/
│   └── config.py              # API key and endpoint config
├── web_app.py                 # Main Flask web app
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .env                       # (You create) API keys and secrets
```

---

## Setup

1. **Clone & setup environment**

   ```bash
   git clone <your-repo-url>
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API keys**

   - Tạo file `.env` với:
     ```
     LLM_API_KEY=sk-...
     LLM_API_URL=.....
     LLM_MODEL_ID=gpt-4o-mini
     ```

3. **Run the app**
   ```bash
   python web_app.py
   ```
   - Truy cập [http://127.0.0.1:8050](http://127.0.0.1:8050)

---

## Notes

- Only English questions are supported.
- Uploaded contracts are stored in `uploaded_contracts/`.
- For issues, open an issue or pull request.

---

## License

MIT License (or your license here)
