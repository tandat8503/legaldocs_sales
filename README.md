# AI Document Assistant

A powerful document processing and question-answering system built with Python. The system uses Milvus vector database for efficient semantic search and OpenAI's GPT models for intelligent responses.

## Features

- Document processing and chunking
- Semantic search using Milvus vector database
- GUI interface for easy interaction
- AI-powered question answering
- Document embedding and storage

## Prerequisites

- Python 3.8 or higher
- Milvus 2.5.x
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd document_assitant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your_api_key_here
LLM_API_URL=https://api.openai.com/v1/chat/completions
```

4. Install and start Milvus:
```bash
# Using Docker
# Follow the official Milvus installation guide for your platform:
Click this link: https://milvus.io/docs. Follow the instruction, and choose the proper installation
```

## Project Structure

```
document_assitant/
├── core/               # Core functionality
│   ├── rag_chain.py   # RAG implementation
│   ├── embedding.py   # Text embedding utilities
│   └── milvus_utilis.py # Milvus database operations
├── config/            # Configuration files
│   └── config.py      # Environment and API settings
├── gui/               # GUI interface
├── testing files/     # Test documents and data
└── requirements.txt   # Python dependencies
```

## Usage

### GUI Application

Run the GUI application:
```bash
Make sure the Milvus db is starting successfully
python gui/app_gui.py
```

The GUI provides the following features:
- Upload and process documents
- Search through documents
- Ask questions about your documents
- View and manage stored documents

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 