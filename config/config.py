from dotenv import load_dotenv
import os
load_dotenv()
LLM_API_URL = "https://vibe-agent-gateway.eternalai.org/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not LLM_API_URL:
    raise ValueError("Chưa thiết lập LLM_API_URL trong .env")

if not OPENAI_API_KEY:
    raise ValueError("Chưa thiết lập OPENAI_API_KEY trong .env")
