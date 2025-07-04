"""
RAG Chain Module - Hybrid mode using AI-based routing
"""

import requests
import json
from config.config import LLM_API_URL, OPENAI_API_KEY
from core.milvus_utilis import collection, search_similar_chunks

# Send a custom prompt to the LLM
def ask_llm_with_context_custom_prompt(prompt: str) -> str:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. You must only use the provided document content to answer."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    # Ensure LLM_API_URL is set
    if not LLM_API_URL:
        raise ValueError("LLM_API_URL is not configured in environment variables")
    response = requests.post(LLM_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# Semantic search using document chunks
def ask_llm_with_context(query: str) -> str:
    results = search_similar_chunks(query, top_k=1000)
    context = "\n".join([r["chunk"] for r in results])

    prompt = f"""
You are a helpful assistant with access to document snippets.

Based on the following context, answer the user's question concisely and clearly:

CONTEXT:
{context}

QUESTION:
{query}
"""
    return ask_llm_with_context_custom_prompt(prompt)


# Load all document chunks for full-context answers
def load_all_chunks_for_context(limit: int = 10) -> str:
    collection.load()
    results = collection.query(
        expr="",
        output_fields=["chunk"],
        limit=limit
    )
    chunks = [r["chunk"] for r in results]
    return "\n".join(chunks)
 

# Use all document data for creative or vague queries
def ask_with_full_context(query: str) -> str:
    full_data = load_all_chunks_for_context(limit=1000)

    prompt = f"""
You are a curious, creative AI assistant.

Below is a collection of document excerpts from various articles and reports:

{full_data}

Now, based on this user question: "{query}", please extract and explain something surprisingly interesting, insightful, or fun. Be creative. Do not repeat the same structure every time. Express it in a friendly and engaging tone.
"""
    return ask_llm_with_context_custom_prompt(prompt)

# Tool function descriptors for LLM tool calling
# Used for OpenAI tool-calling API
tool_functions = [
    {
        "type": "function",
        "function": {
            "name": "ask_llm_with_context",
            "description": "Answer specific or factual questions using semantic search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user question"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_with_full_context",
            "description": "Answer vague or creative questions using full document context",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user question"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Main smart toolcall entry point
# Decides which function to use based on the query
def ask_question_smart_with_toolcall(query: str) -> str:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Based on the user's question, "
                    "you must choose one of the available functions to best answer it. "
                    "Do not respond directly. Always select a tool."
                )
            },
            {"role": "user", "content": query}
        ],
        "tools": tool_functions,
        "tool_choice": "required"  
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(LLM_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        message = data["choices"][0]["message"]
        tool_call = message["tool_calls"][0]
        func_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
    except Exception as e:
        print("⚠️ Tool call parsing error:", e)
        return "Sorry, something went wrong when selecting a tool."

    print(f"🔧 AI chose: {func_name}({arguments})")

    # Call the selected Python function
    if func_name == "ask_llm_with_context":
        return ask_llm_with_context(arguments["query"])
    elif func_name == "ask_with_full_context":
        return ask_with_full_context(arguments["query"])
    else:
        return f"❌ Unknown function selected: {func_name}"
