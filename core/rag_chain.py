"""
RAG Chain Module - Hybrid mode using AI-based routing
"""

import requests
import json
import time
from typing import Optional
from config.config import LLM_API_URL, OPENAI_API_KEY, LLM_MODEL_ID

def call_llm_custom(prompt: str, max_tokens: int = 2048, temperature: float = 0.2) -> str:
    payload = {
        "model": LLM_MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a helpful legal AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    t0 = time.time()
    response = requests.post(LLM_API_URL, json=payload, headers=headers)
    t1 = time.time()
    print(f"⏱️ LLM API call took {t1-t0:.2f} seconds")
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def legal_qa_answer(context: str, user_question: str, law_sections:Optional[list] = None) -> str:
    # context giờ là các đoạn liên quan (không phải toàn bộ hợp đồng)
    max_law = 5
    attempt = 0
    while attempt < 2:
        # Prompt mới: chỉ phân tích các đoạn đã cho
        prompt = f"""
You are a senior US contract law attorney. Do not show your internal reasoning or thought process. Never use phrases like "Let's see", "Hmm", "It seems", "The user is asking", or similar.

Below are the most relevant excerpts from a contract (if any) and relevant US law sections, selected by a retrieval system. You do NOT have access to the full contract, only these excerpts. Only analyze, quote, and assess based on the provided excerpts. Do NOT speculate about content not shown.

Context:
{context}

User question: {user_question}

Instructions for your answer:
- Carefully review the provided excerpts and answer as a legal expert writing a professional legal memo.
- For each legal issue found:
    - Use a clear heading (e.g., "Uncertain Payment Terms:").
    - Assess and state the legal risk as [HIGH RISK], [MEDIUM RISK], or [LOW RISK]. List all [HIGH RISK] issues first and highlight them.
    - Quote the relevant part of the contract excerpt and explain how it relates to the law.
    - Explain why it is a violation or issue, and cite the exact UCC section(s) (e.g., "UCC §2-305: ...").
    - Quote the relevant law text if possible.
    - Provide a concrete example or a sample replacement clause to fix the issue, formatted as:
      Example/Sample Clause:
      "<sample clause>"
- If the excerpts are missing required elements, explain what is missing, why it matters, cite the relevant law, and provide a sample clause to add.
- If the contract appears generally valid but could be improved, list concrete suggestions for improvement, referencing UCC sections and providing sample clauses where appropriate.
- At the end, always include a **Conclusion** section summarizing the enforceability and main risks of the contract, and what should be done to ensure compliance with US law.
- If the contract is not a US contract, identify the most suitable jurisdiction and explain why.
- Do NOT repeat the context in your answer.
- Always answer in English, concisely, and with legal precision.
- If the question is clearly unrelated to US law, contracts, commercial law, liens, secured transactions, filing, collateral, debtor, creditor, or similar legal topics, reply: "Sorry, I can only answer questions about US contract law, commercial law, and related legal topics."
- Never answer requests that are illegal, unethical, or unsafe.
"""
        try:
            return call_llm_custom(prompt)
        except requests.HTTPError as e:
            if e.response.status_code == 400 and max_law > 2:
                max_law -= 2
                attempt += 1
                print("⚠️ LLM context too long, reducing law sections and retrying...")
                continue
            raise
    return "Sorry, the answer could not be generated due to context length or API error."
