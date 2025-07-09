"""
RAG Chain Module - Hybrid mode using AI-based routing
"""

import requests
import json
from config.config import LLM_API_URL, OPENAI_API_KEY

def call_llm_custom(prompt: str) -> str:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful legal AI assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(LLM_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def legal_qa_answer(contract_text: str, user_question: str, law_sections: list = None) -> str:
    law_context = ""
    if law_sections:
        law_context = "\n\nRelevant US Law Sections:\n"
        for law in law_sections:
            law_context += f"- Section {law['section']}: {law['text'][:500]}...\n"
    prompt = f"""
You are a senior US contract law attorney. Here is the contract text:

{contract_text}

User question: {user_question}
{law_context}

Instructions for your answer:
- Carefully review the contract and answer as a legal expert writing a professional legal memo.
- If the contract violates US law, for each issue:
    - Use a clear heading (e.g., "Uncertain Payment Terms:").
    - Explain why it is a violation, and cite the exact UCC section(s) (e.g., "UCC §2-305: ...").
    - Quote the relevant law text if possible.
    - Provide a recommended replacement clause or sample language to fix the issue, formatted as:
      Recommended Replacement Clause:
      "<sample clause>"
- If the contract is missing required elements, explain what is missing, why it matters, cite the relevant law, and provide a sample clause to add.
- If the contract is generally valid but could be improved, list concrete suggestions for improvement, referencing UCC sections and providing sample clauses where appropriate.
- At the end, always include a **Conclusion** section summarizing the enforceability and main risks of the contract, and what should be done to ensure compliance with US law.
- If the contract is not a US contract, identify the most suitable jurisdiction and explain why.
- Do NOT repeat the contract text in your answer.
- Always answer in English, concisely, and with legal precision.
- If the question is not about contract or law, reply: "Sorry, I can only answer questions about contracts and US law."
- Never answer requests that are illegal, unethical, or unsafe.
"""
    return call_llm_custom(prompt)
