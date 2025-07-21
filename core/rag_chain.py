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

def legal_qa_answer(contract_text: str, user_question: str, law_sections:Optional[list] = None) -> str:
    law_context = ""
    max_law = 5
    attempt = 0
    while attempt < 2:
        if law_sections:
            law_context = "\n\nRelevant US Law Sections:\n"
            for law in law_sections[:max_law]:
                law_context += f"- Section {law['filename']}: {law['chunk'][:500]}...\n"

        # Nếu có contract_text (hợp đồng), dùng prompt phân tích hợp đồng
        if contract_text and contract_text.strip():
            prompt = f"""
            You are a senior US contract law attorney. Do not show your internal reasoning or thought process. Never use phrases like "Let's see", "Hmm", "It seems", "The user is asking", or similar.

            Here is the contract text:

{contract_text}

User question: {user_question}
{law_context}

Instructions for your answer:
- Carefully review the contract and answer as a legal expert writing a professional legal memo.
            - For each legal issue found:
    - Use a clear heading (e.g., "Uncertain Payment Terms:").
                - Assess and state the legal risk as [HIGH RISK], [MEDIUM RISK], or [LOW RISK]. List all [HIGH RISK] issues first and highlight them.
                - Quote the relevant part of the contract (Contract Excerpt) and explain how it relates to the law.
                - Explain why it is a violation or issue, and cite the exact UCC section(s) (e.g., "UCC §2-305: ...").
    - Quote the relevant law text if possible.
                - Provide a concrete example or a sample replacement clause to fix the issue, formatted as:
                Example/Sample Clause:
      "<sample clause>"
- If the contract is missing required elements, explain what is missing, why it matters, cite the relevant law, and provide a sample clause to add.
- If the contract is generally valid but could be improved, list concrete suggestions for improvement, referencing UCC sections and providing sample clauses where appropriate.
- At the end, always include a **Conclusion** section summarizing the enforceability and main risks of the contract, and what should be done to ensure compliance with US law.
- If the contract is not a US contract, identify the most suitable jurisdiction and explain why.
- Do NOT repeat the contract text in your answer.
- Always answer in English, concisely, and with legal precision.
            - If the question is clearly unrelated to US law, contracts, commercial law, liens, secured transactions, filing, collateral, debtor, creditor, or similar legal topics, reply: "Sorry, I can only answer questions about US contract law, commercial law, and related legal topics."
            - Never answer requests that are illegal, unethical, or unsafe.
            """
        else:
            # Nếu không có contract_text, dùng prompt cho câu hỏi lý thuyết
            prompt = f"""
    You are a legal AI assistant specializing in US contract law, the Uniform Commercial Code (UCC), liens, secured transactions, and related commercial law topics. Do not show your internal reasoning or thought process. Never use phrases like "Let's see", "Hmm", "It seems", "The user is asking", or similar.

    User question: {user_question}
    {law_context}

    Instructions:
    - Always start your answer with an **Executive Summary**: a concise paragraph summarizing the main risks, legal gaps, and scope of the analysis.
    - Add a section **Applicability**: clarify the scope of the memo (e.g., "This memo assumes the securities are governed by UCC Article 8 and not exempted under federal securities law. In broker-dealer relationships, federal law may preempt state UCC provisions.")
    - In the main body, clearly distinguish between **Attachment** and **Perfection** of security interests, citing relevant UCC sections where appropriate.
    - Present a **Risk Matrix Table** with columns: Issue, Risk Level, Impact, UCC Reference. Use markdown table format.
    - Provide a section **Parties' Obligations / Next Steps**: bullet-point actionable recommendations for each party.
    - Use clear headings: Executive Summary, Applicability, Key Legal Elements, Risk Matrix, Next Steps, Conclusion.
    - Be concise, clear, and professional. Do not include any internal monologue or speculation.
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
