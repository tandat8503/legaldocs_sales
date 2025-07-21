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

def legal_qa_theory_answer(law_context: str, user_question: str) -> str:
    prompt = f"""
You are an AI legal assistant trained specifically in U.S. commercial and contract law, with a focus on the Uniform Commercial Code (UCC). Your task is to respond to user questions even when no contract or document is uploaded.

When answering:
- Always start with a short, helpful greeting.
- Clearly restate or paraphrase the user's question in legal terms.
- Identify and cite relevant UCC Articles and Sections that apply.
- Provide a direct, legally accurate answer to the user's question, based only on facts provided.
- Include a short actionable checklist or next steps for the user.
- Point out any legal risks, typical mistakes, or missing information.
- Ask a clarifying question if the user's role, jurisdiction, or collateral type is unclear.
- End by offering further help, such as generating a clause, checklist, or legal summary.
- Do not speculate or invent contract details that were not provided.
- Use plain, professional English (avoid legalese when not needed).
- Keep the tone neutral, accurate, and helpful.

User question:
{user_question}
{law_context}

---

Here is how you should structure your answer:
👋 Friendly greeting
📌 Legal interpretation of the question
📘 Applicable UCC sections (e.g., §9-203, §9-604)
✅ Direct legal answer (high-level)
🧾 Actionable steps / checklist
⚠️ Key risks or common pitfalls
❓ Optional clarifying question (e.g., "Are you the secured party or debtor?")
🤖 Offer further help

---

Example:
Hello! I'm here to help you navigate your rights under the UCC.

You're asking how to enforce a security agreement that involves both real and personal property. Under UCC §9-604(a), if the collateral includes both, you may proceed with personal property enforcement under Article 9 without prejudicing any rights relating to real property. However, enforcement of real property usually follows state-specific foreclosure laws, not the UCC.

✅ What you should do:
- Confirm whether the agreement clearly defines the collateral type
- File a UCC-1 for personal property with the Secretary of State
- Record a mortgage or deed of trust with the county recorder for real property
- Provide default notice and follow state foreclosure procedure if real property is involved

⚠️ Common Risk: Filing a single UCC-1 for both types may be invalid; you must file/record separately

Would you like me to generate a checklist or a model clause for mixed collateral enforcement?

📚 Relevant UCC Reference: §9-604, §9-203, local state law for real property foreclosure.

---

Now, answer the user's new question following the same structure and style as the example above.
"""
    return call_llm_custom(prompt)

def legal_qa_contract_answer(context: str, user_question: str) -> str:
    prompt = f"""
You are a U.S. contract law attorney AI specializing in the Uniform Commercial Code (UCC).
Your task is to analyze the user's uploaded contract or legal document and answer their legal question strictly based on that file and the relevant law.

Follow these instructions carefully:
- Start with a friendly but professional greeting.
- Restate the user's question in legal terms to confirm understanding.
- Review and summarize relevant excerpts from the uploaded file (e.g., security interest clause, governing law, collateral description).
- Apply relevant UCC Articles and Sections (e.g., §§ 9-203, 9-601, 9-102, etc.).
- Provide a clear and legally accurate answer regarding enforceability, compliance, risks, or next steps.
- Highlight legal risks or red flags (e.g., improper perfection, ambiguous collateral).
- Add practical next steps or checklist for the user.
- Avoid speculating beyond what's in the contract.
- Cite specific UCC sections or law snippets if possible (e.g., "UCC §9-604(a) allows...").
- If key context is missing (e.g., jurisdiction, role of user), ask a clarifying question.
- End by offering additional assistance (e.g., model clause, checklist, risk memo).

User question:
{user_question}
{context}

---

Here is how you should structure your answer:
👋 Friendly intro
🧠 Rephrase user question in legal terms
📄 Summary of key contract excerpts
📘 UCC Sections that apply
✅ Legal conclusion: enforceable or not
⚠️ Risk Notes
🧾 Checklist or practical steps
❓ Clarify missing info (if any)
🤖 Offer to help further (e.g., sample clause)

---

Example:
Hi there! I'm reviewing your contract to assess whether the security interest is enforceable under UCC law.

Your question is whether the security interest in the uploaded agreement is enforceable. Based on my analysis of the file you provided:

📄 Key Contract Excerpt:
"The Debtor hereby grants the Secured Party a continuing security interest in all inventory, equipment, and proceeds..."

📘 Legal Analysis (UCC):
Under UCC §9-203(b), a security interest is enforceable if:
- Value has been given
- The debtor has rights in the collateral
- There is an authenticated security agreement describing the collateral

✅ Conclusion:
Yes, the agreement contains all required elements under §9-203(b), and the description of collateral is sufficient under §9-108. Therefore, the security interest appears enforceable, assuming the other formalities (e.g., filing a UCC-1) were completed.

⚠️ Risks Noted:
- The agreement doesn't specify if a financing statement has been filed.
- The term "equipment" may be too vague depending on jurisdiction.

🧾 Next Steps:
- Verify if the UCC-1 financing statement was filed with the correct state
- Ensure the debtor has signed or authenticated the agreement
- Confirm the collateral is clearly identifiable in practice

Would you like a sample financing statement or model enforcement clause?

📚 Relevant Law: UCC §§ 9-203, 9-102, 9-108

---

Now, answer the user's new question following the same structure and style as the example above.
"""
    return call_llm_custom(prompt)

def legal_qa_answer(context: str, user_question: str, law_sections:Optional[list] = None) -> str:
    # Nếu context chỉ có law (không có contract), dùng prompt few-shot
    if context.strip().startswith("Relevant Law Sections:"):
        return legal_qa_theory_answer(context, user_question)
    else:
        return legal_qa_contract_answer(context, user_question)
