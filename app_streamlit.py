import streamlit as st
import os
from core.rag_chain import legal_qa_answer, call_llm_custom
from core.semantic_search import search_law_sections
from core.milvus_utilis import search_laws, search_contracts
from core.embedding import model as local_embedding_model
import fitz  # PyMuPDF
import docx
from io import BytesIO
import re

st.set_page_config(page_title="Legal Contract AI Agent", layout="centered")
st.title("Legal Contract AI Agent")

UPLOAD_FOLDER = 'uploaded_contracts'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Utility functions ---
def extract_text_from_file(file):
    filename = file.name
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.txt':
        return file.read().decode('utf-8')
    elif ext == '.pdf':
        file_bytes = file.read()
        doc = fitz.open("pdf", file_bytes)
        text = ""
        for page in doc:
            text += page.get_text("text")
        doc.close()
        return text
    elif ext == '.docx':
        file_bytes = BytesIO(file.read())
        doc = docx.Document(file_bytes)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    else:
        return None

def is_contract_text(text):
    contract_keywords = ["agreement", "contract", "party", "obligation", "term", "warranty", "governing law", "indemnify", "confidentiality", "termination", "liability"]
    return any(kw in text.lower() for kw in contract_keywords)

def is_contract_analysis_question(question):
    # Nhận diện các câu hỏi yêu cầu phân tích hợp đồng cụ thể
    contract_phrases = [
        "this contract", "the contract", "in the contract", "analyze contract", "analyze this contract", "what does the contract", "what do contract have", "what is in the contract", "terms of the contract", "contract content", "contract clause", "contract risk", "contract provision", "contract section", "contract analysis"
    ]
    q = question.lower()
    return any(phrase in q for phrase in contract_phrases)

def is_clearly_nonlegal_question(question):
    # Một số từ khóa phổ biến ngoài phạm vi pháp lý (đã loại bỏ các từ liên quan đến stock, share, investment, finance, bank, money, currency, exchange, loan, credit, debt, insurance, tax, account, budget, income, expense, salary, wage, payment, bill, invoice, receipt, sell, buy, sale, market, shop, store, price, discount)
    unrelated_keywords = [
        'cook', 'recipe', 'food', 'weather', 'football', 'soccer', 'movie', 'music', 'game', 'travel', 'vacation', 'holiday', 'song', 'singer', 'actor', 'actress', 'film', 'youtube', 'tiktok', 'instagram', 'facebook', 'twitter', 'garden', 'plant', 'animal', 'pet', 'cat', 'dog', 'fish', 'bird', 'car', 'motorbike', 'bike', 'run', 'swim', 'gym', 'exercise', 'workout', 'draw', 'paint', 'art', 'fashion', 'clothes', 'dress', 'shoes', 'makeup', 'beauty', 'hair', 'skin', 'shopping', 'supermarket', 'mall', 'school', 'university', 'exam', 'test', 'teacher', 'student', 'math', 'physics', 'chemistry', 'biology', 'history', 'geography', 'poem', 'poetry', 'novel', 'story', 'book', 'author', 'writer', 'birthday', 'party', 'gift', 'present', 'love', 'relationship', 'marriage', 'dating', 'friend', 'family', 'parent', 'child', 'baby', 'kid', 'boy', 'girl', 'man', 'woman', 'husband', 'wife', 'brother', 'sister', 'uncle', 'aunt', 'grandparent', 'grandchild', 'neighbor', 'community', 'city', 'village', 'country', 'nation', 'government', 'politics', 'president', 'prime minister', 'minister', 'king', 'queen', 'prince', 'princess', 'army', 'military', 'war', 'peace', 'religion', 'god', 'buddha', 'jesus', 'allah', 'temple', 'church', 'mosque', 'pagoda', 'festival', 'holiday', 'event', 'concert', 'show', 'exhibition', 'conference', 'meeting', 'seminar', 'workshop', 'lecture', 'speech', 'presentation', 'news', 'newspaper', 'magazine', 'radio', 'tv', 'television', 'channel', 'program', 'series', 'episode', 'season', 'advertisement', 'ad', 'commercial', 'promotion', 'campaign', 'project', 'plan', 'goal', 'target', 'result', 'success', 'failure', 'problem', 'solution', 'idea', 'innovation', 'technology', 'science', 'computer', 'phone', 'tablet', 'laptop', 'internet', 'website', 'web', 'app', 'application', 'software', 'hardware', 'device', 'tool', 'machine', 'robot', 'ai', 'artificial intelligence', 'blockchain', 'crypto', 'bitcoin', 'ethereum'
    ]
    q = question.lower()
    return any(kw in q for kw in unrelated_keywords)

def classify_question_with_llm(question: str, contract_uploaded: bool) -> str:
    prompt = f"""
Classify the user's question into one of these categories:
1.  contract_analysis: The user is asking to analyze, review, or find information within a specific contract.
2.  general_legal_query: The user is asking a general question about law, UCC, etc., not specific to an uploaded document.
3.  non_legal: The question is off-topic (e.g., about cooking, weather, sports).

The user has {'UPLOADED' if contract_uploaded else 'NOT UPLOADED'} a contract.

User Question: "{question}"

Category:
"""
    response = call_llm_custom(prompt, max_tokens=10, temperature=0.0)
    return response.strip().lower()

# --- Session state ---
if 'contract_text' not in st.session_state:
    st.session_state['contract_text'] = ''
if 'contract_filename' not in st.session_state:
    st.session_state['contract_filename'] = ''
if 'qa_cache' not in st.session_state:
    st.session_state['qa_cache'] = {}
if 'question_history' not in st.session_state:
    st.session_state['question_history'] = []

# --- UI ---
with st.form("qa_form"):
    # Bỏ các button chọn vai trò người dùng
    contract_file = st.file_uploader("Upload contract (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], key="contract")
    if st.session_state['contract_text']:
        st.info(f"You are asking about contract: **{st.session_state['contract_filename']}**")
    user_question = st.text_area("Enter your legal question about the contract:", value="", height=120, key="user_question")
    submit_btn = st.form_submit_button("Submit Question")
    finish_btn = None
    upload_new_btn = None
    if st.session_state.get('contract_text') or st.session_state.get('last_answered', False):
        col1, col2 = st.columns(2)
        with col1:
            finish_btn = st.form_submit_button("Finish Session")
        with col2:
            upload_new_btn = st.form_submit_button("Upload new contract")

# --- Handle session reset ---
if finish_btn or upload_new_btn:
    st.session_state['contract_text'] = ''
    st.session_state['contract_filename'] = ''
    st.session_state['qa_cache'] = {}
    st.session_state['question_history'] = []
    st.session_state['last_answered'] = False
    st.rerun()

answer = ""
error = ""
relevant_law_sections = []

# --- Handle file upload ---
if contract_file is not None:
    # Check file size (max 5MB)
    contract_file.seek(0, os.SEEK_END)
    file_size = contract_file.tell()
    contract_file.seek(0)
    max_size = 5 * 1024 * 1024  # 5MB
    if file_size > max_size:
        st.warning("The uploaded file is too large (max 5MB). Please upload a smaller contract file.")
        st.stop()
    # Check file extension
    filename = contract_file.name
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ['.txt', '.pdf', '.docx']:
        st.warning("Unsupported file type. Please upload a .txt, .pdf, or .docx contract.")
        st.stop()
    contract_text = extract_text_from_file(contract_file)
    if not contract_text or not contract_text.strip():
        st.warning("The uploaded contract file is empty. Please upload a valid contract.")
        st.stop()
    st.session_state['contract_text'] = contract_text
    st.session_state['contract_filename'] = contract_file.name
    st.session_state['qa_cache'] = {}
    st.session_state['question_history'] = []
    if contract_text and not is_contract_text(contract_text):
        st.warning("The uploaded file does not appear to be a contract. Please upload a valid contract for legal analysis.")
        st.stop()

# --- Main QA logic ---
if submit_btn:
    contract_text = st.session_state.get('contract_text', '')
    contract_filename = st.session_state.get('contract_filename', '')
    qa_cache = st.session_state.get('qa_cache', {})
    question_history = st.session_state.get('question_history', [])
    user_question = st.session_state.get('user_question', '').strip()
    if not user_question:
        error = "Question cannot be empty."
        st.error(error)
        st.stop()
    elif len(user_question) > 500:
        error = "Question is too long (max 500 characters)."
        st.error(error)
        st.stop()
    # Phân loại intent bằng LLM
    question_type = classify_question_with_llm(user_question, bool(contract_text))
    if question_type == 'non_legal':
        st.warning("Your question appears unrelated to legal or contract topics. Please ask about contracts, US law, UCC, liens, secured transactions, or related legal topics.")
        st.session_state['last_answered'] = True
        st.stop()
    elif question_type == 'contract_analysis' and not contract_text:
        st.warning("You have not uploaded the contract yet. Please upload the contract so I can analyze it in detail.")
        st.session_state['last_answered'] = True
        st.stop()
    # Use contract_text and user_question as cache key
    cache_key_str = str(hash(contract_text)) + '||' + user_question
    if cache_key_str in qa_cache:
        answer = qa_cache[cache_key_str]
        answer_type = st.session_state.get('last_answer_type', '')
    else:
        with st.spinner("Processing..."):
            if contract_text and question_type == 'contract_analysis':
                relevant_contract_chunks = search_contracts(user_question, filename=contract_filename, top_k=3, model_override=local_embedding_model)
                law_sections = search_laws(user_question, top_k=5, model_override=local_embedding_model)
                contract_context = "\n\n".join([c['chunk'] for c in relevant_contract_chunks])
                law_context_for_llm = "\n\n".join([s['chunk'] for s in law_sections])
                context_for_llm = f"Relevant Contract Sections:\n{contract_context}\n\nRelevant Law Sections:\n{law_context_for_llm}"
                relevant_law_sections = law_sections
                answer = legal_qa_answer(context_for_llm, user_question, law_sections)
                answer_type = 'contract_analysis'
            else:
                law_sections = search_laws(user_question, top_k=5, model_override=local_embedding_model)
                law_context_for_llm = "\n\n".join([s['chunk'] for s in law_sections])
                context_for_llm = f"Relevant Law Sections:\n{law_context_for_llm}"
                relevant_law_sections = law_sections
                answer = legal_qa_answer(context_for_llm, user_question, law_sections)
                answer_type = 'general_legal_query'
            if answer:
                answer = re.sub(r'<\/?.*?think>', '', answer, flags=re.IGNORECASE).lstrip()
            if not answer or "I cannot answer" in answer or "Sorry" in answer:
                answer = "Sorry, I cannot answer this question."
            qa_cache[cache_key_str] = answer
            st.session_state['last_answer_type'] = answer_type
    st.session_state['qa_cache'] = qa_cache
    st.session_state['last_answered'] = True
    # Update question history
    if contract_text and qa_cache:
        prefix = str(hash(contract_text)) + '||'
        question_history = [key.split('||', 1)[1] for key in qa_cache if key.startswith(prefix)]
        st.session_state['question_history'] = question_history
    # Show answer
    st.success("**Answer:**")
    # Ghi chú rõ nguồn trả lời
    if contract_text and question_type == 'contract_analysis':
        st.info("Analysis based on uploaded contract: " + st.session_state.get('contract_filename', ''))
    elif question_type == 'general_legal_query':
        st.info("General legal answer (no contract uploaded)")
    st.markdown(f"<div style='white-space: pre-wrap'>{answer}</div>", unsafe_allow_html=True)
    # Show relevant law sections in expander
    if relevant_law_sections:
        with st.expander("**📚 Relevant Law References**"):
            for section in relevant_law_sections:
                # Extract article and section numbers from filename (e.g., "9-523.txt" -> "Article 9, Section 523")
                filename = section['filename']
                article_num = filename.split('-')[0]
                section_num = filename.split('-')[1].replace('.txt', '')
                st.markdown(f"""
                **UCC Article {article_num}, Section {section_num}**  
                {section['chunk'][:300]}{'...' if len(section['chunk']) > 300 else ''}
                """)
    else:
        st.info("No relevant law sections found for your question.")
    
    # Offer additional resources based on answer type
    if answer and 'Would you like me to provide:' in answer:
        st.write("---")
        st.write("### Additional Resources")
        resource_col1, resource_col2 = st.columns(2)
        with resource_col1:
            if st.button("📋 Get Detailed Checklist"):
                st.session_state['requested_resource'] = 'checklist'
        with resource_col2:
            if st.button("📄 View Sample Templates"):
                st.session_state['requested_resource'] = 'templates'
    # Show question history
    if contract_text and question_history:
        st.info("**Question history for this contract:**")
        for q in question_history:
            st.markdown(f"- {q}") 