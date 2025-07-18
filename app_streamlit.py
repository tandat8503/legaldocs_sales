import streamlit as st
import os
from core.rag_chain import legal_qa_answer
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

def is_legal_question(question):
    legal_keywords = ["law", "ucc", "contract", "agreement", "clause", "legal", "section", "article", "obligation", "liability", "warranty", "governing law"]
    return any(kw in question.lower() for kw in legal_keywords)

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
    contract_text = extract_text_from_file(contract_file)
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
    elif not is_legal_question(user_question):
        answer = "Sorry, I can only answer questions about contracts and US law."
        st.info(answer)
        st.session_state['last_answered'] = True
        st.stop()
    # Use contract_text and user_question as cache key
    cache_key_str = str(hash(contract_text)) + '||' + user_question
    if cache_key_str in qa_cache:
        answer = qa_cache[cache_key_str]
    else:
        with st.spinner("Processing..."):
            if contract_text:
                contract_sections = search_contracts(user_question, filename=None, top_k=3, model_override=local_embedding_model)
                law_sections = search_laws(user_question, top_k=5, model_override=local_embedding_model)
                context = contract_text + "\n\n" + "\n".join([s['chunk'] for s in law_sections])
                relevant_law_sections = law_sections
            else:
                law_sections = search_laws(user_question, top_k=5, model_override=local_embedding_model)
                context = "\n".join([s['chunk'] for s in law_sections])
                relevant_law_sections = law_sections
            answer = legal_qa_answer(context, user_question, relevant_law_sections)
            if answer:
                answer = re.sub(r'<\/? .*?think>', '', answer, flags=re.IGNORECASE).lstrip()
            if not answer or "I cannot answer" in answer or "Sorry" in answer:
                answer = "Sorry, I cannot answer this question."
            qa_cache[cache_key_str] = answer
    st.session_state['qa_cache'] = qa_cache
    st.session_state['last_answered'] = True
    # Update question history
    if contract_text and qa_cache:
        prefix = str(hash(contract_text)) + '||'
        question_history = [key.split('||', 1)[1] for key in qa_cache if key.startswith(prefix)]
        st.session_state['question_history'] = question_history
    # Show answer
    st.success("**Answer:**")
    st.markdown(f"<div style='white-space: pre-wrap'>{answer}</div>", unsafe_allow_html=True)
    # Show relevant law sections
    if relevant_law_sections:
        st.info("**Relevant Law Sections:**")
        for section in relevant_law_sections:
            st.markdown(f"- **File:** {section['filename']}\n  **Content:** {section['chunk'][:300]}{'...' if len(section['chunk']) > 300 else ''}")
    # Show question history
    if contract_text and question_history:
        st.info("**Question history for this contract:**")
        for q in question_history:
            st.markdown(f"- {q}") 