import os
from flask import Flask, render_template, request, session, redirect, url_for
from core.rag_chain import legal_qa_answer
import fitz # PyMuPDF
import docx
from io import BytesIO
from core.semantic_search import search_law_sections
import re
from core.milvus_utilis import search_laws, search_contracts
from core.embedding import model as local_embedding_model

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Cần cho session
UPLOAD_FOLDER = 'uploaded_contracts'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text_from_file(file_storage):
    filename = file_storage.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.txt':
        return file_storage.read().decode('utf-8')
    elif ext == '.pdf':
        file_bytes = file_storage.read()
        doc = fitz.open("pdf", file_bytes)
        text = ""
        for page in doc:
            text += page.get_text("text")
        doc.close()
        return text
    elif ext == '.docx':
        file_bytes = BytesIO(file_storage.read())
        doc = docx.Document(file_bytes)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    else:
        return None

def is_english(text):
    # Simple check: only allow English letters, numbers, and common punctuation
    return re.match(r'^[A-Za-z0-9 .,;:?!\'\"()\-\n]+$', text) is not None

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ""
    error = ""
    contract_text = session.get('contract_text', '')
    relevant_law_sections = []
    # Initialize or get the QA cache from session
    qa_cache = session.get('qa_cache', {})

    if request.method == 'POST':
        # Handle finish session
        if 'finish_session' in request.form:
            session.pop('contract_text', None)
            session.pop('qa_cache', None)
            return redirect(url_for('index'))
        # Handle upload new contract
        if 'upload_new_contract' in request.form:
            session.pop('contract_text', None)
            session.pop('qa_cache', None)
            return redirect(url_for('index'))
        # Handle file upload
        contract_file = request.files.get('contract')
        if contract_file and contract_file.filename:
            contract_text = extract_text_from_file(contract_file)
            session['contract_text'] = contract_text
            qa_cache = {}  # Reset cache for new contract
        else:
            contract_text = session.get('contract_text', '')
        user_question = request.form.get('user_question', '').strip()
        if not user_question:
            error = "Question cannot be empty."
        elif len(user_question) > 500:
            error = "Question is too long (max 500 characters)."
        else:
            # Use contract_text and user_question as cache key
            cache_key = (contract_text, user_question)
            cache_key_str = str(hash(contract_text)) + '||' + user_question
            if cache_key_str in qa_cache:
                answer = qa_cache[cache_key_str]
            else:
                if contract_text:
                    # Deep search on contract and law
                    contract_sections = search_contracts(user_question, filename=None, top_k=3, model_override=local_embedding_model)
                    law_sections = search_laws(user_question, top_k=5, model_override=local_embedding_model)
                    # Combine context for LLM
                    context = contract_text + "\n\n" + "\n".join([s['chunk'] for s in law_sections])
                    relevant_law_sections = law_sections
                else:
                    # Deep search on law only
                    law_sections = search_laws(user_question, top_k=5, model_override=local_embedding_model)
                    context = "\n".join([s['chunk'] for s in law_sections])
                    relevant_law_sections = law_sections
                answer = legal_qa_answer(context, user_question, relevant_law_sections)
                if answer:
                    answer = re.sub(r'<\/?.*?think>', '', answer, flags=re.IGNORECASE).lstrip()
                if not answer or "I cannot answer" in answer or "Sorry" in answer:
                    answer = "Sorry, I cannot answer this question."
                qa_cache[cache_key_str] = answer
            session['qa_cache'] = qa_cache
    return render_template('index.html', answer=answer, contract=contract_text, error=error, relevant_law_sections=relevant_law_sections)

if __name__ == '__main__':
    app.run(debug=True, port=8050, host='0.0.0.0')