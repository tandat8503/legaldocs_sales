import os
from flask import Flask, render_template, request, session, redirect, url_for
from core.rag_chain import legal_qa_answer
import fitz # PyMuPDF
import docx
from io import BytesIO
from core.semantic_search import search_law_sections
import re

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

    if request.method == 'POST':
        # Handle finish session
        if 'finish_session' in request.form:
            session.pop('contract_text', None)
            return redirect(url_for('index'))
        # Handle upload new contract
        if 'upload_new_contract' in request.form:
            session.pop('contract_text', None)
            return redirect(url_for('index'))
        # Handle file upload
        contract_file = request.files.get('contract')
        if contract_file and contract_file.filename:
            contract_text = extract_text_from_file(contract_file)
            session['contract_text'] = contract_text
        else:
            contract_text = session.get('contract_text', '')
        user_question = request.form.get('user_question', '').strip()
        if not contract_text:
            error = "No contract uploaded."
        elif not user_question:
            error = "Question cannot be empty."
        elif len(user_question) > 500:
            error = "Question is too long (max 500 characters)."
        else:
            # Tìm các section luật liên quan nhất với hợp đồng
            relevant_law_sections = search_law_sections(contract_text, top_k=5)
            answer = legal_qa_answer(contract_text, user_question, relevant_law_sections)
            # Loại bỏ dòng <think> hoặc </think> nếu có
            if answer:
                answer = re.sub(r'<\/?think>', '', answer, flags=re.IGNORECASE).lstrip()
            if not answer or "I cannot answer" in answer or "Sorry" in answer:
                answer = "Sorry, I cannot answer this question."
    return render_template('index.html', answer=answer, contract=contract_text, error=error, relevant_law_sections=relevant_law_sections)

if __name__ == '__main__':
    app.run(debug=True, port=8050, host='0.0.0.0')