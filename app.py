import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import PyPDF2
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# CONFIG
app = Flask(__name__)
app.secret_key = "YOUR_FLASK_SECRET_KEY"
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
app.config["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"  # Add your API key here

# Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# PDF Storage
pdf_embeddings_store = []
pdf_metadata = {}  # To store PDF-specific info like name and author

# Initialize Gemini
genai.configure(api_key=app.config["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

# Updated safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

# EXTRACT TEXT, EMBED, AND ANALYZE METADATA
def extract_and_embed_pdf(pdf_path, pdf_filename):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        # Extract metadata (e.g., author, title)
        pdf_info = reader.metadata or {}
        author = pdf_info.get("/Author", "Unknown")
        pdf_metadata[pdf_filename] = {"name": pdf_filename, "author": author}
        
        # Process content
        full_text = []
        for page_index, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            lines = raw_text.split("\n")
            for line_number, line_text in enumerate(lines, start=1):
                text = line_text.strip()
                if not text:
                    continue
                embedding = embedding_model.encode(text, convert_to_tensor=True)
                pdf_embeddings_store.append({
                    "pdf_name": pdf_filename,
                    "page_number": page_index + 1,
                    "line_number": line_number,
                    "text": text,
                    "embedding": embedding
                })
                full_text.append(text)
        pdf_metadata[pdf_filename]["full_text"] = full_text  # For summarization

# INDEX ROUTE: UPLOAD OR SHOW
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        extract_and_embed_pdf(save_path, filename)
        return "File uploaded and processed!", 200
    return render_template("index.html", pdf_files=_list_uploaded_pdfs())

def _list_uploaded_pdfs():
    pdf_files = []
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        return pdf_files
    for fname in os.listdir(app.config["UPLOAD_FOLDER"]):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            pdf_files.append({"name": fname, "size": round(size_mb, 2)})
    return pdf_files

# CHAT ROUTE
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "I didn't quite catch that. Could you please repeat your question?"})
    
    try:
        # First check for PDF-specific questions
        pdf_response = None
        if "author" in user_message.lower():
            pdf_response = handle_author_question()
        elif "name" in user_message.lower():
            pdf_response = handle_name_question()
        elif "summary" in user_message.lower():
            pdf_response = handle_summary_question()
        elif pdf_embeddings_store:
            pdf_response = find_best_pdf_answer(user_message)
        
        if pdf_response:
            return jsonify({"response": pdf_response})
        
        # If no PDF-specific answer, use Gemini for a natural conversation
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40
        }
        
        prompt = {
            "contents": [{
                "parts": [{"text": f"You are a helpful assistant having a conversation. Respond naturally to: {user_message}"}]
            }]
        }

        gemini_response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return jsonify({
            "response": gemini_response.text
        })
        
    except Exception as e:
        return jsonify({
            "response": "I'm having trouble processing that right now. Could you try asking your question differently?"
        })

# HUMAN-LIKE QUESTION HANDLERS
def handle_name_question():
    if not pdf_metadata:
        return "I don't have any PDFs loaded yet, so there's no name to share."
    # Assume latest PDF if not specified (human-like assumption)
    latest_pdf = list(pdf_metadata.keys())[-1]
    return f"The name of the PDF I've got here is '{pdf_metadata[latest_pdf]['name']}'."

def handle_author_question():
    if not pdf_metadata:
        return "I haven't seen any PDFs yet, so I can't tell you about an author."
    latest_pdf = list(pdf_metadata.keys())[-1]
    author = pdf_metadata[latest_pdf]["author"]
    return f"The author of '{latest_pdf}' is {author if author != 'Unknown' else 'not specified in the PDF'}."

def handle_summary_question():
    if not pdf_metadata:
        return "There's no PDF for me to summarize yet. Give me one to read first!"
    latest_pdf = list(pdf_metadata.keys())[-1]
    full_text = pdf_metadata[latest_pdf]["full_text"]
    # Simple summarization: first few lines + key points (mimicking human skimming)
    summary_length = min(3, len(full_text))
    summary = " ".join(full_text[:summary_length])
    if len(full_text) > summary_length:
        summary += "… It covers a lot more, but that's the gist of the start!"
    return f"Here's a quick summary of '{latest_pdf}': {summary}"

# FIND BEST PDF ANSWER
def find_best_pdf_answer(query, similarity_threshold=0.4):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    all_embeddings = [entry["embedding"] for entry in pdf_embeddings_store]
    scores = util.cos_sim(query_embedding, torch.stack(all_embeddings))[0]
    
    top_idx = scores.argmax().item()
    top_score = float(scores[top_idx])
    if top_score < similarity_threshold:
        return None
    
    best_entry = pdf_embeddings_store[top_idx]
    return (
        f"Based on what I found in '{best_entry['pdf_name']}', here's your answer: {best_entry['text']}. "
        f"That's from page {best_entry['page_number']}, line {best_entry['line_number']}."
    )

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return '', 204  # Success with no content
        return 'File not found', 404
    except Exception as e:
        return str(e), 500

# MAIN
if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)