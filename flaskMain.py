#make more interactive

import os

import faiss
import numpy as np
import openai
from PyPDF2 import PdfReader
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"txt", "pdf", "md"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OpenAI API key! Set OPENAI_API_KEY as an environment variable.")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

D = 384
index = faiss.IndexFlatL2(D)
documents = []


def extract_examples(text):
    # Define a pattern that captures "Example: ..." type sections
    # example_pattern = r"(Example\s*\d*:?|Problem\s*\d*:?).*?(?=\nExample\s*\d*:|\nProblem\s*\d*:|\Z)"
    #
    # examples = re.findall(example_pattern, text, re.DOTALL)

    examples = text.split('Problem')
    # Clean up extracted examples
    cleaned_examples = [ex.strip() for ex in examples if len(ex.strip()) > 20]  # Filter out too-short snippets
    print(cleaned_examples)
    return cleaned_examples


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    return text


def add_document_to_index(text):
    examples = extract_examples(text)

    if not examples:
        print("No valid examples found")
        return

    embeddings = embedding_model.encode(examples)
    index.add(np.array(embeddings, dtype=np.float32))
    documents.extend(examples)


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        print("No File Part")
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        text = extract_text(file_path)
        if text:
            add_document_to_index(text)
            return jsonify({"message": "File uploaded and processed successfully!"})
        else:
            return jsonify({"error": "Could not extract text from file"}), 400

    return jsonify({"error": "Invalid file format"}), 400


def find_relevant_examples(query):
    """Retrieves the most relevant examples using FAISS & keyword filtering."""
    query_embedding = embedding_model.encode([query])[0]
    D, I = index.search(np.array([query_embedding], dtype=np.float32), k=5)

    candidate_examples = [documents[i] for i in I[0] if i < len(documents)]

    query_keywords = set(query.lower().split())
    refined_results = sorted(
        candidate_examples,
        key=lambda ex: sum(1 for word in query_keywords if word in ex.lower()),
        reverse=True
    )

    return "\n\n".join(refined_results[:3]) if refined_results else ""



def get_openai_response(messages):
    """Fetches a response from OpenAI's API given messages."""
    client = openai.Client(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()


@app.route("/api/analyze", methods=["POST"])
def analyze_lambda():
    data = request.json
    expression = data.get("expression", "")

    # Find relevant examples
    relevant_examples = find_relevant_examples(expression)
    print(relevant_examples)

    prompt = f"""
    Break down the following problem step by step.
    Follow the structure and depth of the relevant examples provided below.

    Relevant Examples:
    {relevant_examples}

    Now, apply the same level of detail to this problem:
    Expression: {expression}
    """

    messages = [
        {"role": "system",
         "content": "You are a tutor helping students understand concepts in a Programming Languages and Concepts Class by breaking problems into simpler steps."},
        {"role": "user", "content": prompt}
    ]

    response = get_openai_response(messages)
    return jsonify({"content": response})


if __name__ == "__main__":
    app.run(debug=True, port=5000)