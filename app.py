from flask import Flask, request, jsonify, render_template, session
import os, uuid, tempfile
from rag_engine import RAGEngine

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "academic-assistant-secret-2024")

# In-memory store: session_id -> RAGEngine instance
engines = {}

@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    sid = session.get("sid", str(uuid.uuid4()))
    session["sid"] = sid

    api_key = request.form.get("api_key", "").strip()
    file = request.files.get("pdf")

    if not api_key:
        return jsonify({"error": "Please provide your OpenAI API key."}), 400
    if not file or not file.filename.endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF file."}), 400

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        file.save(tmp.name)
        tmp.close()

        engine = RAGEngine(api_key=api_key)
        engine.load_document(tmp.name)
        engines[sid] = engine
        os.unlink(tmp.name)

        return jsonify({"success": True, "filename": file.filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    sid = session.get("sid")
    engine = engines.get(sid)
    if not engine:
        return jsonify({"error": "No document loaded. Please upload a PDF first."}), 400

    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        answer = engine.ask(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/summarize", methods=["POST"])
def summarize():
    sid = session.get("sid")
    engine = engines.get(sid)
    if not engine:
        return jsonify({"error": "No document loaded."}), 400

    detail = request.json.get("detail", "Medium (1–2 paragraphs)")
    try:
        summary = engine.summarize(detail)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/quiz", methods=["POST"])
def quiz():
    sid = session.get("sid")
    engine = engines.get(sid)
    if not engine:
        return jsonify({"error": "No document loaded."}), 400

    data = request.json
    quiz_type = data.get("quiz_type", "Multiple Choice (MCQ)")
    num_q = int(data.get("num_questions", 5))
    try:
        result = engine.generate_quiz(quiz_type, num_q)
        return jsonify({"quiz": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    sid = session.get("sid")
    if sid and sid in engines:
        del engines[sid]
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))