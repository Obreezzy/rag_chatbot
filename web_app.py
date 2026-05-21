from flask import Flask, request, jsonify, render_template
from flask import send_from_directory
import sys
import os

# Add rag_project to path so we can import from rag_system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import ask_datacompany_with_memory, embedder, collection

app = Flask(__name__)

# Global conversation history
conversation_history = []


@app.route("/")
def home():
    return render_template("chat.html")

SUGGESTIONS_MAP = {
    "rent":      ["Can I get a payment plan?", "What happens after 3 missed payments?"],
    "deposit":   ["What deductions can DataCompany make?", "How do I dispute a deduction?"],
    "repair":    ["What is the emergency repair number?", "What repairs am I responsible for?"],
    "pet":       ["What pets are not allowed?", "Are assistance dogs permitted?"],
    "evict":     ["What are my rights during eviction?", "How much notice must DataCompany give?"],
    "noise":     ["What counts as anti-social behaviour?", "How do I report a noise complaint?"],
    "complaint": ["How long does DataCompany take to respond?", "Can I go to the Housing Ombudsman?"],
    "sublet":    ["Can I have a lodger?", "Can I use Airbnb?"],
    "inspect":   ["How much notice before an inspection?", "What happens if I fail an inspection?"],
    "tenancy":   ["What is my notice period to leave?", "What are my obligations as a tenant?"],
}

def get_suggestions(answer_text):
    answer_lower = answer_text.lower()
    for keyword, suggestions in SUGGESTIONS_MAP.items():
        if keyword in answer_lower:
            return suggestions
    return []

@app.route("/ask", methods=["POST"])
def ask():
    global conversation_history

    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please ask a question."})

    # Handle memory clear command
    if question.lower() == "clear":
        conversation_history = []
        return jsonify({"answer": "Memory cleared.", "best_source": ""})

    # Get answer from RAG system
    result, conversation_history = ask_datacompany_with_memory(
        question,
        conversation_history
    )
    NOT_FOUND = "this information is not covered"
    suggestions = []
    if NOT_FOUND not in result["answer"].lower():
        suggestions = get_suggestions(result["answer"])
    result["suggestions"] = suggestions

    return jsonify(result)

@app.route("/policies/<filename>")
def serve_policy(filename):
    """Serves PDF policy files inline in the browser."""
    from flask import send_from_directory
    # Ensure as_attachment is False
    return send_from_directory("policies", filename, 
                               mimetype="application/pdf", 
                               as_attachment=False)

@app.route("/api/get-pdf", methods=["POST"])
def get_pdf_post():
    """Endpoint to fetch PDF via POST to bypass download managers like IDM."""
    data = request.json
    filename = data.get("filename")
    from flask import send_from_directory
    return send_from_directory("policies", filename, mimetype="application/pdf", as_attachment=False)


if __name__ == "__main__":
    print("\nDataCompany RAG Web Interface starting...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)