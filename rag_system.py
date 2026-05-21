from groq import Groq
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import os
import time
import re

load_dotenv()
client = Groq()

print("DataCompany RAG Policy Chatbot")
print("="*55)

# ============================================================
# STEP 1 - Load embedding model
# ============================================================

print("\nStep 1: Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")

# ============================================================
# STEP 2 - Set up vector database
# ============================================================

print("\nStep 2: Setting up vector database...")
db_client = chromadb.Client()
collection = db_client.create_collection("datacompany_rag")

# ============================================================
# STEP 3 - Load and chunk documents
# ============================================================

def load_pdf(filepath):
    """
    Reads a PDF and returns all text including table content.
    Uses pdfplumber which handles tables far better than pypdf.
    """
    import pdfplumber

    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_text = " | ".join(
                        cell.strip() for cell in row if cell
                    )
                    if row_text:
                        text += row_text + "\n"

    # Fix broken words from narrow table cells
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'(\w)\n(\w)', r'\1\2', text)

    # Normalize whitespace
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)

    return text


def load_text(filepath):
    """Reads a plain text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, chunk_size=400, overlap=50):
    """
    Splits a long document into overlapping chunks.
    Overlap ensures no information is lost at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ============================================================
# UTILITY FUNCTIONS - defined before they are used
# ============================================================

def get_confidence(distance):
    """
    Converts ChromaDB distance score to human readable confidence.
    Lower distance = more similar = higher confidence.
    """
    if distance < 0.3:
        return "DIRECT MATCH"
    elif distance < 0.6:
        return "STRONG MATCH"
    elif distance < 0.9:
        return "GOOD MATCH"
    elif distance < 1.2:
        return "PARTIAL MATCH"
    else:
        return "EXPANDED MATCH"


def expand_query(question):
    """
    Generates up to 3 alternative phrasings of the question
    to improve retrieval accuracy when wording does not match
    the PDF language exactly.
    """
    expansions = [question]

    replacements = {
        "stages":              "sanctions and warnings stage 1 stage 2 stage 3 stage 4",
        "stages of action":    "stage 1 warning letter stage 2 formal hearing stage 3 enforcement",
        "fix":                 "repair and resolve",
        "miss rent":           "fail to pay rent arrears",
        "leave":               "vacate and end tenancy",
        "kick out":            "eviction proceedings",
        "deposit":             "security deposit return",
        "complaints":          "formal warnings issued",
        "enter":               "access and inspection rights",
        "asb":                 "anti-social behaviour sanctions stage warning letter",
        "anti-social":         "stage 1 warning stage 2 formal hearing eviction proceedings",
        "noise":               "anti-social behaviour formal warning stage sanctions",
        "action does":         "sanctions warnings stage 1 stage 2 stage 3 stage 4",
        "end of tenancy":    "notice to vacate end tenancy termination procedure",
        "what about end":    "end of tenancy notice period termination eviction",
    }

    expanded = question.lower()
    for term, replacement in replacements.items():
        if term in expanded:
            new_q = expanded.replace(term, replacement)
            if new_q not in expansions:
                expansions.append(new_q)

    return expansions[:3]


# ============================================================
# LOAD DOCUMENTS
# ============================================================

print("\nStep 3: Loading DataCompany policy documents...")

policies_folder = "policies"
all_chunks = []
all_ids = []
all_metadata = []
chunk_counter = 0

policy_files = []
if os.path.exists(policies_folder):
    policy_files = [f for f in os.listdir(policies_folder)
                    if f.endswith(".pdf") or f.endswith(".txt")]

if policy_files:
    for filename in policy_files:
        filepath = os.path.join(policies_folder, filename)
        print(f"  Loading: {filename}")

        if filename.endswith(".pdf"):
            text = load_pdf(filepath)
        else:
            text = load_text(filepath)

        chunks = chunk_text(text, chunk_size=150, overlap=30)
        print(f"  Split into {len(chunks)} chunks")

        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_counter}")
            all_metadata.append({"source": filename})
            chunk_counter += 1
else:
    print("\nERROR: No policy documents found in the policies folder.")
    print("Please add DataCompany PDF policy documents to:")
    print(f"  {os.path.abspath(policies_folder)}")
    print("\nExpected files:")
    expected = [
        "DC-POL-001_Rent_and_Payments.pdf",
        "DC-POL-002_Repairs_and_Maintenance.pdf",
        "DC-POL-003_Tenancy_Agreement.pdf",
        "DC-POL-004_Security_Deposit.pdf",
        "DC-POL-005_Anti_Social_Behaviour.pdf",
        "DC-POL-006_Property_Inspections.pdf",
        "DC-POL-007_Pets_and_Alterations.pdf",
        "DC-POL-008_Subletting_and_Occupancy.pdf",
        "DC-POL-009_Complaints_and_Disputes.pdf",
        "DC-POL-010_Eviction_and_Legal_Action.pdf",
    ]
    for f in expected:
        print(f"  - {f}")
    print("\nSystem cannot start without policy documents. Exiting.")
    exit()

# ============================================================
# STEP 4 - Embed and store in vector database
# ============================================================

print(f"\nStep 4: Embedding {chunk_counter} chunks into vector database...")
embeddings = embedder.encode(all_chunks).tolist()

collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    ids=all_ids,
    metadatas=all_metadata
)
print("Vector database ready.")


def contextualize_question(question, conversation_history):
    """
    If the question is a vague follow-up, rewrites it as a
    standalone question using conversation context.
    Costs one small LLM call but greatly improves retrieval.
    """
    # Only bother if there is conversation history
    if not conversation_history:
        return question

    # Check if question is too short or vague to search directly
    vague_indicators = [
        "what about", "and that", "how about", "what if",
        "same for", "does that", "is that", "tell me more"
    ]
    is_vague = any(indicator in question.lower() for indicator in vague_indicators)
    if not is_vague and len(question.split()) > 6:
        return question  # question is specific enough, no rewrite needed

    # Build last 2 exchanges as context
    recent = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
    history_text = "\n".join([
        f"{'Tenant' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    ])

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=60,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite vague follow-up questions into clear standalone questions "
                        "about housing policy. Return ONLY the rewritten question. Nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{history_text}\n\n"
                        f"Follow-up question: {question}\n\n"
                        f"Rewrite as a standalone housing policy question:"
                    )
                }
            ]
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"  Query rewritten: '{question}' -> '{rewritten}'")
        return rewritten

    except Exception:
        return question  # fallback to original if rewrite fails

# ============================================================
# STEP 7 - RAG engine WITH conversation memory
# ============================================================

def ask_datacompany_with_memory(question, conversation_history, top_k=5):
    """
    Full RAG pipeline with conversation memory and confidence scoring.
    Used by interactive mode and the Flask web interface.
    """
    # Rewrite vague follow-up questions before searching ChromaDB
    search_question = contextualize_question(question, conversation_history)
    # Use query expansion to improve retrieval
    query_versions = expand_query(search_question)
    all_retrieved_chunks = []
    all_retrieved_metadatas = []
    all_retrieved_distances = []
    seen_ids = set()

    for query in query_versions:
        question_embedding = embedder.encode([query]).tolist()
        results = collection.query(
            query_embeddings=question_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            if doc not in seen_ids:
                seen_ids.add(doc)
                all_retrieved_chunks.append(doc)
                all_retrieved_metadatas.append(meta)
                all_retrieved_distances.append(dist)

    # Sort by distance and keep top results
    combined = sorted(
        zip(all_retrieved_distances, all_retrieved_chunks, all_retrieved_metadatas),
        key=lambda x: x[0]
    )[:top_k]

    best_distance = combined[0][0]
    final_chunks  = [c[1] for c in combined]
    confidence    = get_confidence(best_distance)

    # Pick the most frequently appearing source across all chunks
    # This is more accurate than just taking the top ranked chunk
    all_sources = [c[2]["source"] for c in combined]
    best_source = max(set(all_sources), key=all_sources.count)

    context = "\n\n".join(
        [f"Policy section {i+1}:\n{chunk}"
         for i, chunk in enumerate(final_chunks)]
    )

    # Build messages with full conversation history
    messages = [
        {
            "role": "system",
            "content": (
                "You are DataCompany's housing policy assistant.\n"
                "Answer questions using ONLY the policy sections provided below.\n"
                "Be clear, professional and concise.\n"
                "You have memory of the conversation — use it for follow-up questions.\n"
                "If the answer is not found in the context say exactly:\n"
                "'This information is not covered in the current DataCompany policy documents.'\n"
                "NEVER use knowledge from outside the provided policy sections.\n"
                "NEVER invent procedures, timelines or steps not explicitly stated.\n\n"
                f"Current policy context:\n{context}"
            )
        }
    ]

    messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=300,
            messages=messages
        )
        answer = response.choices[0].message.content

        # Update conversation history
        conversation_history.append({"role": "user",      "content": question})
        conversation_history.append({"role": "assistant", "content": answer})

        # Keep last 6 exchanges to avoid token overflow
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]

        return {
            "question":         question,
            "answer":           answer,
            "best_source":      best_source,
            "chunks_used":      len(final_chunks),
            "history_length":   len(conversation_history) // 2,
            "confidence_label": confidence,
            "best_distance":    round(best_distance, 3)
        }, conversation_history

    except Exception as e:
        return {
            "question":         question,
            "answer":           f"System error: {e}",
            "best_source":      "Unknown",
            "chunks_used":      0,
            "history_length":   0,
            "confidence_label": "UNKNOWN",
            "best_distance":    0
        }, conversation_history


def print_answer_with_memory(result):
    """Prints a clean formatted answer with confidence and memory info."""
    NOT_FOUND = "this information is not covered"

    print(f"\n{'='*55}")
    print(f"QUESTION: {result['question']}")
    print(f"{'='*55}")
    print(f"ANSWER:\n{result['answer']}")

    if NOT_FOUND not in result['answer'].lower():
        print(f"\nSource:     {result['best_source']}")
        print(f"Confidence: {result['confidence_label']} (distance: {result['best_distance']})")
        print(f"Chunks used: {result['chunks_used']}")
    else:
        print("\nSource: None — question is outside DataCompany policy scope.")

    print(f"Conversation turns remembered: {result['history_length']}")


