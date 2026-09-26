from groq import Groq
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import os
import re
import time
import threading
import pickle

load_dotenv()
client = Groq()

print("DataCompany RAG Policy Chatbot")
print("="*55)

# ============================================================
# STEP 1 - Document loading functions
# ============================================================

def load_pdf(filepath):
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
                    row_text = " | ".join(cell.strip() for cell in row if cell)
                    if row_text:
                        text += row_text + "\n"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'(\w)\n(\w)', r'\1\2', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text


def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, chunk_size=150, overlap=30):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ============================================================
# STEP 2 & 3 - Load all policy documents and build BM25 index
#
# This used to run as top-level module code, which meant it executed
# the instant this file was imported — blocking Flask/gunicorn from
# starting and binding a port until all PDFs were loaded and indexed.
# It's now wrapped in a function that web_app.py runs in a background
# thread, so the web server can bind its port immediately while
# indexing happens in parallel. index_ready signals when it's safe
# to actually run searches.
# ============================================================

policies_folder = "policies"
all_chunks    = []
all_metadata  = []
chunk_counter = 0
bm25          = None
index_ready   = threading.Event()


INDEX_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_cache.pkl")


def build_index():
    """
    Loads all policy documents and builds the BM25 index. Safe to run in a
    background thread. If a pre-built cache file exists, loads it instantly
    instead of re-parsing every PDF and rebuilding BM25 from scratch — this
    is what makes startup take milliseconds instead of minutes.
    """
    global all_chunks, all_metadata, chunk_counter, bm25

    # Fast path: load a pre-built cache if one exists
    if os.path.exists(INDEX_CACHE_PATH):
        print("\nLoading cached policy index...")
        try:
            with open(INDEX_CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            all_chunks    = cache["all_chunks"]
            all_metadata  = cache["all_metadata"]
            chunk_counter = len(all_chunks)
            bm25          = cache["bm25"]
            print(f"Loaded cached index: {chunk_counter} chunks. Ready instantly.")
            index_ready.set()
            return
        except Exception as e:
            print(f"Could not load index cache ({e}). Rebuilding from PDFs instead.")

    # Slow path: no cache found, build from scratch (same as before)
    print("\nStep 2: Loading DataCompany policy documents...")

    policy_files = []
    if os.path.exists(policies_folder):
        policy_files = [f for f in os.listdir(policies_folder)
                         if f.endswith(".pdf") or f.endswith(".txt")]

    if not policy_files:
        print("\nERROR: No policy documents found in the policies folder.")
        print(f"Expected location: {os.path.abspath(policies_folder)}")
        return

    for filename in sorted(policy_files):
        filepath = os.path.join(policies_folder, filename)
        print(f"  Loading: {filename}")
        text   = load_pdf(filepath) if filename.endswith(".pdf") else load_text(filepath)
        chunks = chunk_text(text)
        print(f"  Split into {len(chunks)} chunks")
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({"source": filename})
            chunk_counter += 1

    print(f"\nStep 3: Building BM25 search index over {chunk_counter} chunks...")
    tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    print("BM25 index ready.")

    # Save a cache so future startups (including redeploys) skip straight
    # to the fast path above instead of re-parsing every PDF again.
    try:
        with open(INDEX_CACHE_PATH, "wb") as f:
            pickle.dump({"all_chunks": all_chunks, "all_metadata": all_metadata, "bm25": bm25}, f)
        print(f"Saved index cache to {INDEX_CACHE_PATH}")
    except Exception as e:
        print(f"Could not save index cache ({e}). Will rebuild from PDFs next time.")

    index_ready.set()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_confidence(score):
    """Converts BM25 score to confidence label."""
    if score >= 10:
        return "DIRECT MATCH"
    elif score >= 6:
        return "STRONG MATCH"
    elif score >= 3:
        return "GOOD MATCH"
    elif score >= 1:
        return "PARTIAL MATCH"
    else:
        return "EXPANDED MATCH"


def expand_query(question):
    """Generates alternative phrasings to improve retrieval."""
    expansions = [question]
    replacements = {
        "stages of action":    "sanctions warnings stage 1 stage 2 stage 3 stage 4",
        "stages":              "sanctions warnings stage 1 stage 2 stage 3",
        "fix":                 "repair resolve maintenance",
        "miss rent":           "fail pay rent arrears overdue",
        "leave":               "vacate end tenancy notice",
        "kick out":            "eviction proceedings notice quit",
        "deposit":             "security deposit return deductions",
        "complaints":          "formal warnings issued complaint",
        "enter":               "access inspection rights notice",
        "asb":                 "anti-social behaviour sanctions stage warning",
        "anti-social":         "stage 1 warning stage 2 formal hearing eviction",
        "noise":               "anti-social behaviour formal warning stage sanctions",
        "end of tenancy":      "notice vacate termination procedure end tenancy",
        "what about end":      "end tenancy notice period termination",
        "pets":                "pet permission consent allowed",
        "sublet":              "subletting prohibited consent written",
    }
    expanded = question.lower()
    for term, replacement in replacements.items():
        if term in expanded:
            new_q = expanded.replace(term, replacement)
            if new_q not in expansions:
                expansions.append(new_q)
    return expansions[:3]


def bm25_search(query, top_k=5):
    """Search using BM25 and return top chunks with scores."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(all_chunks[i], all_metadata[i], float(scores[i])) for i in top_indices]


def search_with_expansion(question, top_k=5):
    """Search with query expansion, merge and deduplicate results."""
    queries      = expand_query(question)
    seen         = set()
    merged       = []

    for q in queries:
        results = bm25_search(q, top_k=top_k)
        for chunk, meta, score in results:
            if chunk not in seen:
                seen.add(chunk)
                merged.append((chunk, meta, score))

    merged.sort(key=lambda x: x[2], reverse=True)
    return merged[:top_k]


# ============================================================
# CONTEXTUALIZE VAGUE FOLLOW-UP QUESTIONS
# ============================================================

def contextualize_question(question, conversation_history):
    """Rewrites vague follow-up questions using conversation context."""
    if not conversation_history:
        return question

    vague_indicators = [
        "what about", "and that", "how about", "what if",
        "same for", "does that", "is that", "tell me more"
    ]
    is_vague = (
        any(ind in question.lower() for ind in vague_indicators)
        or len(question.split()) <= 5
    )
    if not is_vague:
        return question

    recent = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
    history_text = "\n".join([
        f"{'Tenant' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    ])

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0,
            max_tokens=60,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite vague follow-up questions into clear standalone "
                        "housing policy questions. Return ONLY the rewritten question."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation:\n{history_text}\n\n"
                        f"Follow-up: {question}\n\n"
                        f"Rewrite as standalone question:"
                    )
                }
            ]
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"  Query rewritten: '{question}' -> '{rewritten}'")
        return rewritten
    except Exception:
        return question


# ============================================================
# MAIN RAG FUNCTION WITH MEMORY
# ============================================================

def ask_datacompany_with_memory(question, conversation_history, top_k=5, verbose=False):
    """
    Full RAG pipeline with BM25 retrieval, query expansion,
    conversation memory and confidence scoring.
    """
    if not index_ready.is_set():
        return {
            "question":         question,
            "answer":           "Still loading policy documents — please try again in a few seconds.",
            "best_source":      "",
            "chunks_used":      0,
            "history_length":   0,
            "confidence_label": "LOADING",
            "best_score":       0
        }, conversation_history

    # Rewrite vague follow-up questions
    search_question = contextualize_question(question, conversation_history)

    # Search with expansion
    results    = search_with_expansion(search_question, top_k=top_k)
    best_score = results[0][2] if results else 0
    confidence = get_confidence(best_score)

    # Pick most frequent source across top results
    all_sources = [r[1]["source"] for r in results]
    best_source = max(set(all_sources), key=all_sources.count)

    final_chunks = [r[0] for r in results]

    if verbose:
        print(f"\n--- Retrieved chunks (BM25 score: {best_score:.2f}) ---")
        for chunk, meta, score in results[:3]:
            print(f"  [{score:.2f}] {meta['source']} | {chunk[:100]}...")
        print("---")

    context = "\n\n".join(
        [f"Policy section {i+1}:\n{chunk}"
         for i, chunk in enumerate(final_chunks)]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are DataCompany's housing policy assistant.\n\n"
                "If the tenant's message is a greeting or general small talk "
                "(e.g. 'hello', 'how are you', 'thanks', 'what's your name'), "
                "respond warmly and briefly in 1-2 sentences, and invite them "
                "to ask a question about their tenancy. Do NOT use the policy "
                "context or the 'not covered' fallback for these messages.\n\n"
                "For anything else, treat it as a policy question and follow "
                "these rules strictly:\n"
                "- Answer using ONLY the policy sections provided below.\n"
                "- Be clear, professional and concise.\n"
                "- You have memory of the conversation — use it for follow-up questions.\n"
                "- If the answer is not found in the context say exactly:\n"
                "  'This information is not covered in the current DataCompany policy documents.'\n"
                "- NEVER use knowledge from outside the provided policy sections.\n"
                "- NEVER invent procedures, timelines or steps not explicitly stated.\n\n"
                f"Current policy context:\n{context}"
            )
        }
    ]

    messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0,
            max_tokens=300,
            messages=messages
        )
        answer = response.choices[0].message.content

        conversation_history.append({"role": "user",      "content": question})
        conversation_history.append({"role": "assistant", "content": answer})

        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]

        return {
            "question":         question,
            "answer":           answer,
            "best_source":      best_source,
            "chunks_used":      len(final_chunks),
            "history_length":   len(conversation_history) // 2,
            "confidence_label": confidence,
            "best_score":       round(best_score, 2)
        }, conversation_history

    except Exception as e:
        return {
            "question":         question,
            "answer":           f"System error: {e}",
            "best_source":      "Unknown",
            "chunks_used":      0,
            "history_length":   0,
            "confidence_label": "UNKNOWN",
            "best_score":       0
        }, conversation_history

