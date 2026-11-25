from difflib import SequenceMatcher
import json
from pathlib import Path

try:
    from rapidfuzz import fuzz as _rapidfuzz
except ImportError:
    _rapidfuzz = None


def _sim_ratio(a, b):
    if _rapidfuzz is not None:
        return _rapidfuzz.token_sort_ratio(a, b)
    return int(round(SequenceMatcher(None, a, b).ratio() * 100))


def _partial_ratio(a, b):
    if _rapidfuzz is not None:
        return _rapidfuzz.partial_ratio(a, b)
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0
    return int(round(100 * len(a_tokens & b_tokens) / min(len(a_tokens), len(b_tokens))))

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "iset_faq_enhanced.json"

greeting_phrases = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "what can you do", "who are you", "help", "start", "can you help me",
    "greetings", "bonjour", "salut", "coucou", "bonsoir", "welcome",
]

iset_keywords = [
    "iset", "iset sfax", "admission", "department", "departments", "program",
    "courses", "timetable", "exam", "internship", "pfe", "moodle", "registration",
    "scholarship", "library", "contact", "administration",
]

_embedder = None
_embedder_failed = False


def _load_pages():
    if DATA_PATH.exists():
        with open(DATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def prepare_blocks(data):
    blocks = []
    for entry in data:
        text = entry.get("content", "")
        chunks = [c.strip() for c in text.split("\n") if len(c.strip().split()) > 4]
        blocks.extend(chunks)
    return blocks


def prepare_qa_data(data):
    qa_blocks = []
    for entry in data:
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()
        if question and answer:
            qa_blocks.append({"question": question, "answer": answer})
    return qa_blocks


iset_pages = _load_pages()
blocks = prepare_blocks(iset_pages)
qa_data = prepare_qa_data(iset_pages)


def _get_embedder():
    global _embedder, _embedder_failed
    if _embedder is not None or _embedder_failed:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _embedder_failed = True
        _embedder = None
    return _embedder


def is_greeting(message, threshold=70):
    clean = (message or "").strip().lower()
    if not clean:
        return False
    best = max(_sim_ratio(clean, g) for g in greeting_phrases)
    return best >= threshold


def search_local_knowledge(question, top_k=3):
    if not qa_data:
        return "Sorry, I could not find matching details in the local ISET Sfax knowledge base."

    clean = (question or "").strip().lower()
    scored = []
    for item in qa_data:
        score = max(
            _partial_ratio(clean, item["question"].lower()),
            _partial_ratio(clean, item["answer"].lower()),
        )
        scored.append((score, item["answer"]))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_answers = [ans for score, ans in scored[:top_k] if score >= 35]
    return "\n\n".join(best_answers) if best_answers else qa_data[0]["answer"]


def is_iset_related(question):
    clean = (question or "").lower()
    return any(kw in clean for kw in iset_keywords)
