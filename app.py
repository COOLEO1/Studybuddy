import os
import io
import json
import base64
import secrets
import requests
import psycopg2
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, session, redirect
from werkzeug.security import generate_password_hash, check_password_hash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pypdf import PdfReader
import docx
from bs4 import BeautifulSoup

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB upload cap
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_ENV") != "development",
)


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# --- Rate limiting so a public deploy can't blow through your API budget ---
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per hour", "10 per minute"],
    storage_uri="memory://",
)

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-large-latest"

# NOTE: Voxtral (voice) endpoint paths below are based on Mistral's published
# audio API structure as of writing. Mistral's audio API is newer than the
# chat API and could change — if these calls start failing, check
# https://docs.mistral.ai/studio/audio/overview for the current paths/params.
AUDIO_TRANSCRIBE_URL = "https://api.mistral.ai/v1/audio/transcriptions"
AUDIO_SPEECH_URL = "https://api.mistral.ai/v1/audio/speech"
TRANSCRIBE_MODEL = "voxtral-mini-latest"
TTS_MODEL = "voxtral-mini-tts-2603"

# Per-call chunk size (safe single-request size) and total ceiling across chunks.
CHUNK_SIZE = 12000
MAX_TOTAL_CHARS = 60000  # ~5 chunks worth — enough for a full play/long chapter
MAX_CHUNKS = 5


def split_into_chunks(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks near chunk_size, breaking on paragraph boundaries
    where possible so we don't cut a sentence in half mid-thought."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining)
            break
        window = remaining[:chunk_size]
        split_at = window.rfind("\n\n")
        if split_at < chunk_size * 0.5:
            split_at = window.rfind(". ")
        if split_at < chunk_size * 0.5:
            split_at = chunk_size
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return chunks

# ---------------------------------------------------------------------------
# Shared deck storage (SQLite)
#
# Note on Render's free tier: disk storage is ephemeral and can be wiped on
# redeploys or when the service is rebuilt. Shared links will keep working
# as long as the service instance stays up, but could reset after a redeploy.
# For guaranteed-permanent links later, swap this for a hosted Postgres DB.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Database — Supabase Postgres (persistent, survives redeploys). Use the
# "Connection pooling" (transaction pooler) connection string from your
# Supabase project: Settings -> Database -> Connection string -> URI,
# set as the DATABASE_URL environment variable.
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def db_query(sql, params=None, fetchone=False, fetchall=False, commit=False):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        result = None
        if fetchone:
            result = cur.fetchone()
        elif fetchall:
            result = cur.fetchall()
        if commit:
            conn.commit()
        return result
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist yet. Called once at startup."""
    if not DATABASE_URL:
        return
    statements = [
        """CREATE TABLE IF NOT EXISTS shared_decks (
            id TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            title TEXT,
            data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )""",
        """CREATE TABLE IF NOT EXISTS past_papers (
            id TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            year TEXT,
            school TEXT,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )""",
        """CREATE TABLE IF NOT EXISTS pageviews (
            id SERIAL PRIMARY KEY,
            path TEXT NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )""",
        # Safe to re-run: adds the columns if this table already existed
        # from before IP/device tracking was added.
        "ALTER TABLE pageviews ADD COLUMN IF NOT EXISTS ip_address TEXT",
        "ALTER TABLE pageviews ADD COLUMN IF NOT EXISTS user_agent TEXT",
        """CREATE TABLE IF NOT EXISTS error_log (
            id SERIAL PRIMARY KEY,
            context TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )""",
        """CREATE TABLE IF NOT EXISTS generation_events (
            id SERIAL PRIMARY KEY,
            mode TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )""",
        """CREATE TABLE IF NOT EXISTS tutor_events (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP DEFAULT NOW()
        )""",
        """CREATE TABLE IF NOT EXISTS site_content (
            page TEXT PRIMARY KEY,
            content TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMP DEFAULT NOW()
        )""",
    ]
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        for stmt in statements:
            cur.execute(stmt)
        conn.commit()
    finally:
        conn.close()


init_db()


def parse_device_summary(user_agent):
    """Lightweight device/browser summary from a User-Agent string, without
    pulling in an extra dependency. Good enough for admin-panel readability,
    not meant to be a precise device-detection library."""
    if not user_agent:
        return "Unknown device"
    ua = user_agent.lower()

    if "iphone" in ua:
        os_name = "iPhone"
    elif "ipad" in ua:
        os_name = "iPad"
    elif "android" in ua:
        os_name = "Android"
    elif "windows" in ua:
        os_name = "Windows"
    elif "mac os" in ua or "macintosh" in ua:
        os_name = "Mac"
    elif "linux" in ua:
        os_name = "Linux"
    else:
        os_name = "Unknown OS"

    if "edg/" in ua:
        browser = "Edge"
    elif "chrome/" in ua and "chromium" not in ua:
        browser = "Chrome"
    elif "firefox/" in ua:
        browser = "Firefox"
    elif "safari/" in ua and "chrome/" not in ua:
        browser = "Safari"
    elif "opera" in ua or "opr/" in ua:
        browser = "Opera"
    else:
        browser = "Unknown browser"

    return f"{os_name} · {browser}"


def get_client_ip():
    # Render (and most hosts behind a proxy) put the real client IP first in
    # X-Forwarded-For; fall back to remote_addr for local/direct requests.
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def log_pageview(path):
    if not DATABASE_URL:
        return
    try:
        db_query(
            "INSERT INTO pageviews (path, ip_address, user_agent) VALUES (%s, %s, %s)",
            (path, get_client_ip(), request.headers.get("User-Agent", "")[:300]),
            commit=True,
        )
    except Exception:
        pass  # analytics must never break the actual page


def log_error(context, message):
    if not DATABASE_URL:
        return
    try:
        db_query(
            "INSERT INTO error_log (context, message) VALUES (%s, %s)",
            (context, str(message)[:500]), commit=True,
        )
    except Exception:
        pass


def log_generation_event(mode):
    if not DATABASE_URL:
        return
    try:
        db_query("INSERT INTO generation_events (mode) VALUES (%s)", (mode,), commit=True)
    except Exception:
        pass


def log_tutor_event():
    if not DATABASE_URL:
        return
    try:
        db_query("INSERT INTO tutor_events DEFAULT VALUES", commit=True)
    except Exception:
        pass


@app.before_request
def track_pageview():
    if request.method == "GET" and not request.path.startswith(("/api/", "/static/", "/admin")):
        log_pageview(request.path)


def generate_share_id():
    return secrets.token_urlsafe(6).replace("-", "").replace("_", "")[:8]


# ---------------------------------------------------------------------------
# Text extraction helpers (file upload, URL fetch, photo OCR)
# ---------------------------------------------------------------------------

def extract_from_pdf(file_stream):
    reader = PdfReader(file_stream)
    pages = []
    for page in reader.pages[:40]:  # cap pages to keep it fast/cheap
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def extract_from_docx(file_stream):
    document = docx.Document(file_stream)
    return "\n".join(p.text for p in document.paragraphs).strip()


def extract_from_image(file_bytes, mime_type):
    """Use Mistral's vision model to read text AND describe diagrams/figures
    in a photo — important for subjects like Maths where the diagram often
    carries the actual content (angles, shapes, graphs), not just the words."""
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "pixtral-12b-2409",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This image may contain text, and/or diagrams, figures, graphs, "
                            "or geometric shapes (common in Maths and Science material). Do both:\n"
                            "1. Transcribe all readable text exactly as written.\n"
                            "2. For any diagram, figure, graph, or shape, describe it precisely in "
                            "words immediately after the related text — include labeled points, "
                            "angles, measurements, axis values, shape type, and how parts relate "
                            "(e.g. 'Triangle ABC with angle B = 40°, angle C = 90°, side AB = 5cm'). "
                            "If it's a graph, describe the curve/line shape and key points.\n"
                            "Output only the transcription and descriptions, no extra commentary, no markdown."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{b64}",
                    },
                ],
            }
        ],
        "temperature": 0,
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def extract_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0 (RecallStudyBuddy/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if len(p) > 40)
    if not text:
        text = soup.get_text(" ", strip=True)
    return text.strip()


def build_prompt(mode, text, count, difficulty):
    if mode == "flashcards":
        return (
            f"You are an expert tutor. Read the study material below and generate "
            f"{count} high-quality flashcards at {difficulty} difficulty. "
            f"Return ONLY valid JSON, no markdown fences, no commentary, in this exact shape:\n"
            f'{{"cards": [{{"front": "question or term", "back": "answer or definition"}}]}}\n\n'
            f"STUDY MATERIAL:\n{text}"
        )
    elif mode == "exam":
        return (
            f"You are an expert exam-question writer. Read the study material below and generate "
            f"{count} exam questions at {difficulty} difficulty, testing real understanding — not "
            f"just recall. Mix THREE question types across the set:\n"
            f"- 'multiple_choice': exactly 4 options, one correct_index, plus a brief explanation.\n"
            f"- 'short_answer': a question expecting a short factual/definitional typed answer. "
            f"Include 2-4 key_points the answer should cover.\n"
            f"- 'explain': a question asking the student to explain, describe, or discuss something "
            f"in their own words (like 'Describe the process of...' or 'Explain why...'). Include "
            f"3-5 key_points a strong answer would cover.\n"
            f"Aim for roughly a third of each type, adjusted to fit the material. "
            f"Return ONLY valid JSON, no markdown fences, no commentary, in this exact shape:\n"
            f'{{"questions": [\n'
            f'  {{"type": "multiple_choice", "question": "...", "options": ["A","B","C","D"], "correct_index": 0, "explanation": "..."}},\n'
            f'  {{"type": "short_answer", "question": "...", "key_points": ["...", "..."]}},\n'
            f'  {{"type": "explain", "question": "...", "key_points": ["...", "...", "..."]}}\n'
            f"]}}\n\n"
            f"STUDY MATERIAL:\n{text}"
        )
    elif mode == "pastpaper":
        return (
            f"You are an expert exam-question writer, specifically experienced with MSCE "
            f"(Malawi School Certificate of Education) style exams. Below is a real past exam "
            f"paper or excerpt from one. Study its SUBJECT, question STYLE, phrasing conventions, "
            f"topic coverage, and difficulty level carefully.\n\n"
            f"Generate {count} NEW multiple-choice practice questions that closely match this "
            f"paper's style, subject, and difficulty at a {difficulty} level — testing the SAME "
            f"topics and skills the original paper tests, but with different specific questions "
            f"than what's literally written in the source, so the student gets fresh practice "
            f"rather than just seeing the same questions again. Match the exam board's typical "
            f"phrasing and question structure as closely as possible.\n"
            f"Each question must have exactly 4 options with exactly one correct answer, plus a "
            f"brief explanation of why that answer is correct.\n"
            f"Return ONLY valid JSON, no markdown fences, no commentary, in this exact shape:\n"
            f'{{"questions": [{{"question": "...", "options": ["A","B","C","D"], '
            f'"correct_index": 0, "explanation": "brief reason"}}]}}\n\n'
            f"PAST EXAM PAPER:\n{text}"
        )
    else:  # quiz
        return (
            f"You are an expert tutor. Read the study material below and generate "
            f"{count} multiple-choice quiz questions at {difficulty} difficulty. "
            f"Each question must have exactly 4 options with exactly one correct answer. "
            f"Return ONLY valid JSON, no markdown fences, no commentary, in this exact shape:\n"
            f'{{"questions": [{{"question": "...", "options": ["A","B","C","D"], '
            f'"correct_index": 0, "explanation": "brief reason"}}]}}\n\n'
            f"STUDY MATERIAL:\n{text}"
        )


def call_mistral(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You always respond with strictly valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.5,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend_dist")


@app.route("/assets/<path:filename>")
@limiter.exempt
def frontend_assets(filename):
    from flask import send_from_directory
    return send_from_directory(os.path.join(FRONTEND_DIST, "assets"), filename)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
@limiter.exempt
def serve_spa(path):
    from flask import send_from_directory
    index_path = os.path.join(FRONTEND_DIST, "index.html")
    if not os.path.exists(index_path):
        return jsonify({"error": "Frontend build not found. Run the React build first."}), 500
    return send_from_directory(FRONTEND_DIST, "index.html")


@app.route("/api/share", methods=["POST"])
@limiter.limit("20 per minute")
def create_share():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    deck = data.get("deck")
    title = (data.get("title") or "").strip()[:120]

    if mode not in ("flashcards", "quiz", "pastpaper", "exam") or not deck or not isinstance(deck, list):
        return jsonify({"error": "Invalid deck data."}), 400
    if len(deck) > 25:
        return jsonify({"error": "Deck too large to share."}), 400

    share_id = generate_share_id()
    db_query(
        "INSERT INTO shared_decks (id, mode, title, data) VALUES (%s, %s, %s, %s)",
        (share_id, mode, title, json.dumps(deck)),
        commit=True,
    )

    return jsonify({"id": share_id})


@app.route("/api/share/<share_id>", methods=["GET"])
def get_share(share_id):
    row = db_query(
        "SELECT mode, title, data FROM shared_decks WHERE id = %s", (share_id,), fetchone=True
    )

    if not row:
        return jsonify({"error": "This shared deck wasn't found — the link may be old or invalid."}), 404

    mode, title, data = row
    return jsonify({"mode": mode, "title": title, "deck": json.loads(data)})


# ---------------------------------------------------------------------------
# Past papers library — a shared, growing collection of REAL past papers
# that people upload. This sidesteps auto-fetching papers from the internet
# (no reliable source for MSCE specifically, plus real copyright risk) —
# instead the library grows from what people who already have real papers
# choose to contribute, tagged by subject/year/school so others can browse
# and reuse them for Past Paper mode instead of re-uploading their own copy.
# ---------------------------------------------------------------------------

MAX_PAPER_CHARS = 40000


@app.route("/api/papers", methods=["POST"])
@limiter.limit("10 per minute")
def add_paper():
    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "").strip()[:60]
    year = (data.get("year") or "").strip()[:20]
    school = (data.get("school") or "").strip()[:120]
    title = (data.get("title") or "").strip()[:150]
    text = (data.get("text") or "").strip()

    if not subject or not text:
        return jsonify({"error": "Subject and paper text are required."}), 400
    if len(text) > MAX_PAPER_CHARS:
        return jsonify({"error": f"Paper is too long (max {MAX_PAPER_CHARS:,} characters)."}), 400
    if not title:
        title = f"{subject} {year}".strip()

    paper_id = generate_share_id()
    db_query(
        "INSERT INTO past_papers (id, subject, year, school, title, text) VALUES (%s, %s, %s, %s, %s, %s)",
        (paper_id, subject, year, school, title, text),
        commit=True,
    )

    return jsonify({"id": paper_id})


@app.route("/api/papers", methods=["GET"])
def list_papers():
    subject_filter = (request.args.get("subject") or "").strip()
    if subject_filter:
        rows = db_query(
            "SELECT id, subject, year, school, title, created_at FROM past_papers "
            "WHERE subject ILIKE %s ORDER BY created_at DESC LIMIT 100",
            (f"%{subject_filter}%",), fetchall=True,
        )
    else:
        rows = db_query(
            "SELECT id, subject, year, school, title, created_at FROM past_papers "
            "ORDER BY created_at DESC LIMIT 100", fetchall=True,
        )

    papers = [
        {"id": r[0], "subject": r[1], "year": r[2], "school": r[3], "title": r[4], "created_at": str(r[5])}
        for r in (rows or [])
    ]
    return jsonify({"papers": papers})


@app.route("/api/papers/<paper_id>", methods=["GET"])
def get_paper(paper_id):
    row = db_query(
        "SELECT subject, year, school, title, text FROM past_papers WHERE id = %s", (paper_id,), fetchone=True
    )

    if not row:
        return jsonify({"error": "Paper not found."}), 404

    subject, year, school, title, text = row
    return jsonify({"subject": subject, "year": year, "school": school, "title": title, "text": text})


@app.route("/api/extract-file", methods=["POST"])
@limiter.limit("15 per minute")
def extract_file():
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400
    f = request.files["file"]
    filename = (f.filename or "").lower()

    try:
        if filename.endswith(".pdf"):
            text = extract_from_pdf(io.BytesIO(f.read()))
        elif filename.endswith(".docx"):
            text = extract_from_docx(io.BytesIO(f.read()))
        elif filename.endswith(".txt") or filename.endswith(".md"):
            text = f.read().decode("utf-8", errors="ignore")
        elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
            file_bytes = f.read()
            mime = f.mimetype or "image/jpeg"
            text = extract_from_image(file_bytes, mime)
        else:
            return jsonify({"error": "Unsupported file type. Use PDF, DOCX, TXT, or an image."}), 400
    except Exception as e:
        return jsonify({"error": f"Couldn't read that file: {e}"}), 400

    if not text.strip():
        return jsonify({"error": "Couldn't find any readable text in that file."}), 400

    return jsonify({"text": text[:MAX_TOTAL_CHARS]})


@app.route("/api/extract-url", methods=["POST"])
@limiter.limit("15 per minute")
def extract_url():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Please provide a valid URL starting with http:// or https://"}), 400

    try:
        text = extract_from_url(url)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Couldn't fetch that page: {e}"}), 400

    if not text.strip():
        return jsonify({"error": "Couldn't extract readable article text from that page."}), 400

    return jsonify({"text": text[:MAX_TOTAL_CHARS]})


TUTOR_SYSTEM_PROMPT = (
    "You are a patient, encouraging tutor helping a student preparing for their MSCE "
    "(Malawi School Certificate of Education) exam. You can teach ANY subject the student "
    "asks about — Maths, Physics, Biology, Chemistry, English, Geography, History, or "
    "anything else — not just Maths. Teach the way a great human tutor would:\n"
    "- Explain concepts step by step, in plain language, building from what the student "
    "already seems to know.\n"
    "- After explaining a concept, give the student ONE practice question/exercise to try, "
    "then wait for their answer before continuing.\n"
    "- When they answer, check their WORKING or reasoning, not just the final answer — MSCE "
    "marks method and explanation, not just the final answer. Point out exactly where a "
    "mistake happened if there is one, and explain the fix. If they're correct, briefly "
    "confirm why before moving on.\n"
    "- For Maths/Science, check numerical working step by step. For subjects like English, "
    "Biology, or History, check that their explanation covers the key points a marker would "
    "look for.\n"
    "- If they seem to be struggling with the same idea repeatedly, slow down and re-explain "
    "it a different way rather than pushing forward.\n"
    "- If a diagram, photo of notebook work, or exercise was described to you, reason about "
    "it using the description given, and comment on whether the student's working is correct.\n"
    "- Keep responses focused and conversational, not a lecture.\n"
    "- MATH NOTATION: write any mathematical expressions, equations, or formulas using LaTeX "
    "notation — wrap inline math like variables or short expressions in single dollar signs, "
    "e.g. $x^2 + 5x + 6 = 0$, and wrap larger displayed equations in double dollar signs, e.g. "
    "$$x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$$. This will be rendered as properly typeset math, "
    "so always use this notation for anything mathematical rather than writing it in plain text "
    "or ASCII approximations.\n"
    "- OTHER FORMATTING: do not use markdown symbols like **, #, or bullet dashes for anything "
    "that isn't math — write plain sentences and paragraphs. If you need to list steps, write "
    "them as 'First, ... Next, ... Then, ...' in prose, not as a markdown list."
)


@app.route("/api/tutor", methods=["POST"])
@limiter.limit("20 per minute")
def tutor():
    if not MISTRAL_API_KEY:
        return jsonify({"error": "Server misconfigured: no API key set."}), 500

    data = request.get_json(silent=True) or {}
    history = data.get("history", [])
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Please type or say something first."}), 400
    if len(message) > 4000:
        return jsonify({"error": "That message is too long."}), 400
    if not isinstance(history, list) or len(history) > 400:
        return jsonify({"error": "Conversation too long — please start a new tutor session."}), 400

    messages = [{"role": "system", "content": TUTOR_SYSTEM_PROMPT}]
    for turn in history[-50:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": message})

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL, "messages": messages, "temperature": 0.6}

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        log_error("tutor_request", e)
        return jsonify({"error": f"Upstream API error: {e}"}), 502
    except (KeyError, IndexError) as e:
        log_error("tutor_parse", e)
        return jsonify({"error": "The tutor didn't respond properly. Try again."}), 502

    log_tutor_event()
    return jsonify({"reply": reply})


@app.route("/api/voice/transcribe", methods=["POST"])
@limiter.limit("20 per minute")
def voice_transcribe():
    if not MISTRAL_API_KEY:
        return jsonify({"error": "Server misconfigured: no API key set."}), 500
    if "audio" not in request.files:
        return jsonify({"error": "No audio was uploaded."}), 400

    audio_file = request.files["audio"]
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    files = {"file": (audio_file.filename or "audio.webm", audio_file.stream, audio_file.mimetype)}
    data = {"model": TRANSCRIBE_MODEL}

    try:
        resp = requests.post(AUDIO_TRANSCRIBE_URL, headers=headers, files=files, data=data, timeout=45)
        resp.raise_for_status()
        result = resp.json()
        text = result.get("text", "")
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Voice transcription failed: {e}"}), 502
    except (KeyError, ValueError):
        return jsonify({"error": "Couldn't understand the audio. Try again or type instead."}), 502

    return jsonify({"text": text})


@app.route("/api/voice/speak", methods=["POST"])
@limiter.limit("20 per minute")
def voice_speak():
    if not MISTRAL_API_KEY:
        return jsonify({"error": "Server misconfigured: no API key set."}), 500

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text to speak."}), 400
    text = text[:2000]  # keep TTS cost bounded per call

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": TTS_MODEL, "input": text}

    try:
        resp = requests.post(AUDIO_SPEECH_URL, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Voice generation failed: {e}"}), 502

    return resp.content, 200, {"Content-Type": "audio/mpeg"}


@app.route("/api/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate():
    if not MISTRAL_API_KEY:
        return jsonify({"error": "Server misconfigured: no API key set."}), 500

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    mode = data.get("mode", "flashcards")
    count = data.get("count", 8)
    difficulty = data.get("difficulty", "medium")

    if not text:
        return jsonify({"error": "Please paste some study material first."}), 400
    if len(text) > MAX_TOTAL_CHARS:
        return jsonify({"error": f"Text is too long (max {MAX_TOTAL_CHARS:,} characters)."}), 400
    if mode not in ("flashcards", "quiz", "pastpaper", "exam"):
        return jsonify({"error": "Invalid mode."}), 400
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 8
    if count < 3 or count > 25:
        return jsonify({"error": "Card count must be between 3 and 25."}), 400

    chunks = split_into_chunks(text)[:MAX_CHUNKS]
    key = "cards" if mode == "flashcards" else "questions"

    if len(chunks) == 1:
        prompt = build_prompt(mode, chunks[0], count, difficulty)
        try:
            result = call_mistral(prompt)
        except requests.exceptions.RequestException as e:
            log_error("generate_request", e)
            return jsonify({"error": f"Upstream API error: {e}"}), 502
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            log_error("generate_parse", e)
            return jsonify({"error": "The model returned something unexpected. Try again."}), 502
        log_generation_event(mode)
        return jsonify(result)

    # Long input: distribute the requested count across chunks, generate all
    # chunks IN PARALLEL (each chunk is a separate Mistral call — running them
    # concurrently keeps total wait time close to a single call instead of
    # stacking up sequentially, which risks the server/proxy timing out).
    per_chunk = max(2, count // len(chunks))

    def generate_chunk(chunk):
        prompt = build_prompt(mode, chunk, per_chunk, difficulty)
        try:
            result = call_mistral(prompt)
            return result.get(key, [])
        except requests.exceptions.RequestException:
            return []
        except (KeyError, IndexError, json.JSONDecodeError):
            return []

    merged = []
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [executor.submit(generate_chunk, c) for c in chunks]
        for future in as_completed(futures):
            merged.extend(future.result())

    if not merged:
        return jsonify({"error": "Couldn't generate a deck from that material. Try again."}), 502

    log_generation_event(mode)
    return jsonify({key: merged[:count if count >= len(chunks) else len(merged)]})


# ---------------------------------------------------------------------------
# Admin dashboard — session-based login (not a key in the URL), rate-limited
# login attempts. The dashboard itself is a React page (in the SPA build);
# these are just the JSON APIs it talks to.
# ---------------------------------------------------------------------------
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

# --- Supabase Storage config (for profile photo / content images) ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "studybuddy-content")

DEFAULT_CONTENT = {
    "about": (
        "# About Study Buddy\n\nBuilt by Leon Mapelera, from Zomba, Malawi.\n\n"
        "Study Buddy turns notes, articles, files, or photos into flashcards, quizzes, "
        "and exams — with an AI tutor built in."
    ),
    "privacy": "# Privacy Policy\n\nNo accounts, no sign-up. See the admin panel to edit this page.",
    "terms": "# Terms\n\nProvided as-is, no warranty. See the admin panel to edit this page.",
}


@app.route("/api/content/<page>", methods=["GET"])
@limiter.exempt
def get_content(page):
    if page not in ("about", "privacy", "terms"):
        return jsonify({"error": "Unknown page."}), 404
    try:
        row = db_query("SELECT content FROM site_content WHERE page = %s", (page,), fetchone=True)
        content = row[0] if row else DEFAULT_CONTENT.get(page, "")
    except Exception:
        content = DEFAULT_CONTENT.get(page, "")
    return jsonify({"content": content})


@app.route("/api/content/<page>", methods=["PUT"])
def update_content(page):
    if not require_admin():
        return jsonify({"error": "Not authorized."}), 403
    if page not in ("about", "privacy", "terms"):
        return jsonify({"error": "Unknown page."}), 404
    data = request.get_json(silent=True) or {}
    content = (data.get("content") or "")[:20000]
    db_query(
        "INSERT INTO site_content (page, content, updated_at) VALUES (%s, %s, NOW()) "
        "ON CONFLICT (page) DO UPDATE SET content = EXCLUDED.content, updated_at = NOW()",
        (page, content),
        commit=True,
    )
    return jsonify({"ok": True})


@app.route("/api/content/profile-photo", methods=["GET"])
@limiter.exempt
def get_profile_photo():
    try:
        row = db_query("SELECT content FROM site_content WHERE page = %s", ("profile-photo",), fetchone=True)
        return jsonify({"url": row[0] if row else ""})
    except Exception:
        return jsonify({"url": ""})


@app.route("/api/admin/upload-photo", methods=["POST"])
@limiter.limit("10 per minute")
def upload_photo():
    if not require_admin():
        return jsonify({"error": "Not authorized."}), 403
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return jsonify({"error": "Supabase Storage isn't configured (SUPABASE_URL / SUPABASE_SERVICE_KEY)."}), 500
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded."}), 400

    photo = request.files["photo"]
    ext = os.path.splitext(photo.filename or "")[1] or ".jpg"
    object_path = f"profile{ext}"
    file_bytes = photo.read()

    try:
        upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{object_path}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": photo.mimetype or "image/jpeg",
            "x-upsert": "true",
        }
        resp = requests.post(upload_url, headers=headers, data=file_bytes, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        log_error("photo_upload", e)
        return jsonify({"error": f"Upload failed: {e}"}), 502

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{object_path}"
    db_query(
        "INSERT INTO site_content (page, content, updated_at) VALUES ('profile-photo', %s, NOW()) "
        "ON CONFLICT (page) DO UPDATE SET content = EXCLUDED.content, updated_at = NOW()",
        (public_url,),
        commit=True,
    )
    return jsonify({"url": public_url})


@app.route("/api/exam/evaluate", methods=["POST"])
@limiter.limit("20 per minute")
def evaluate_exam_answer():
    if not MISTRAL_API_KEY:
        return jsonify({"error": "Server misconfigured: no API key set."}), 500

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()[:1000]
    question_type = data.get("question_type", "short_answer")
    user_answer = (data.get("user_answer") or "").strip()[:3000]
    key_points = data.get("key_points") or []

    if not question or not user_answer:
        return jsonify({"error": "Missing question or answer."}), 400

    key_points_text = "\n".join(f"- {p}" for p in key_points[:6])
    prompt = (
        f"You are a supportive tutor grading a student's typed answer. This is a "
        f"'{question_type}' style question, so judge it on understanding and coverage of "
        f"key points, not exact wording.\n\n"
        f"QUESTION: {question}\n\n"
        f"KEY POINTS A STRONG ANSWER SHOULD COVER:\n{key_points_text or '(use your own judgement)'}\n\n"
        f"STUDENT'S ANSWER: {user_answer}\n\n"
        f"Write brief, encouraging feedback (3-5 sentences): what they got right, what's "
        f"missing or could be improved, and a one-line correct/model answer at the end. "
        f"Do not use markdown symbols like ** or # — plain sentences only."
    )

    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        feedback = resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        log_error("exam_evaluate", e)
        return jsonify({"error": f"Upstream API error: {e}"}), 502
    except (KeyError, IndexError) as e:
        log_error("exam_evaluate_parse", e)
        return jsonify({"error": "Couldn't evaluate that answer. Try again."}), 502

    return jsonify({"feedback": feedback})



# Rough cost-per-request estimates for the admin cost ticker. These are
# deliberately approximate — good enough to spot a runaway spike, not an
# exact invoice. Adjust if Mistral's pricing changes.
EST_COST_PER_GENERATION = 0.01
EST_COST_PER_TUTOR_MSG = 0.004


@app.route("/admin/api/login", methods=["POST"])
@limiter.limit("10 per minute")
def admin_api_login():
    data = request.get_json(silent=True) or {}
    password = data.get("password", "")
    if ADMIN_PASSWORD and secrets.compare_digest(password, ADMIN_PASSWORD):
        session["is_admin"] = True
        return jsonify({"ok": True})
    return jsonify({"error": "Incorrect password."}), 401


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect("/")


def require_admin():
    return session.get("is_admin") is True


def gather_admin_stats():
    total_pageviews = db_query("SELECT COUNT(*) FROM pageviews", fetchone=True)[0]
    views_by_path = db_query(
        "SELECT path, COUNT(*) c FROM pageviews GROUP BY path ORDER BY c DESC LIMIT 12", fetchall=True
    )
    views_by_day = db_query(
        "SELECT date(created_at) d, COUNT(*) c FROM pageviews "
        "WHERE created_at > NOW() - INTERVAL '14 days' GROUP BY d ORDER BY d", fetchall=True
    )
    total_generations = db_query("SELECT COUNT(*) FROM generation_events", fetchone=True)[0]
    generations_by_mode = db_query(
        "SELECT mode, COUNT(*) c FROM generation_events GROUP BY mode ORDER BY c DESC", fetchall=True
    )
    total_tutor_msgs = db_query("SELECT COUNT(*) FROM tutor_events", fetchone=True)[0]
    total_shared_decks = db_query("SELECT COUNT(*) FROM shared_decks", fetchone=True)[0]
    total_papers = db_query("SELECT COUNT(*) FROM past_papers", fetchone=True)[0]
    recent_errors = db_query(
        "SELECT context, message, created_at FROM error_log ORDER BY id DESC LIMIT 25", fetchall=True
    )
    recent_visits_raw = db_query(
        "SELECT path, ip_address, user_agent, created_at FROM pageviews ORDER BY id DESC LIMIT 40",
        fetchall=True,
    )
    recent_visits = [
        {"path": p, "ip": ip or "unknown", "device": parse_device_summary(ua), "at": str(t)}
        for p, ip, ua, t in recent_visits_raw
    ]
    unique_ips_today = db_query(
        "SELECT COUNT(DISTINCT ip_address) FROM pageviews WHERE created_at > NOW() - INTERVAL '1 day'",
        fetchone=True,
    )[0]
    active_last_5min = db_query(
        "SELECT COUNT(DISTINCT ip_address) FROM pageviews WHERE created_at > NOW() - INTERVAL '5 minutes'",
        fetchone=True,
    )[0]
    requests_per_minute = db_query(
        "SELECT COUNT(*) FROM pageviews WHERE created_at > NOW() - INTERVAL '1 minute'",
        fetchone=True,
    )[0]
    errors_last_hour = db_query(
        "SELECT COUNT(*) FROM error_log WHERE created_at > NOW() - INTERVAL '1 hour'",
        fetchone=True,
    )[0]
    generations_today = db_query(
        "SELECT COUNT(*) FROM generation_events WHERE created_at > NOW() - INTERVAL '1 day'",
        fetchone=True,
    )[0]
    tutor_msgs_today = db_query(
        "SELECT COUNT(*) FROM tutor_events WHERE created_at > NOW() - INTERVAL '1 day'",
        fetchone=True,
    )[0]
    estimated_cost_today = (
        generations_today * EST_COST_PER_GENERATION + tutor_msgs_today * EST_COST_PER_TUTOR_MSG
    )

    # Device/browser breakdown — parsed on the fly from recent visits (no
    # extra DB column needed since parse_device_summary is cheap).
    device_raw = db_query(
        "SELECT user_agent FROM pageviews ORDER BY id DESC LIMIT 300", fetchall=True
    )
    device_counts = {}
    for (ua,) in device_raw:
        d = parse_device_summary(ua)
        device_counts[d] = device_counts.get(d, 0) + 1
    device_breakdown = sorted(
        [{"device": d, "count": c} for d, c in device_counts.items()],
        key=lambda x: -x["count"],
    )[:6]

    return {
        "total_pageviews": total_pageviews,
        "views_by_path": [{"path": p, "count": c} for p, c in views_by_path],
        "views_by_day": [{"day": str(d), "count": c} for d, c in views_by_day],
        "total_generations": total_generations,
        "generations_by_mode": [{"mode": m, "count": c} for m, c in generations_by_mode],
        "total_tutor_msgs": total_tutor_msgs,
        "total_shared_decks": total_shared_decks,
        "total_papers": total_papers,
        "recent_errors": [{"context": c, "message": m, "at": str(t)} for c, m, t in recent_errors],
        "recent_visits": recent_visits,
        "unique_ips_today": unique_ips_today,
        "active_last_5min": active_last_5min,
        "requests_per_minute": requests_per_minute,
        "errors_last_hour": errors_last_hour,
        "estimated_cost_today": estimated_cost_today,
        "device_breakdown": device_breakdown,
    }


@app.route("/admin/api/stats")
def admin_api_stats():
    if not require_admin():
        return jsonify({"error": "Not authorized."}), 403
    if not DATABASE_URL:
        return jsonify({"error": "No database configured."}), 500
    return jsonify(gather_admin_stats())


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Too many requests — please slow down and try again shortly."}), 429


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
