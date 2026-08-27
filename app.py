import os
import io
import json
import base64
import secrets
import requests
import psycopg2
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, render_template, session, redirect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pypdf import PdfReader
import docx
from bs4 import BeautifulSoup

app = Flask(__name__)
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
            id INTEGER PRIMARY KEY CHECK (id = 1),
            about_title TEXT NOT NULL DEFAULT 'StudyBuddy',
            developer_name TEXT NOT NULL DEFAULT 'Leon Mapelera',
            developer_role TEXT NOT NULL DEFAULT 'Creator & Developer 🇲🇼',
            tagline TEXT NOT NULL DEFAULT 'Building tools that make learning easier.',
            story TEXT NOT NULL DEFAULT 'I love building useful tools people can actually use. StudyBuddy turns notes, articles, files, or photos into practical study material so learning feels less like a chore and more like a tool you want to use.',
            quote TEXT NOT NULL DEFAULT 'Study smarter. Remember more.',
            contact_email TEXT NOT NULL DEFAULT 'leoc39063@gmail.com',
            privacy_text TEXT NOT NULL DEFAULT 'StudyBuddy does not require an account. Material is sent to the configured AI provider to generate study content. Avoid submitting sensitive information.',
            terms_text TEXT NOT NULL DEFAULT 'StudyBuddy is provided as-is. AI-generated study material can contain mistakes, so verify important information against trusted sources.',
            photo_data TEXT,
            photo_mime TEXT,
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


def ensure_site_content():
    if not DATABASE_URL:
        return
    try:
        db_query("""INSERT INTO site_content (id) VALUES (1) ON CONFLICT (id) DO NOTHING""", commit=True)
    except Exception:
        pass


ensure_site_content()


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


@app.route("/")
def index():
    return render_template("index.html", share_id=None)


@app.route("/deck/<share_id>")
def shared_deck(share_id):
    return render_template("index.html", share_id=share_id)


@app.route("/api/share", methods=["POST"])
@limiter.limit("20 per minute")
def create_share():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    deck = data.get("deck")
    title = (data.get("title") or "").strip()[:120]

    if mode not in ("flashcards", "quiz", "pastpaper") or not deck or not isinstance(deck, list):
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
    if mode not in ("flashcards", "quiz", "pastpaper"):
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
# login attempts, real charts instead of raw JSON.
# ---------------------------------------------------------------------------
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")


@app.route("/admin/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def admin_login():
    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        if ADMIN_PASSWORD and secrets.compare_digest(password, ADMIN_PASSWORD):
            session["is_admin"] = True
            return redirect("/admin")
        error = "Incorrect password."
    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect("/admin/login")


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
    generations_by_day = db_query(
        "SELECT date(created_at) d, COUNT(*) c FROM generation_events WHERE created_at > NOW() - INTERVAL '14 days' GROUP BY d ORDER BY d", fetchall=True
    )
    tutor_by_day = db_query(
        "SELECT date(created_at) d, COUNT(*) c FROM tutor_events WHERE created_at > NOW() - INTERVAL '14 days' GROUP BY d ORDER BY d", fetchall=True
    )
    device_rows = db_query(
        "SELECT user_agent, COUNT(*) c FROM pageviews GROUP BY user_agent ORDER BY c DESC LIMIT 100", fetchall=True
    )
    device_counts = {}
    for ua, c in device_rows:
        device = parse_device_summary(ua).split(" · ")[0]
        device_counts[device] = device_counts.get(device, 0) + c

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
        "generations_by_day": [{"day": str(d), "count": c} for d, c in generations_by_day],
        "tutor_by_day": [{"day": str(d), "count": c} for d, c in tutor_by_day],
        "device_breakdown": [{"device": k, "count": v} for k, v in sorted(device_counts.items(), key=lambda x: x[1], reverse=True)],
    }


def get_site_content():
    defaults = {
        "about_title": "StudyBuddy",
        "developer_name": "Leon Mapelera",
        "developer_role": "Creator & Developer 🇲🇼",
        "tagline": "Building tools that make learning easier.",
        "story": "I love building useful tools people can actually use. StudyBuddy turns notes, articles, files, or photos into practical study material so learning feels less like a chore and more like a tool you want to use.",
        "quote": "Study smarter. Remember more.",
        "contact_email": "leoc39063@gmail.com",
        "privacy_text": "StudyBuddy does not require an account. Material is sent to the configured AI provider to generate study content. Avoid submitting sensitive information.",
        "terms_text": "StudyBuddy is provided as-is. AI-generated study material can contain mistakes, so verify important information against trusted sources.",
        "photo_data": None, "photo_mime": None,
    }
    if not DATABASE_URL:
        return defaults
    try:
        row = db_query("SELECT about_title, developer_name, developer_role, tagline, story, quote, contact_email, privacy_text, terms_text, photo_data, photo_mime FROM site_content WHERE id=1", fetchone=True)
        if not row:
            return defaults
        keys = list(defaults.keys())
        return dict(zip(keys, row))
    except Exception:
        return defaults


@app.route("/api/site-content")
def site_content_api():
    return jsonify(get_site_content())


@app.route("/admin/content", methods=["POST"])
def admin_update_content():
    if not require_admin():
        return jsonify({"error": "Not authorized."}), 403
    if not DATABASE_URL:
        return jsonify({"error": "No database configured."}), 500
    fields = ["about_title","developer_name","developer_role","tagline","story","quote","contact_email","privacy_text","terms_text"]
    values = [request.form.get(f, "").strip() for f in fields]
    photo_data = None
    photo_mime = None
    photo = request.files.get("photo")
    if photo and photo.filename:
        allowed = {"image/jpeg":"jpg", "image/png":"png", "image/webp":"webp"}
        if photo.mimetype not in allowed:
            return jsonify({"error": "Photo must be JPG, PNG, or WebP."}), 400
        raw = photo.read()
        if len(raw) > 2 * 1024 * 1024:
            return jsonify({"error": "Photo must be 2MB or smaller."}), 400
        photo_data = "data:" + photo.mimetype + ";base64," + base64.b64encode(raw).decode("ascii")
        photo_mime = photo.mimetype
    current = get_site_content()
    if not values[0]:
        return jsonify({"error": "About title is required."}), 400
    if photo_data is None:
        photo_data, photo_mime = current.get("photo_data"), current.get("photo_mime")
    db_query("""INSERT INTO site_content (id, about_title, developer_name, developer_role, tagline, story, quote, contact_email, privacy_text, terms_text, photo_data, photo_mime, updated_at)
        VALUES (1,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
        ON CONFLICT (id) DO UPDATE SET about_title=EXCLUDED.about_title, developer_name=EXCLUDED.developer_name, developer_role=EXCLUDED.developer_role, tagline=EXCLUDED.tagline, story=EXCLUDED.story, quote=EXCLUDED.quote, contact_email=EXCLUDED.contact_email, privacy_text=EXCLUDED.privacy_text, terms_text=EXCLUDED.terms_text, photo_data=EXCLUDED.photo_data, photo_mime=EXCLUDED.photo_mime, updated_at=NOW()""", tuple(values+[photo_data,photo_mime]), commit=True)
    return jsonify({"ok": True, "content": get_site_content()})


@app.route("/admin")
def admin_dashboard():
    if not require_admin():
        return redirect("/admin/login")
    if not DATABASE_URL:
        return render_template("admin_dashboard.html", no_db=True, content=get_site_content())
    stats = gather_admin_stats()
    return render_template("admin_dashboard.html", no_db=False, content=get_site_content(), **stats)


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
