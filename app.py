import os
import io
import json
import base64
import sqlite3
import secrets
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pypdf import PdfReader
import docx
from bs4 import BeautifulSoup

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB upload cap

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
DB_PATH = os.environ.get("DB_PATH", "shared_decks.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS shared_decks (
            id TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            title TEXT,
            data TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    return conn


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

    if mode not in ("flashcards", "quiz") or not deck or not isinstance(deck, list):
        return jsonify({"error": "Invalid deck data."}), 400
    if len(deck) > 25:
        return jsonify({"error": "Deck too large to share."}), 400

    share_id = generate_share_id()
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO shared_decks (id, mode, title, data) VALUES (?, ?, ?, ?)",
            (share_id, mode, title, json.dumps(deck)),
        )
        conn.commit()
    finally:
        conn.close()

    return jsonify({"id": share_id})


@app.route("/api/share/<share_id>", methods=["GET"])
def get_share(share_id):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT mode, title, data FROM shared_decks WHERE id = ?", (share_id,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return jsonify({"error": "This shared deck wasn't found — the link may be old or invalid."}), 404

    mode, title, data = row
    return jsonify({"mode": mode, "title": title, "deck": json.loads(data)})


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
    "- CRITICAL FORMATTING RULE: never use markdown symbols like **, *, #, or bullet dashes "
    "in your reply — this is displayed as plain text and may be read aloud, so raw asterisks "
    "would look broken. Write in plain sentences and paragraphs only. If you need to list "
    "steps, write them as 'First, ... Next, ... Then, ...' in prose, not as a markdown list."
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
    if not isinstance(history, list) or len(history) > 60:
        return jsonify({"error": "Conversation too long — please start a new tutor session."}), 400

    messages = [{"role": "system", "content": TUTOR_SYSTEM_PROMPT}]
    for turn in history[-40:]:
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
        return jsonify({"error": f"Upstream API error: {e}"}), 502
    except (KeyError, IndexError):
        return jsonify({"error": "The tutor didn't respond properly. Try again."}), 502

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
    if mode not in ("flashcards", "quiz"):
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
            return jsonify({"error": f"Upstream API error: {e}"}), 502
        except (KeyError, IndexError, json.JSONDecodeError):
            return jsonify({"error": "The model returned something unexpected. Try again."}), 502
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

    return jsonify({key: merged[:count if count >= len(chunks) else len(merged)]})


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Too many requests — please slow down and try again shortly."}), 429


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
