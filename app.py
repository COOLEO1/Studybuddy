import os
import io
import json
import base64
import requests
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

# Hard cap on how much text a single request can send, to control token cost
MAX_INPUT_CHARS = 12000


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
    """Use Mistral's vision model to read text out of a photo/image."""
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
                            "Transcribe all readable text in this image exactly as written. "
                            "Output only the transcribed text, no commentary, no markdown."
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
    return render_template("index.html")


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

    return jsonify({"text": text[:MAX_INPUT_CHARS]})


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

    return jsonify({"text": text[:MAX_INPUT_CHARS]})


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
    if len(text) > MAX_INPUT_CHARS:
        return jsonify({"error": f"Text is too long (max {MAX_INPUT_CHARS} characters)."}), 400
    if mode not in ("flashcards", "quiz"):
        return jsonify({"error": "Invalid mode."}), 400
    try:
        count = max(3, min(int(count), 15))
    except (TypeError, ValueError):
        count = 8

    prompt = build_prompt(mode, text, count, difficulty)

    try:
        result = call_mistral(prompt)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Upstream API error: {e}"}), 502
    except (KeyError, IndexError, json.JSONDecodeError):
        return jsonify({"error": "The model returned something unexpected. Try again."}), 502

    return jsonify(result)


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Too many requests — please slow down and try again shortly."}), 429


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
