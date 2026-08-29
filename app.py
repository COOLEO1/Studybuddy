import os, json, time, secrets, hashlib
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

load_dotenv()
BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
UPLOADS = STATIC / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

APP_NAME = "L3o Study"
CREATOR = "Leon Mapelera"
COUNTRY = "Malawi 🇲🇼"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

try:
    import psycopg
except Exception:
    psycopg = None

app = FastAPI(title=APP_NAME, version="2.0.0", docs_url="/docs", redoc_url="/redoc")
secure_cookie = os.getenv("RENDER", "false").lower() == "true"
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "dev-only-change-this"),
    same_site="lax",
    https_only=secure_cookie,
    max_age=60 * 60 * 12,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

SCHEMA = """
CREATE TABLE IF NOT EXISTS site_content (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS events (
  id BIGSERIAL PRIMARY KEY,
  event_type TEXT NOT NULL,
  mode TEXT,
  path TEXT,
  device TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS errors (
  id BIGSERIAL PRIMARY KEY,
  message TEXT NOT NULL,
  path TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
"""

DEFAULT_CONTENT = {
    "hero_title": "Study smarter. Learn deeper.",
    "hero_subtitle": "Turn notes, questions and study material into focused practice with an AI workspace built for learning.",
    "developer_name": CREATOR,
    "developer_role": "Founder & Developer · Malawi 🇲🇼",
    "developer_story": "I built L3o Study to make learning feel more focused, practical and personal — less scrolling, more understanding.",
    "founder_quote": "Build tools that make learning easier.",
    "contact_email": "leoc39063@gmail.com",
    "privacy": "L3o Study stores information needed to provide the service and improve reliability. Do not submit passwords, payment details, or other sensitive secrets as study material. AI output can contain mistakes and should be checked for important academic work.",
    "terms": "Use L3o Study responsibly and only with material you are permitted to use. AI-generated material is an aid to learning, not a replacement for teachers, textbooks, or official exam guidance.",
    "announcement": "New · A calmer AI study workspace is here.",
}

ALLOWED_MODES = {"flashcards", "quiz", "exam", "tutor", "summary", "study_plan"}
RATE_WINDOW = 60
RATE_LIMIT = 20
_hits: dict[str, list[float]] = {}


def db_conn():
    if not DATABASE_URL or psycopg is None:
        return None
    return psycopg.connect(DATABASE_URL, connect_timeout=5)


def db_init() -> bool:
    conn = db_conn()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
            for k, v in DEFAULT_CONTENT.items():
                cur.execute(
                    "INSERT INTO site_content(key,value) VALUES(%s,%s) ON CONFLICT(key) DO NOTHING",
                    (k, v),
                )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        conn.close()


def get_content() -> dict[str, str]:
    data = DEFAULT_CONTENT.copy()
    conn = db_conn()
    if not conn:
        return data
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT key,value FROM site_content")
            data.update(dict(cur.fetchall()))
    except Exception:
        pass
    finally:
        conn.close()
    return data


def client_device(request: Request) -> str:
    ua = request.headers.get("user-agent", "").lower()
    return "mobile" if any(x in ua for x in ("android", "iphone", "ipad", "mobile")) else "desktop"


def track(event_type: str, request: Request, mode: str | None = None):
    conn = db_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO events(event_type,mode,path,device) VALUES(%s,%s,%s,%s)",
                (event_type, mode, request.url.path, client_device(request)),
            )
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


def record_error(message: str, request: Request):
    conn = db_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO errors(message,path) VALUES(%s,%s)", (message[:1000], request.url.path))
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


def rate_limit(request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown").split(",")[0].strip()
    now = time.time()
    values = [x for x in _hits.get(ip, []) if now - x < RATE_WINDOW]
    if len(values) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests. Please wait a minute and try again.")
    values.append(now)
    _hits[ip] = values


def admin_ok(request: Request) -> bool:
    return request.session.get("admin") is True


def require_admin(request: Request):
    if not admin_ok(request):
        raise HTTPException(401, "Admin authentication required")


def extract_json_content(raw: Any):
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return raw
    cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return raw


@app.on_event("startup")
def startup():
    db_init()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    track("pageview", request)
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request):
    if not admin_ok(request):
        return RedirectResponse("/admin/login")
    return (STATIC / "admin.html").read_text(encoding="utf-8")


@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page():
    return (STATIC / "admin-login.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    ready = db_init()
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": "2.0.0",
        "database_configured": bool(DATABASE_URL and psycopg),
        "database_ready": ready,
        "mistral_configured": bool(os.getenv("MISTRAL_API_KEY")),
        "model": MISTRAL_MODEL,
        "ocr_model": MISTRAL_OCR_MODEL,
    }


@app.get("/api/content")
def content():
    return get_content()


@app.post("/api/generate")
async def generate(request: Request):
    rate_limit(request)
    body = await request.json()
    mode = body.get("mode", "flashcards")
    text = (body.get("text") or "").strip()
    if mode not in ALLOWED_MODES:
        raise HTTPException(400, "Unsupported study mode.")
    if not text:
        raise HTTPException(400, "Add study material first.")
    if len(text) > 50000:
        raise HTTPException(413, "Study material is too long. Keep it under 50,000 characters.")
    key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not key:
        raise HTTPException(503, "Mistral API is not configured on the server.")

    prompts = {
        "flashcards": (
            "Create 10 high-quality active-recall flashcards from the material. "
            "Return ONLY a JSON object with an `items` array. Each item must have `question` and `answer` strings. "
            "Keep answers concise and test understanding, not copying sentences."
        ),
        "quiz": (
            "Create 8 multiple-choice questions from the material. Return ONLY a JSON object with an `items` array. "
            "Each item must have `question`, `options` (exactly 4 strings), `answer_index` (0-3), and `explanation`. "
            "Do not make trick questions unless the material clearly supports them."
        ),
        "exam": (
            "Create 6 exam-practice questions from the material. Return ONLY a JSON object with an `items` array. "
            "Each item must have `question`, `answer`, and `marks` (integer). Include calculations when appropriate."
        ),
        "summary": (
            "Create a clean study summary in Markdown with: key ideas, important terms, formulas/facts, common mistakes, "
            "and a 5-line final recap. Use LaTeX for mathematical expressions when useful."
        ),
        "study_plan": (
            "Create a practical 7-day study plan based on the material. Return ONLY a JSON object with an `items` array. "
            "Each item must have `day`, `focus`, `tasks`, and `check`."
        ),
        "tutor": (
            "Act as a patient expert tutor. Explain the material step by step in Markdown. "
            "Use LaTeX for mathematics. Identify likely misconceptions, give a simple example, then finish with 5 quick checks."
        ),
    }
    instruction = prompts[mode]
    messages = [
        {
            "role": "system",
            "content": (
                "You are L3o Study, an academic learning assistant. Be accurate, encouraging and explicit about uncertainty. "
                "Never invent facts that are absent from the supplied material. When the material is insufficient, say so."
            ),
        },
        {"role": "user", "content": instruction + "\n\nSTUDY MATERIAL:\n" + text},
    ]
    payload: dict[str, Any] = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "temperature": 0.25,
        "max_tokens": 5000,
    }
    if mode in {"flashcards", "quiz", "exam", "study_plan"}:
        payload["response_format"] = {"type": "json_object"}

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                MISTRAL_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
            )
        if response.status_code >= 400:
            detail = response.text[:500]
            record_error(f"Mistral {response.status_code}: {detail}", request)
            raise HTTPException(502, "Mistral could not complete that study request.")
        data = response.json()
        out = data["choices"][0]["message"]["content"]
        track("generation", request, mode)
        return {"mode": mode, "content": extract_json_content(out), "model": MISTRAL_MODEL}
    except HTTPException:
        raise
    except Exception as exc:
        record_error(str(exc), request)
        raise HTTPException(502, "The AI service could not complete that request.")


@app.post("/api/extract-text")
async def extract_text(request: Request, file: UploadFile = File(...)):
    """Extract text from a small text/markdown file without sending it to the AI provider."""
    rate_limit(request)
    if file.content_type not in {"text/plain", "text/markdown", "text/csv", "application/json"}:
        raise HTTPException(400, "For direct extraction use TXT, Markdown, CSV or JSON. Images/PDF OCR can be added with Mistral OCR.")
    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(413, "File is too large. Maximum is 5 MB.")
    text = data.decode("utf-8", errors="replace")
    track("upload", request)
    return {"filename": file.filename, "text": text[:50000]}


@app.post("/api/admin/login")
async def admin_login(request: Request):
    body = await request.json()
    user = str(body.get("username", ""))
    pwd = str(body.get("password", ""))
    expected_user = os.getenv("ADMIN_USERNAME", "admin")
    expected_pwd = os.getenv("ADMIN_PASSWORD", "change-me")
    if secrets.compare_digest(user, expected_user) and secrets.compare_digest(pwd, expected_pwd):
        request.session["admin"] = True
        return {"ok": True}
    raise HTTPException(401, "Invalid credentials")


@app.post("/api/admin/logout")
def admin_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/api/admin/me")
def admin_me(request: Request):
    return {"authenticated": admin_ok(request)}


@app.get("/api/admin/analytics")
def analytics(request: Request):
    require_admin(request)
    conn = db_conn()
    if not conn:
        return {
            "configured": False,
            "message": "Set DATABASE_URL to enable analytics and editable content.",
            "totals": {}, "series": [], "modes": {}, "devices": {}, "errors": 0,
        }
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT event_type,COUNT(*) FROM events GROUP BY event_type")
            totals = dict(cur.fetchall())
            cur.execute("SELECT to_char(date_trunc('day',created_at),'Mon DD'),COUNT(*) FROM events WHERE created_at >= now()-interval '14 days' GROUP BY 1 ORDER BY min(created_at)")
            series = [{"day": day, "count": count} for day, count in cur.fetchall()]
            cur.execute("SELECT COALESCE(mode,'other'),COUNT(*) FROM events WHERE event_type='generation' GROUP BY 1 ORDER BY 2 DESC")
            modes = dict(cur.fetchall())
            cur.execute("SELECT device,COUNT(*) FROM events GROUP BY device")
            devices = dict(cur.fetchall())
            cur.execute("SELECT COUNT(*) FROM errors WHERE created_at >= now()-interval '14 days'")
            errors = cur.fetchone()[0]
            return {"configured": True, "totals": totals, "series": series, "modes": modes, "devices": devices, "errors": errors}
    finally:
        conn.close()


@app.get("/api/admin/content")
def admin_content(request: Request):
    require_admin(request)
    return get_content()


@app.put("/api/admin/content")
async def update_content(request: Request):
    require_admin(request)
    body = await request.json()
    updates = {k: str(v)[:5000] for k, v in body.items() if k in DEFAULT_CONTENT}
    conn = db_conn()
    if not conn:
        raise HTTPException(503, "Database is not configured")
    try:
        with conn.cursor() as cur:
            for key, value in updates.items():
                cur.execute(
                    "INSERT INTO site_content(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=now()",
                    (key, value),
                )
        conn.commit()
        return get_content()
    finally:
        conn.close()


@app.post("/api/admin/photo")
async def upload_photo(request: Request, file: UploadFile = File(...)):
    require_admin(request)
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "Use JPG, PNG or WebP")
    data = await file.read()
    if len(data) > 3 * 1024 * 1024:
        raise HTTPException(413, "Image must be 3 MB or smaller")
    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}[file.content_type]
    target = UPLOADS / f"developer.{ext}"
    target.write_bytes(data)
    for other in ("jpg", "png", "webp"):
        p = UPLOADS / f"developer.{other}"
        if p != target and p.exists():
            p.unlink()
    return {"url": f"/static/uploads/developer.{ext}?v={int(time.time())}"}


@app.exception_handler(404)
async def not_found(_: Request, __):
    return JSONResponse({"detail": "Not found"}, status_code=404)
