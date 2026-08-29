# Study Buddy

A neo-brutalist/editorial-styled AI study platform: flashcards, quizzes, exams (mixed question types with AI-graded typed answers), an AI tutor with voice input and math rendering, a past papers library, and a full real-time admin dashboard. Built by Leon Mapelera.

## Architecture

The frontend is a React + Vite + Tailwind + Framer Motion single-page app, **pre-built to static files** in `frontend_dist/`. Flask serves those static files for every page route and provides all the JSON APIs. This means:

- **No Node.js needed to deploy** — the frontend is already built and committed as plain HTML/CSS/JS in `frontend_dist/`.
- One Flask service, one Render deployment, same simple `git push` flow as before.
- If you ever want to change the frontend yourself, the React source isn't included in this deploy package (it was built separately) — ask for it if you want to modify the UI further.

## Run locally

```
pip install -r requirements.txt
export MISTRAL_API_KEY="your-key-here"
export DATABASE_URL="your-supabase-postgres-url-here"
export ADMIN_PASSWORD="choose-a-password"
export SECRET_KEY="any-long-random-string"
export SUPABASE_URL="https://your-project-ref.supabase.co"
export SUPABASE_SERVICE_KEY="your-supabase-service-role-key"
python app.py
```

Visit http://localhost:5000

## Setting up Supabase Storage (profile photo / admin uploads)

1. In your Supabase dashboard: **Storage → New bucket** → name it `studybuddy-content` (or whatever you set `SUPABASE_BUCKET` to) → make it **public**.
2. Get your **Project URL** and a **service_role key** (Settings → API) — set these as `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`.
3. Upload your profile photo from the admin panel at `/admin/content` once deployed.

## Setting up Supabase (persistent database)

Shared decks, the papers library, admin analytics, and site content (About/Privacy/Terms text) all live in a real Supabase Postgres database — this survives redeploys.

1. In your Supabase project dashboard: **Settings → Database → Connection string**.
2. Choose the **Connection pooling** (transaction pooler) URI — not the direct connection — since Render's free web service works better with pooled connections. It looks like:
   `postgresql://postgres.xxxx:[password]@aws-0-region.pooler.supabase.com:6543/postgres`
3. Set that full string as the `DATABASE_URL` environment variable.
4. Tables are created automatically the first time the app starts — no manual migration needed.

## Admin dashboard

Visit `/admin` on your deployed site. You'll be asked to log in with the `ADMIN_PASSWORD` you set. From there you get:
- Live-ish stats (active visitors, pageviews, decks generated, tutor usage) that auto-refresh every 12 seconds
- A traffic chart (last 14 days) and a generation-format breakdown chart
- Recent visits with IP address and parsed device/browser summary
- Recent errors, so you catch problems fast

Set `SECRET_KEY` to a fixed random string (not left blank) so admin login sessions survive server restarts — otherwise everyone gets logged out every time Render restarts the service.

## Deploy to Render (free tier works fine)

**Option A — one-click via Blueprint (render.yaml is already set up):**
1. Push this folder to a GitHub repo.
2. Go to https://dashboard.render.com → New → Blueprint.
3. Connect the repo. Render reads `render.yaml` and configures everything automatically.
4. When prompted, paste in `MISTRAL_API_KEY`, `DATABASE_URL`, `ADMIN_PASSWORD`, and `SECRET_KEY`.
5. Deploy. You'll get a public URL like `recall-study-buddy.onrender.com`.

**Option B — manual setup:**
1. Push this folder to a GitHub repo.
2. Go to https://dashboard.render.com → New → Web Service.
3. Connect the repo.
4. Set:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Under Environment, add `MISTRAL_API_KEY` = your key.
6. Deploy.

**Heads up on Render's free tier**: free web services spin down after 15 minutes of no traffic, so the first request after idling takes ~30-50 seconds to wake back up (a "cold start"). Totally fine for a side project; if that ever bugs you, Render's cheapest paid tier ($7/mo) keeps it always-on.

## Cost control (already built in)

- Rate limit: 10 requests/minute and 60/hour per visitor IP (in-memory — resets on restart; fine for a small public app).
- Input capped at 60,000 characters total. Text over ~12,000 characters is automatically split into up to 5 chunks (on paragraph boundaries), generated separately, and merged into one deck — so long material like a full play or book chapter still works.
- Card count capped at 3–25 per generation.

## Shared decks (SQLite)

The "Share deck" feature stores decks in a local SQLite file (`shared_decks.db`). On Render's free tier, disk storage can be wiped on redeploys or when a service is rebuilt from scratch — shared links may stop working after that happens. For guaranteed-permanent links, swap this for a hosted database (e.g. Render's free Postgres tier) later.

If it gets popular and you want tighter control, consider:
- Lowering the rate limits in `app.py` (`Limiter` config).
- Switching `storage_uri` to Redis so rate limits persist across restarts/instances.
- Adding a daily global request cap.
