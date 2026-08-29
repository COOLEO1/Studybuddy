# StudyBuddy redesign notes

This package keeps the existing Flask backend and study functionality while upgrading the public UI and admin console.

## Public UI
- Midnight Scholar palette: midnight navy, indigo/violet, cyan accents.
- Responsive mobile-first surfaces and cleaner controls.
- Dynamic About content loaded from `/api/site-content`.
- Public GitHub source link removed from About; contact Gmail remains.
- Developer photo, story, tagline, quote, privacy text, terms, and contact email can be changed from Admin > Content.
- Existing KaTeX/math support is preserved.

## Admin
- Overview KPI cards.
- Traffic line chart.
- Study-mode doughnut chart.
- Generation trend line chart.
- Device mix doughnut chart.
- Recent visits and errors.
- Content management for About, story, quote, photo, email, privacy and terms.
- System health panel.

## Developer photo
Recommended path for a local static asset: `static/uploads/leon.jpg`.
For the public deployment, use the Admin > Content photo uploader so the image is stored with the database-backed site content and does not require rebuilding the frontend.

## Environment
Keep using the existing environment variables, especially `DATABASE_URL`, `SECRET_KEY`, `ADMIN_PASSWORD`, and `MISTRAL_API_KEY`.
