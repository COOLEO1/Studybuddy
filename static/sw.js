// Study Buddy service worker — caches the app shell so the page itself
// loads offline. API calls (generation, tutor, etc.) still need a live
// connection — this only makes the interface and previously-saved decks
// available without one.

const CACHE_NAME = 'studybuddy-shell-v1';
const SHELL_URLS = [
  '/',
  '/static/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
  '/static/icons/favicon-32.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_URLS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Never cache API calls — those need a live network response every time.
  if (url.pathname.startsWith('/api/')) {
    return;
  }

  // Cache-first for the app shell itself, falling back to network.
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).catch(() => {
        // Offline and not cached — for navigations, fall back to the cached homepage.
        if (event.request.mode === 'navigate') {
          return caches.match('/');
        }
      });
    })
  );
});
