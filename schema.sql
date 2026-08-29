create table if not exists site_content (
  key text primary key,
  value text not null,
  updated_at timestamptz default now()
);
create table if not exists events (
  id bigserial primary key,
  event_type text not null,
  mode text,
  path text,
  device text,
  created_at timestamptz default now()
);
create table if not exists errors (
  id bigserial primary key,
  message text not null,
  path text,
  created_at timestamptz default now()
);
create index if not exists idx_events_created_at on events(created_at);
create index if not exists idx_events_type on events(event_type);
