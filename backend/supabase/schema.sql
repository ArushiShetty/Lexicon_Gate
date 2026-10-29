-- Lexicon Gate / Self-RAG — Supabase schema
-- Run this in the Supabase SQL Editor (Project → SQL → New query).
-- Uses service-role access from the backend; RLS is enabled and locked down.

create table if not exists app_users (
  user_id text primary key,
  email text not null unique,
  name text not null default '',
  password_hash text not null default '',
  google_sub text unique,
  auth_provider text not null default '',
  researcher_id text unique,
  created_at timestamptz not null default now()
);

create index if not exists app_users_email_idx on app_users (email);
create index if not exists app_users_researcher_id_idx on app_users (researcher_id);

create table if not exists app_sessions (
  token text primary key,
  user_id text not null references app_users (user_id) on delete cascade,
  email text not null default '',
  name text not null default '',
  researcher_id text not null default '',
  expires_at timestamptz not null,
  created_at timestamptz not null default now()
);

create index if not exists app_sessions_user_id_idx on app_sessions (user_id);
create index if not exists app_sessions_expires_at_idx on app_sessions (expires_at);

create table if not exists app_stories (
  story_id text primary key,
  author_user_id text not null,
  payload jsonb not null,
  updated_at timestamptz not null default now()
);

create index if not exists app_stories_author_idx on app_stories (author_user_id);
create index if not exists app_stories_updated_idx on app_stories (updated_at desc);

alter table app_users enable row level security;
alter table app_sessions enable row level security;
alter table app_stories enable row level security;

-- Backend uses the service role key (bypasses RLS). No anon policies by default.
