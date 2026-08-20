# Secondhand Plastic — working notes

Matches secondhand food containers with people who'll use them, instead of the bin.
Server-rendered FastAPI + Jinja2, no frontend build step. Public repo, so assume
anything committed is world-readable.

## Commands

```bash
uv sync                    # install
make up                    # Postgres on :5433, wait for it, apply migrations
uv run fastapi dev         # local server on :8000
uv run pytest -q           # tests
uv run ruff check . --fix && uv run ruff format .   # lint + format (CI checks both)
```

`make up` is the only shortcut, because starting the container, waiting for it and
migrating is three steps that must happen in order. Everything else runs the ordinary
way. `make` on its own lists the rest (`down`, `reset`, `psql`, `logs`).

Postgres must be running before `pytest` or `fastapi dev`, or both fail with
`connection refused`.

After changing a model, generate a migration and commit it with the model change:

```bash
uv run alembic revision --autogenerate -m "what changed"
```

`tests/test_migrations.py` fails if the two ever drift. To reset the local database,
`make reset` — the old "delete `data/`" trick is gone with SQLite.

The suite runs against `containerswap_test`, a separate database it creates on first
run, because it drops and truncates tables wholesale. Never point it at the database
you are developing against.

## Deployment

**Pushing to `main` deploys to production.** FastAPI Cloud's GitHub integration builds
and ships the default branch automatically. There is no staging. CI runs tests on
push but does not gate the deploy, so a red build still ships — check tests locally
before pushing to main.

Env vars live in the FastAPI Cloud dashboard (App → Environment Variables). Secrets
must be marked Secret **at creation**; the toggle is not available afterwards.

## Platform constraints that shape the code

- **No persistent volumes.** Anything written to local disk is lost on redeploy.
  State lives in Supabase: Postgres for data, Storage for uploaded images. `data/`
  is local-dev scratch space only.
- **Zero-downtime deploys run old and new instances at once.** Nothing may assume a
  single process. Consequences: `app/ratelimit.py` is Postgres-backed rather than
  in-memory, so limits are shared across instances and survive a restart; the app
  does no schema work at startup, so migrations run as a separate step before the
  deploy; and a migration must leave the *old* code working, since it keeps serving
  during the rollover.
- **Connect through the transaction pooler (6543), migrate through the direct
  connection (5432).** Direct connections are too few for multiple instances, and
  DDL through the pooler is unreliable. Server-side prepared statements are disabled
  in `db._make_engine` for the same reason.

## Non-negotiables

These are promises made to users in `SECURITY.md`. Treat a change that breaks one as
a bug, not a tradeoff:

1. **A user's email is never rendered.** Not in a page, an API response, or a log.
   The public label is `display_name`. `tests/test_privacy.py` enforces this.
2. **Exact coordinates are never stored.** `app/geo.fuzz` jitters before write; the
   browser's precise value is dropped.
3. **Uploads are re-encoded from raw pixels** so EXIF GPS cannot survive.
4. **`/inbox` only returns rows where `recipient_id` is the session user.**
5. **No feature pretends to work.** A control that silently does nothing is a bug of
   the same kind as the four above — it is a promise to the user that the code does
   not keep. Either wire it up or take it out. This is why the report button now
   emails someone, and why the PWA layer and public comments were removed rather
   than left half-alive.

## Conventions

- Every user-facing string goes through `t()` against `locales/en.json`. No English
  in templates. v1 ships English only; that is not a reason to hardcode.
- Dates render client-side in the visitor's locale from ISO-8601. Never format a date
  server-side, and never assume USD or US date order.
- Prices are free text (`$5`, `500 KSh`, `free`, `trade for basil starts`). Do not
  add a structured currency field.
- Every state-changing POST takes `Depends(verify_csrf)` and includes the hidden
  `csrf_token` input in its form.
- Simplest working implementation wins. Do not add an abstraction for a second
  backend that does not exist yet.

## Style

Comments explain *why*, especially for the privacy and platform decisions above —
those look like over-engineering until you know the reason. Skip comments that
restate the code.
