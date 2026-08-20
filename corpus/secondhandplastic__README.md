# Secondhand Plastic

**Plastic tubs, jars and containers looking for a second life.**

[secondhandplastic.com →](https://secondhandplastic.com/)

Matches secondhand food containers — deli tubs, yogurt pots, glass jars — with
someone who'll use them, instead of leaving that to the bin. The gap this fills is
specifically plastic: jars already have easy homes elsewhere (thrift stores,
Freecycle), but deli tubs and yogurt cups are polypropylene (#5), and only 3-8% of
it actually gets recycled in practice — most municipal programs don't take it,
whatever the symbol on the bottom says. Mobile-first, low-bandwidth, and built to
work as well on a cheap phone in Nairobi as on a laptop in Arcata.

Server-rendered FastAPI + Jinja2 + Postgres. No frontend build step, no bundler, no
node_modules — the mobile-first, low-bandwidth goal above is easier to hold onto
without one, and the whole CSS + JS footprint comfortably stays under 30 KB as a
result.

---

## Design commitments

These are the decisions worth defending, not implementation trivia:

**Nobody's contact details are ever published.** Email is the login identity and is
never rendered on any page, in any API response, or in any log line. What strangers
see is `display_name`, a separate public label that we generate at signup so nobody
has to invent one — signup is two fields with no "that name is taken" dance. Buyers
reach sellers through a contact form that writes to a `messages` table, readable only
by the recipient at `/inbox`. This is the Craigslist model: hold the address, relay
through it, never publish it. A test asserts a known address never appears in any
response, signed in or out.

**A pin is a neighbourhood, not a doorstep.** Coordinates are jittered by up to
400 m before they are written, and the precise value the browser reported is never
stored or logged. Listing pages draw a 500 m circle rather than a point, so the
approximation is visible rather than implied.

**Photos are stripped and shrunk.** Every upload is decoded and re-encoded from raw
pixels, which drops EXIF — including the GPS tags a phone camera embeds by default —
and caps the result at ~300 KB and 1280 px.

**Price is free text.** `$5`, `500 KSh`, `free`, `trade for basil starts` are all
valid. A structured currency field would quietly exclude barter and most of the world.

**Nothing is hardcoded English.** Every visible string goes through `t()` against
`locales/en.json`. Adding a language is one JSON file — no template changes. Dates
render in the visitor's own locale and timezone, client-side, from ISO-8601.

**One deployment, framed locally for everyone.** Cloudflare's `CF-IPCountry` header
gives the tagline and the map's default view a per-visitor country ("in Kenya and
beyond"), with no config and no per-region deployment — see `app/region.py`. It's
read-only and request-scoped, matching the "no IP address persisted" promise above. A
country outside that table, or a request with no Cloudflare in front (local dev),
falls back to the plain global tagline and the Humboldt County default in
`CS_DEFAULT_LAT`/`CS_DEFAULT_LNG` — a nod to where this started, not the primary
framing anyone sees.

---

## Run it locally

```bash
uv sync
uv run playwright install chromium   # once, for the e2e tests in tests/e2e
make up                       # Postgres on :5433, then migrate
uv run fastapi dev
```

`make up` is the one shortcut — starting the container, waiting for it and migrating
has to happen in that order. Tests, lint and the server run the ordinary way:
`uv run pytest`, `uv run ruff check .`, `uv run fastapi dev`.

Open http://127.0.0.1:8000. Tests: `uv run pytest`. Lint: `uv run ruff check .`

No `.env` is needed: the defaults point at the Docker database, and with no object
store configured uploads go to `data/uploads` and are served from `/uploads`. Local
development never touches production storage.

Tests run against a separate `containerswap_test` database, created on first run,
because the suite drops and truncates tables wholesale. To reset development data,
`docker compose down -v`.

After changing a model, generate a migration and commit it alongside:

```bash
uv run alembic revision --autogenerate -m "what changed"
```

`tests/test_migrations.py` fails if the models and migrations ever drift apart.

## Deploy

Pushing to `main` deploys. FastAPI Cloud builds the default branch automatically, and
CI does not gate it — a red build still ships, so run the tests before you push. The
`.githooks/pre-push` hook enforces that, plus one thing CI cannot check: that
production's schema is already at the repo's migration head.

**Migrate before you push, unless the migration is `post_deploy`.** There is no
release phase, so the new code starts serving the moment the push lands. Applying an
ordinary migration first is safe because it's additive and the old instances keep
working through the rollover:

```bash
CS_CONFIRM_PROD_MIGRATION=yes uv run alembic upgrade head
git push origin main
```

A migration that removes something the *currently deployed* code still reads is the
opposite case — applying it first would break the old instances still serving through
the rollover. Mark it `post_deploy = True` in the revision file instead (see
`migrations/deploy_safety.py`), and the hook lets the push land ahead of it, on the
condition that every migration production is missing is marked that way:

```bash
git push origin main
# once the deploy has finished rolling over:
CS_CONFIRM_PROD_MIGRATION=yes uv run alembic upgrade head
```

`CS_MIGRATION_DATABASE_URL` lives in `.env` (production's direct connection, port 5432)
since it's only ever set to production — but `migrations/env.py` refuses to use it
unless `CS_CONFIRM_PROD_MIGRATION` is also passed inline, so a bare `alembic upgrade
head` run out of habit can't silently reach production. `make up` never hits this: it
forces both variables to the local container regardless of `.env`.

Environment variables, set in the FastAPI Cloud dashboard. Anything secret must be
marked Secret **at creation** — the toggle disappears afterwards:

| Variable | Value |
| --- | --- |
| `CS_SECRET_KEY` | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `CS_DEBUG` | `false` |
| `CS_DATABASE_URL` | Supabase **transaction pooler**, port 6543 |
| `CS_SUPABASE_URL` | `https://<project-ref>.supabase.co` |
| `CS_SUPABASE_SERVICE_KEY` | a **secret** key — publishable and anon cannot write |
| `CS_RESEND_API_KEY` | Resend API key; without it nobody is told they have a message |
| `CS_EMAIL_FROM` | an address on a domain verified with Resend |
| `CS_REPORT_EMAIL` | where abuse reports *and* beta feedback (`/feedback`) go |
| `CS_SITE_URL` | `https://…` — used for links in emails |

`CS_STORAGE_BUCKET` defaults to `listing-photos`; the bucket must be public, since
listing photos are served to signed-out visitors and signed URLs would defeat caching
without adding privacy.

### Why the pooler

Zero-downtime deploys run old and new instances at once, and Supabase's direct
connection allows far too few connections for that — so the app connects through the
transaction pooler on 6543. Server-side prepared statements are disabled to suit it.
Migrations are the exception: DDL through the transaction pooler is unreliable, so
`CS_MIGRATION_DATABASE_URL` should point at the direct or session connection on 5432.

Nothing may assume a single process. Most migrations must be additive-only, aside
from the `post_deploy` exception above; the app does no schema work at startup; and
`app/ratelimit.py` is Postgres-backed rather than in-memory, so limits are shared
and survive a restart.

## Layout

```
main.py              app wiring, security headers
app/config.py        settings (all CS_-prefixed env vars)
app/models.py        users, listings, messages, reports, event_log
app/auth.py          argon2 + signed-cookie sessions + CSRF
app/geo.py           coordinate fuzzing
app/images.py        upload validation, EXIF stripping, compression
app/storage.py       where processed images go: local disk or object store
app/i18n.py          translation lookup
app/routes/          account.py, listings.py
migrations/          Alembic; versions/ holds the schema history; deploy_safety.py
                      backs the pre-push hook's post_deploy check
templates/           Jinja2, mobile-first
static/              css, js, vendored Leaflet, icons
locales/en.json      every user-facing string
docker-compose.yml   local Postgres for development and tests
tests/e2e/            Playwright, against a real uvicorn instance
```

## Analytics

The whole analytics layer is the `event_log` table: `listing_created`,
`listing_viewed`, `contact_sent`, `listing_completed`. No dashboard, no third-party
tracker, no IP or user-agent stored. Query it directly:

```sql
SELECT event_type, date(created_at) AS day, count(*)
FROM event_log GROUP BY 1, 2 ORDER BY 2 DESC;
```

## Contributing & security

Bug reports and PRs welcome. Please read [SECURITY.md](SECURITY.md) before reporting a
vulnerability — do not open a public issue for one.

MIT licensed. See [LICENSE](LICENSE).
