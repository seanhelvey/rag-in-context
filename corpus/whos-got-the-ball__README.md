# Who's Got the Ball

A small full-stack app for a real clean-energy delivery painpoint: on a solar,
storage, or efficiency contract, work constantly changes hands between **your own
team, the customer, the utility, the installer, the financier, and the permitting
office**. At any moment one of them "has the ball" — owns the next action before
the deal can move. When the ball sits too long, deals stall. This app makes the
current owner, the pending action, and the full handoff history visible at a
glance.

The domain is clean energy, but nothing about the model is specific to it: the
stakeholder roles and contract stages are lists in `models.py`, so any
multi-party delivery pipeline with handoffs fits the same three tables.

Stack: **Flask + SQLAlchemy + Strawberry GraphQL** on the backend, **React +
TypeScript** on the frontend, SQLite by default and Postgres via `DATABASE_URL`,
deployed as a single Docker image.

<br>

## Run it with Docker

The only prerequisite is Docker. One image, one process — Flask serves the
GraphQL API *and* the built React bundle, so there's a single thing to deploy:

```bash
docker build -t whos-got-the-ball .
docker run --rm -p 8080:8080 whos-got-the-ball
```

Then open **http://localhost:8080** — UI, API, and the GraphiQL explorer all on
one port, no CORS and no dev proxy involved. The SQLite database is created and
seeded inside the container on boot, so there's nothing to configure. To run
against Postgres instead, pass `-e DATABASE_URL=postgresql://…`.

## Run it for development

For hot reload you run the two sides separately, which needs **Python 3.9+** and
**Node 18+**. Two terminals:

```bash
# Terminal 1 — API on :5000, auto-reloads on save
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python app.py                      # honours HOST, PORT, DEBUG

# Terminal 2 — UI on :5173, hot module reload
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. Vite proxies `/graphql` to Flask, so there's no
CORS to configure in development either.

`pip install` works fine, but [`uv`](https://github.com/astral-sh/uv) does the
same install in a fraction of the time if you have it:
`uv pip install -r requirements-dev.txt`.

<br>

## What each piece is

```
Dockerfile              one image: Flask serving the API and the built React bundle
backend/
  models.py             SQLAlchemy models: Stakeholder, Contract, Handoff
  schema.py             GraphQL schema (Strawberry) — types, queries, and the
                        passBall and updateStatus mutations
  seed.py               Fictional-but-realistic seed data
  app.py                Flask entrypoint; serves GraphQL at /graphql
  test_api.py           Backend tests, readable as an API tour
  requirements.txt      Runtime dependencies (what the image installs)
  requirements-dev.txt  The above, plus pytest
frontend/
  src/types.ts          TypeScript types mirroring the GraphQL schema
  src/api.ts            A small fetch-based GraphQL client (no Apollo)
  src/format.ts         Pure presentation helpers (dates, currency, due status)
  src/App.tsx           Board, filter state, and the pass-the-ball flow
  src/components/       BoardFilter (the counts *are* the filter), card, modal,
                        timeline, and small primitives
  src/styles.css        One plain CSS file — design tokens + styles, no framework
.claude/agents/         Two review agents (see below)
```

Deliberately **light**: no Apollo, no CSS framework, no state library — just
React, a tiny `fetch` GraphQL helper, and one stylesheet. The frontend dev
server proxies `/graphql` to Flask, so there's no CORS to deal with.

### Review agents

Two Claude Code subagents live in `.claude/agents/`, each carrying a different
review rubric so they can run independently:

| Agent | Lens |
| --- | --- |
| `product-advisor` | The delivery lead who named the painpoint — would this survive contact with real contract delivery, and what would stop people using it |
| `staff-engineer` | The engineering bar this repo holds itself to — type coverage, test ratio, resolver efficiency, schema lifecycle |

`.claude/settings.json` also configures two hooks: the tests and typecheck run when
a turn ends (surfacing a message only when something fails), and writes to
generated paths — `ball.db`, `dist/`, `.venv/`, `node_modules/`, `__pycache__/`
— are refused.

<br>

## The stack, and where to look

| Learning goal | Where it lives |
| --- | --- |
| **Flask** app + routing | `backend/app.py` |
| **SQLAlchemy** models + relationships | `backend/models.py` |
| **GraphQL** schema, queries, mutations | `backend/schema.py` |
| **TypeScript** domain types | `frontend/src/types.ts` |
| **React** components + hooks | `frontend/src/App.tsx`, `frontend/src/components/` |
| Talking to GraphQL from the client | `frontend/src/api.ts` |

### Tests (also the clearest tour of the API)

The backend tests double as documentation — read `backend/test_api.py` top to
bottom to see every query and the `passBall` mutation in action. They run
against a throwaway database, so they never touch your real data:

```bash
cd backend && .venv/bin/pytest -v
```

And the frontend typechecks with `cd frontend && npx tsc --noEmit`.

Try a query in GraphiQL:

```graphql
{
  contracts {
    name
    isOverdue
    currentHolder { name organization kind }
    currentAction
  }
}
```

<br>

## Could it drop into an existing system?

The backend is intentionally a thin layer over three tables, which makes a few
interop paths realistic:

- **Swap the datastore.** The database is read from `DATABASE_URL`, defaulting to
  local SQLite: `DATABASE_URL=postgresql://… python app.py` boots and creates its
  tables against Postgres. Two honest caveats before that's a production path —
  there are no migrations (the app calls `create_all()`, which is a demo
  convenience), and timestamps are stored as naive UTC to keep SQLite
  comparisons honest, so they'd want to become `timestamptz`.
- **Map onto existing records.** `Contract`, `Stakeholder`, and `Handoff` map
  cleanly onto a deal / contact / activity model. Only the resolvers in
  `schema.py` would change to read from existing tables — the GraphQL contract
  and the whole frontend stay the same.
- **Use `passBall` as the hook.** That single mutation is the natural place to
  fire a Slack/email nudge to the new owner or write the handoff back to the CRM,
  so "who has the ball" can ride on top of tools people already use.

<br>

## What I'd do next

Known gaps, roughly in the order I'd close them:

- **The data-entry problem.** The app knows who has the ball only because someone
  typed it in — but the reason nobody knows who has the ball is that the handoff
  happened in an email thread. Reading the thread and proposing the handoff is
  the feature that makes this real rather than another tracker to maintain.
- **Nothing nudges anyone.** `passBall` is the natural hook for a Slack or email
  poke to the new owner; today the board only rewards people who remember to
  visit it.
- **Resolvers over-fetch.** The `map_*` helpers in `schema.py` build the whole
  object graph regardless of what the client selected, which means N+1 queries
  through the relationships. Lazy field resolvers plus a dataloader is the
  idiomatic Strawberry fix.
- **Migrations.** Alembic, so the Postgres path above is real.
- **Enums, enforced in the schema.** `STAKEHOLDER_KINDS` and `CONTRACT_STATUSES`
  are lists nothing checks, while the TypeScript side declares strict literal
  unions — so the client asserts a guarantee the server doesn't hold. The
  mutations now validate against those lists by hand, which closes the hole but
  puts the rule in a second place. Real GraphQL enums would enforce it at the
  schema boundary, document it in the introspected schema, and flow the types
  outward instead.
- **More tests.** The backend is around 20% tests by line, against the ~30% bar
  I'd want to hold. There are no frontend tests at all.
- **`updateStatus` isn't wired up.** The mutation exists and is tested, but the
  UI never calls it, so a deal can't be moved between stages from the app.

## Notes

- The DB seeds only when empty, so restarts keep any handoffs you record. Delete
  `backend/ball.db` to reset to the sample data.
- There's no authentication: any caller can pass the ball on any contract, and
  the Docker image serves the GraphiQL explorer on the same port as the UI, so
  every mutation is reachable from a browser. That's a deliberate omission for a
  learning project, not an oversight — but it's the first thing to close before
  this is exposed to anyone.
- Everything is fictional; names and numbers are illustrative.
