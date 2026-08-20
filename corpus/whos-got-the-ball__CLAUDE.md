# Working in this repo

A contract-delivery handoff tracker: Flask + SQLAlchemy + Strawberry GraphQL
behind React + TypeScript. Full context is in [README.md](README.md); this file
covers only the things that aren't obvious from reading the code.

## Deliberate constraints

The lightness is a design choice, not an omission. Don't add without asking:

- **No Apollo or urql.** `src/api.ts` is a plain `fetch` client on purpose.
- **No CSS framework and no CSS-in-JS.** One `styles.css` with design tokens.
- **No state library.** `useState` and `useMemo` are sufficient at this size.
- **No new runtime dependencies** in `backend/requirements.txt` — it's what the
  Docker image installs, so additions ship to production. Dev-only tools go in
  `requirements-dev.txt`.

## Commands

```bash
cd backend && .venv/bin/pytest -v        # backend tests
cd frontend && npx tsc --noEmit          # frontend typecheck
docker build -t whos-got-the-ball . && docker run --rm -p 8080:8080 whos-got-the-ball
```

Both checks should pass before anything is considered done. Development runs as
two processes; see the README.

## Conventions

- **Domain rules belong in `models.py`**, not in React components. `is_overdue`,
  `is_stalled`, and `days_waiting` are computed there and arrive as flags; the
  frontend decides how to *show* them and nothing more. If you catch yourself
  writing a threshold in a `.tsx` file, it goes in `models.py` instead.
- **The board's counts and its filter are one predicate.** `ATTENTION` in
  `BoardFilter.tsx` is what both the tile numbers and the visible list run
  through. A number you can read but not click is the thing that was removed;
  don't add one back.
- **One SQLAlchemy idiom.** Reads go through `db.session` — `db.select(...)` for
  collections, `db.session.get(...)` by primary key. Don't reintroduce
  `Model.query`.
- **Annotate every function**, including return types. The backend is
  near-complete on this; keep it that way.
- **Tests are documentation.** `test_api.py` should stay readable top to bottom
  as a tour of the GraphQL API.
- **Comments explain why, not what.** Match the density of the surrounding file.
- Never commit `backend/ball.db` — it's generated and gitignored.
