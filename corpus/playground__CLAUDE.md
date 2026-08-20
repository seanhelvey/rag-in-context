# Playground

A personal flywheel for habits, dreams, goals, and projects. Go API + SQLite + PWA. Deployed on Fly.io.

## Git
- Push directly to `main`, no branches.

## Local dev
```bash
cd api
go mod tidy
go run .
```
Open `http://localhost:8080`. The server seeds SQLite from `data.json` + `tasks.json` on first run and serves the PWA from `static/`.

## Deploy
```bash
fly deploy
```

## Data model

### Items (`data.json`)
Each item has:
- **Type**: `Core` (daily non-negotiables), `Habit` (daily practices being built), `Dream` (bigger aspirations), `Goal` (SMART goals with target dates). Type captures expected **cadence** — a daily habit and a slow-burn goal like homeownership both have momentum, but "stalling" means different things for each. Evaluate momentum relative to the item's natural rhythm.
- **Momentum**: `rising`, `steady`, `stalling`, `dormant` — updated based on log activity relative to expected cadence, not guesswork.
- **Focus**: one honest sentence about where this actually stands right now.
- **Next**: one specific, concrete action.
- **Milestones**: array of `{ date, label }` — wins and achievements. Proof that things are real.
- **Log**: append-only. Each entry has a date, optional type, and a short note.
  - Regular: `{ date, note }`
  - Recommendations: `{ date, type: "recommendation", note }` — specific, actionable, timely. Refreshed during check-ins.

### Goals (SMART)
Goal-type items also have:
- **target_date**: when it should be done (e.g. `"2027-03-31"`)
- **success_criteria**: one sentence defining what "done" looks like

### Engagement metric ("Stick with the process")
This item is **derived, not self-reported**. Its momentum comes from:
- Check-in streak (how many consecutive days the daily check-in ran)
- % of items updated in the last 7 days
- Goals on pace vs their target dates
The daily check-in agent should compute and report this.

### Wins (`data.json`)
`wins` array — cross-cutting good moments that prove the flywheel is working:
```json
{ "date": "2026-04-12", "note": "First time fishing the bay. Didn't catch anything but loved it." }
```
Wins can relate to specific items or cut across many. Log them during check-ins or whenever something good happens. They're the record of the life being built.

### Check-ins (`data.json`)
`check_ins` array — weekly wellness snapshots:
```json
{
  "date": "2026-03-31",
  "body": 7,
  "mind": 8,
  "social": 5,
  "feeling": "restless",
  "more_of": "time outside",
  "less_of": "evening screens"
}
```
- **body, mind, social**: 1-10 scores. Simple, honest.
- **feeling**: one word for the overall vibe.
- **more_of / less_of**: one word or short phrase each.
- Keep these public-safe — no clinical language, no private details.

### Tasks (`tasks.json`)
Agent backlog — things for the system to work on, not user to-dos:
```json
{
  "id": 1,
  "task": "Description of what needs to happen",
  "status": "pending",
  "created": "2026-04-01"
}
```
- Status: `pending` or `done`
- When a task is done, remove it. Keep the list lean.
- These are system improvement tasks, not item-level actions (those live in each item's `next` field).

## How check-ins work
**Two channels, not connected yet:**

1. **App (PWA)** — user opens the app, taps +, logs activity / records wins / does weekly check-in (body/mind/social, feeling, more/less). Data goes to SQLite via API.

2. **Claude Code (iOS)** — user opens Claude Code separately, reviews how things are going, gives tasks for system improvement. Claude reads the repo, proposes changes, commits when approved.

**When using Claude Code to update items:**
1. Read `data.json` and `tasks.json` first.
2. Update focus and momentum based on what was shared. **Evaluate momentum relative to the item's expected cadence** — a daily habit stalling after 3 missed days is different from a 1-year goal with no update in a week.
3. Append a dated log entry.
4. Add milestones for any wins or achievements.
5. Update `last_updated` to today's date.
6. **Show the user all changes and wait for their OK before committing and pushing.**

The user may share updates conversationally:
- **Direct**: "meditation 6/7 this week" → update focus, momentum, append log.
- **Conversational**: something comes up naturally → ask if they want it logged before adding.
- **Review**: "how's everything looking" → summarize the state. Be a friend, not a manager.

## Recommendations
When providing recommendations for items (especially Nature, Coloft):
- Be **specific and local** — Humboldt County, current season, named places and organizations.
- Be **timely** — what can be done THIS week or month.
- Be **actionable** — not "consider gardening" but "show up to First Saturday native gardening, 11:30am, 2nd & F St, Old Town Eureka."
- Store as log entries with `type: "recommendation"` so they persist and can be refreshed.

## Tone
Be a friend. Supportive, honest, not pushy. If something hasn't been touched in a while, mention it gently — don't lecture. Match the user's energy. Sometimes they want structure, sometimes they're just thinking out loud.

## Scope
This system is relevant when:
- The user is explicitly checking in on habits or dreams.
- Something in the conversation naturally relates to a tracked item.
- The user asks for a review or summary.

It is NOT relevant when:
- The user is working on something unrelated and hasn't referenced it.
- Forcing a connection would feel annoying.

Use judgment. When in doubt, don't bring it up.

## Adding new items
If the user mentions a new habit, interest, or dream that seems like it belongs here, ask once: "want to add that to the tracker?" Don't assume.

## Public-facing — treat like a portfolio
This repo is public. Think of it like a resume or personal brand artifact — sharing interests and growth is fine, but nothing sensitive, private, or unprofessional. No full names, no specific addresses, no personal struggles, no financial details, no private relationships. Keep the tone something you'd be comfortable with a potential collaborator or employer reading. When in doubt, leave it out.

## Vision: Intelligent PDCA Flywheel

**This is the entire point of this project.** Not a dashboard. Not a form. A system that learns and adapts through continuous improvement — for both the user's life and the system itself.

```
        ┌─────────────────────────────────────┐
        │          THE FLYWHEEL               │
        │                                     │
        │   PLAN ──── DO ──── CHECK ──── ACT  │
        │     │                          │    │
        │     └──────── REPEAT ──────────┘    │
        │                                     │
        │   Today: human-driven (manual PDCA) │
        │   Goal: system-driven (auto PDCA)   │
        └─────────────────────────────────────┘
```

### Where we are now (v1 — manual loop)
- **PWA on Fly.io** — check in, log activity, record wins from phone
- **Claude Code on iOS** — separate tool, user-initiated, reads repo + reviews state, proposes improvements
- **The loop is manual**: user checks in via app, then separately opens Claude Code to reflect and give tasks
- The two halves (app data + Claude intelligence) are **not connected yet**

### Where this is headed (v2 — intelligent loop)
The app itself becomes smart. Not just a place to enter data, but a system that:
- **Notices patterns** — "you always stall on meditation after weekends"
- **Adapts questions** — asks what's relevant today, not the same 6 fields every time
- **Computes momentum** — from actual behavior, not self-reported status
- **Proposes experiments** — "try morning meditation instead of evening for a week"
- **Tracks what works** — closes the loop on its own suggestions
- **Gets smarter over time** — learns from what you actually do, not what you plan to do

This could mean Claude API calls from the server, or smarter rule engines, or both. The right approach will emerge from actually using v1 and seeing what's missing. **Ship simple, use it, improve it.**

### Architecture
```
┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  PWA         │────▶│  Go API      │────▶│  SQLite  │
│  (phone)     │◀────│  (Fly.io)    │◀────│  (volume)│
└──────────────┘     └──────────────┘     └──────────┘

Claude Code = dev tool only (no tokens in deployed app)
```
- **Go + SQLite + Fly.io free tier** ($0/mo)
- **Session auth** (bcrypt + cookies), rate-limited
- **PWA** — installable on phone, works offline for static assets
- First user registers freely, subsequent users need INVITE_CODE env var

### Design principles
- **Don't make me think** — Interface explains itself.
- **Single source of truth** — CLAUDE.md for system design. Database for user data.
- **Flywheel > features** — Every addition must make the daily loop better.
- **Phone-first** — If it doesn't work on the phone, it doesn't work.
- **Ship simple, improve always** — A working v1 beats a perfect plan.
