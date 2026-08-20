# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Use GitHub's private reporting: **Security → Report a vulnerability** on this
repository. That opens a private thread visible only to maintainers.

Include what you did, what happened, and what you expected. A proof of concept helps
but is not required. We aim to acknowledge within 72 hours.

We will not pursue legal action against anyone acting in good faith who reports a
problem privately and does not access, modify, or retain other people's data while
investigating.

## What this project promises its users

These are the guarantees a vulnerability report should measure us against:

1. **Contact details are never published.** Email is optional, and is stored solely
   to send notifications and account recovery. It must never appear in a rendered
   page, an API response, or a log line. Any code path that surfaces one user's
   address to another user is a critical bug.
2. **Exact locations are never persisted.** Coordinates are jittered before being
   written (`app/geo.py`). A report showing precise coordinates recoverable from the
   database, the API, or the logs is a high-severity issue.
3. **Uploads carry no metadata.** Images are re-encoded from raw pixels to strip EXIF
   GPS (`app/images.py`). A path that stores an original file unaltered is a
   high-severity issue.
4. **Messages are private to their recipient.** `/inbox` only ever returns rows where
   `recipient_id` is the session user. Any read path that widens this is critical.

## Deployment checklist

An instance is not safe to invite users to until all of these are true:

- [ ] `CS_SECRET_KEY` is set to a random 48-byte value. The default is a known string;
      leaving it means anyone can forge a session cookie for any account.
- [ ] `CS_DEBUG=false` — this is what makes session cookies `Secure` and hides
      tracebacks and the OpenAPI schema.
- [ ] The site is served over HTTPS only (FastAPI Cloud does this by default).
- [ ] `CS_SUPABASE_SERVICE_KEY`, `CS_RESEND_API_KEY`, and the other secrets in
      README's environment table are set and marked Secret in the FastAPI Cloud
      dashboard.
- [ ] Reports at `/listings/{id}/report` are actually being read by a human.

## Current known limitations

Stated plainly rather than left for someone to discover:

- **Posting and messaging require a verified email.** Signing up with an address you
  do not own works, but nothing that reaches another person or the public listings
  does, until the address is proven: `/verify-email/{token}` (sent at signup and
  resendable from a banner shown to any logged-in, unverified account) or
  `/reset-password/{token}` (which verifies as a side effect of proving inbox
  control). Browsing and searching stay open to everyone, verified or not.
- **There is no moderation queue.** Reports are written to a table and emailed to
  `CS_REPORT_EMAIL`, but a maintainer still has to read and act on each one by hand.
- **Beta feedback (`/feedback`) works the same way.** Written to a table and emailed
  to `CS_REPORT_EMAIL` (shared with abuse reports — one operator, one inbox), read
  and actioned by hand — no automated triage.
- **Comments and messages are not filtered** for abuse beyond length caps.
