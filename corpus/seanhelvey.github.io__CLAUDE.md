# seanhelvey.com

Jekyll site on GitHub Pages, custom domain via `CNAME`. Personal site and blog for Sean Helvey.

## Writing voice

Site copy should sound like a person, not a model.

- No em dashes or en dashes. Ordinary hyphens in compound words are fine and wanted.
- Avoid AI-tell phrasing: overpolished tricolons, "the next chapter involves", and similar.
- Do not repeat a distinctive word twice in close proximity on the same page.

## Content that is intentional, do not "fix"

- **The prose on `index.html` and `about.html` is deliberate.** It is written to read honest and organic, not optimized for recruiters. Do not add metrics, tighten it into resume bullets, or otherwise make it read like a candidate profile.
- **AppFolio (2019-2020) is omitted from the site on purpose.** It stays on the resume PDF only. Not a gap, do not flag it.
- **`resume.html` is deliberately `noindex, nofollow`** with `sitemap: false`, and redirects to `assets/HelveyResume2026.pdf`. The resume is not meant to be indexed. `robots.txt` disallows both paths to match.

## LLM and search visibility

Set up in August 2026, based on Evil Martians' measured LLM traffic study
(<https://evilmartians.com/chronicles/which-ai-actually-reads-your-site-two-months-of-llm-traffic-measured>),
not on their `llms-visibility` skill, which is partly stale relative to their own data.

What is in place:

- `robots.txt` with `Content-Signal: search=yes, ai-input=yes, ai-train=yes`, AI crawlers named explicitly, resume paths disallowed, sitemap declared.
- `jekyll-sitemap` plugin. Both PDFs excluded via `defaults` in `_config.yml`.
- Per-page `<title>` and `<meta name="description">` in `_includes/head.html`, plus Open Graph and Twitter cards. Pages may set `description` and `seo_title` in front matter; posts fall back to `subtitle` then `excerpt`.
- `_includes/structured-data.html`: JSON-LD `Person` on every page, `BlogPosting` on posts linked to the same `@id`.

### Deliberately not done, these are decisions and not oversights

- **`Accept: text/markdown` content negotiation.** The single most effective technique in the study, and impossible on GitHub Pages, which gives no control over response headers. Would require putting Cloudflare in front of the site. Judged not worth the infrastructure.
- **`llms.txt` and `llms-full.txt`.** The study found only ~37 of 660 fetches came from named AI assistants, the rest generic bots.
- **`.md` route mirrors and `Link rel="alternate"` hints.** Not independently valuable per the study, and the discovery hints got zero attributable fetches over two months.
- **Hidden pointer text for LLMs.** Recommended by the skill, contradicted by the same company's later data.

### JSON-LD, a known disagreement

The `llms-visibility` skill lists JSON-LD as an anti-pattern, on the grounds that no LLM tool parses it. It is kept here anyway, for a different purpose: Google and Bing entity resolution, and the `sameAs` links tying this domain to the GitHub, LinkedIn, Bluesky, Mastodon, and Goodreads accounts as one person. That serves the recruiter-searches-your-name path, which matters more here than direct agent fetching. Remove it only on purpose, not because a skill flags it.

## Local development

```
bundle exec jekyll build     # or: bundle exec jekyll serve
```

`_site/` is generated and gitignored. `sitemap.xml` and `robots.txt` will not appear in the repo source; the sitemap is generated at build time and GitHub Pages runs that build on push.

## Conventions

- Sean handles his own commits and pushes. Make edits and stop.
- Posts live in `_posts/`, front matter uses `permalink: /:title/`.
