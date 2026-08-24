"""Every invariant this repo has, in one command.

    python check.py           run everything (use this)
    python check.py --quiet   exit status only

Each check exists because something broke silently once. The expensive failures
here are never crashes: a marker edited out of the corpus, a results number that
moved and was updated in two files out of three, a section renumbered and a
cross-reference left dangling. Nothing errors, the notebook still runs, and the
document quietly starts lying.
"""
import json, pathlib, re, sys

PER_CELL, TOTAL_CODE, PROSE_CEILING, TABLE_WORDS, MAX_SENTENCE = 15, 160, 3000, 150, 32

md = pathlib.Path("rag.md").read_text()
body = re.sub(r"^---.*?^---", "", md, flags=re.S | re.M)
prose = re.sub(r"```.*?```", "", body, flags=re.S)
cells = re.findall(r"```python\n((?:.|\n)*?)```", body)
nb = json.loads(pathlib.Path("rag.ipynb").read_text())
fails = []


def out_text(cell):
    parts = []
    for o in cell.get("outputs", []):
        t = o.get("text", "")
        parts.append("".join(t) if isinstance(t, list) else t)
    return "".join(parts)


def check(name, problems, detail=""):
    print(f"{'FAIL' if problems else 'ok  '}  {name}{detail}")
    for line in problems:
        print(f"        {line}")
    fails.extend(problems)


# --- eval markers: ten of eighteen live in exactly one file ------------------
corpus = {p.name: p.read_text() for p in pathlib.Path("corpus").glob("*.md")}
queries = json.loads(pathlib.Path("queries.json").read_text())
missing = [f"marker {q['marker']!r} appears in no corpus file  ({q['q']})"
           for q in queries if not any(q["marker"] in t for t in corpus.values())]
check("markers resolve", missing, f"  ({len(queries)} queries, {len(corpus)} files)")

# --- the pipeline figure states the file count as a literal ------------------
fig = pathlib.Path("figures.py").read_text()
stated = re.search(r'"(\d+) markdown files"', fig)
check("figure file count",
      [] if not stated or int(stated.group(1)) == len(corpus)
      else [f"figures.py says {stated.group(1)}, corpus has {len(corpus)}"])

# --- results table agrees across the notebook, README and CLAUDE.md ----------
# This is the one that has gone wrong twice. Any parameter change moves all
# eight numbers, and they are quoted in three places plus the prose.
actual = {}
for cell in nb["cells"]:
    for line in out_text(cell).splitlines():
        m = re.match(r"\s*(by meaning|by keyword|both, fused|fused \+ rerank)\s+"
                     r"recall@5 ([\d.]+)\s+MRR ([\d.]+)", line)
        if m:
            actual[m.group(1)] = (m.group(2), m.group(3))

stale = []
if len(actual) < 4:
    stale.append(f"only found {len(actual)} of 4 result rows in notebook output; "
                 "re-execute before trusting the rest of this check")
else:
    for fname in ("README.md", "CLAUDE.md"):
        text = pathlib.Path(fname).read_text()
        for method, (r, mrr) in actual.items():
            row = re.search(rf"^\| {re.escape(method)}[^|]*\| \*?\*?([\d.]+)\*?\*?"
                            rf" \| \*?\*?([\d.]+)\*?\*? \|", text, flags=re.M)
            if not row:
                stale.append(f"{fname}: no results row for {method!r}")
            elif (row.group(1), row.group(2)) != (r, mrr):
                stale.append(f"{fname}: {method} says {row.group(1)}/{row.group(2)}, "
                             f"notebook printed {r}/{mrr}")
    # Prose quotes individual scores. Harvest every recall/MRR the notebook printed,
    # including the damp sweep, so a real number is never flagged as invented.
    live = {v for pair in actual.values() for v in pair}
    for cell in nb["cells"]:
        live |= set(re.findall(r"(?:recall@5|MRR) ([\d.]+)", out_text(cell)))
    for quoted in set(re.findall(r"(?<![-\d.])0\.\d\d\b", prose)):
        if quoted not in live and quoted not in {"0.42", "0.12", "0.06"}:
            stale.append(f"rag.md prose quotes {quoted}, which no method scored")
check("results consistent", stale, f"  ({len(actual)} rows found)")

# --- section cross-references ------------------------------------------------
heads = {int(n) for n in re.findall(r"^## (\d)\.", body, flags=re.M)}
refs = {int(n) for n in re.findall(r"[Ss]ection (\d)", body)}
check("section refs", [f"section {n} referenced, no such heading" for n in sorted(refs - heads)])

# --- the pipeline figure carries its own section labels ----------------------
# These are hand-written strings inside figures.py, so a renumbering leaves them
# stale and the reader's first map is wrong before any content lands.
fig_refs = {int(n) for n in re.findall(r'"section (\d)"', fig)}
table_refs = {int(n) for n in re.findall(r"\| section (\d) \|", body)}
bad_fig = [f"figures.py says section {n}, no such heading" for n in sorted(fig_refs - heads)]
if table_refs and not fig_refs <= table_refs | {5, 7}:
    bad_fig.append(f"figures.py labels {sorted(fig_refs)} but the roadmap table lists "
                   f"{sorted(table_refs)}; one of them has drifted")
check("figure section labels", bad_fig, f"  ({sorted(fig_refs)})")

# --- every locator names a box that exists -----------------------------------
labels = re.findall(r'\("([a-z +]+)", \d, [\d.]+\)', fig)
called = re.findall(r'"([a-z +]+)"', "".join(re.findall(r"figures\.locate\(([^)]*)\)", body)))
check("locator steps", [f"locate({c!r}) matches no box in the strip" for c in called
                        if c not in labels], f"  ({len(called)} lit, {len(labels)} boxes)")

# --- code budget and committed stderr ----------------------------------------
code = []
total = 0
for i, block in enumerate(cells, 1):
    lines = [l for l in block.split("\n") if l.strip()]
    # A lone figures.locate() call is navigation, not implementation, the same way
    # a table of contents is not prose. One per section, and no budget pressure.
    if len(lines) == 1 and lines[0].startswith("figures.locate("):
        continue
    n = len(lines)
    total += n
    if n > PER_CELL:
        code.append(f"cell {i}: {n} lines, over the {PER_CELL}-line budget")
if total > TOTAL_CODE:
    code.append(f"{total} code lines, over the {TOTAL_CODE}-line budget")
for i, cell in enumerate(nb["cells"], 1):
    for o in cell.get("outputs", []):
        if o.get("name") == "stderr" or o.get("output_type") == "error":
            code.append(f"cell {i}: {o.get('name', 'error')} committed into the notebook")
check("code budget", code, f"  ({len(cells)} cells, {total} lines)")

# --- every code cell has committed output ------------------------------------
blank = [f"cell {i} has no committed output" for i, c in enumerate(nb["cells"], 1)
         if c["cell_type"] == "code" and not c.get("outputs")]
check("outputs committed", blank)

# --- prose length and sentence length ----------------------------------------
# Table rows are navigation, not explanation. Counting them against the prose
# ceiling puts a table of contents in competition with the text it indexes, which
# is backwards, so they get their own smaller budget.
rows = [l for l in prose.splitlines() if l.strip().startswith("|")]
table_words = sum(len(l.split()) for l in rows)
words = len(prose.split()) - table_words
long_s = []
if table_words > TABLE_WORDS:
    long_s.append(f"{table_words} words in tables, over the {TABLE_WORDS} budget")
for para in re.split(r"\n\n", prose):
    flat = " ".join(para.split())
    if flat[:1] in "#-|*":
        continue
    for sent in re.split(r"(?<=[.!?]) ", flat):
        if len(sent.split()) > MAX_SENTENCE:
            long_s.append(f"{len(sent.split())} words: {sent[:70]}...")
if words > PROSE_CEILING:
    long_s.insert(0, f"{words} prose words, over the {PROSE_CEILING} ceiling")
check("prose", long_s, f"  ({words} words, {table_words} in tables)")

# --- the standing dash rule ---------------------------------------------------
BANNED = ("\u2014", "\u2013")   # spelled by codepoint so this file does not trip its own rule
dashes = [f"{f}: {n} dash(es)" for f in ("rag.md", "README.md", "figures.py", "check.py")
          for n in [sum(pathlib.Path(f).read_text().count(d) for d in BANNED)] if n]
check("no em or en dashes", dashes)

print()
if fails:
    print(f"{len(fails)} problem(s)")
    sys.exit(1)
print("all clear")
