"""Verify every eval marker still appears in the corpus, and that the pipeline
figure still states the right file count.

Ten of the eighteen markers live in exactly one file, so editing that string
away makes the query score zero for every method and quietly drags the whole
results table down. Run this after any change to corpus/ or queries.json.
"""
import json, pathlib, re, sys

corpus = {p.name: p.read_text() for p in pathlib.Path("corpus").glob("*.md")}
queries = json.loads(pathlib.Path("queries.json").read_text())

missing = []
for q in queries:
    hits = [f for f, text in corpus.items() if q["marker"] in text]
    if not hits:
        missing.append(q)
    print(f"{len(hits)} file(s)  {q['marker']!r}")

# The pipeline figure states the file count as a literal, drawn before any
# chunking happens, so nothing else can catch it going stale.
fig = pathlib.Path("figures.py").read_text()
stated = re.search(r'"(\d+) markdown files"', fig)
if stated and int(stated.group(1)) != len(corpus):
    missing.append({"marker": f"figures.py says {stated.group(1)} markdown files, corpus has {len(corpus)}",
                    "q": "pipeline figure caption"})

print(f"\n{len(queries)} queries, {len(corpus)} files")
if missing:
    for q in missing:
        print(f"MISSING  {q['marker']!r}  ({q['q']})")
    sys.exit(1)
print("every marker resolves")
