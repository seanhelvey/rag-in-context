# RAG from scratch: working notes

One notebook that builds retrieval-augmented generation up from numpy, on a corpus of
my own repo documentation. Built as interview material, so the priority is *legible*
over *complete*: every cell should be explainable out loud in under a minute.

## Commands

```bash
.venv/bin/jupyter lab rag.ipynb                    # read/run it
.venv/bin/python build_notebook.py                 # regenerate rag.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 rag.ipynb   # run it and save outputs
```

All paths in the notebook are relative, so **run it from the repo root**. The first
data cell asserts `corpus/` is present and says so rather than failing later with an
empty-corpus `NaN` error. A full execute takes ~90s on CPU.

The venv already exists. Fresh setup is `python -m venv .venv && .venv/bin/pip install
-r requirements.txt`.

## The one rule that matters

**Never hand-edit `rag.ipynb`.** It is generated. Edit `build_notebook.py`, re-run it,
then re-execute the notebook to refresh the outputs. Editing the JSON directly means
the next `build_notebook.py` run silently throws the change away.

Cells are defined by `md(...)` and `code(...)` calls in order, each holding a `'''`
string. Cell sources therefore must not contain `'''`, and a literal `\n` inside a
cell needs to be written `\\n` so it survives into the notebook.

**That escaping breaks when editing through a shell heredoc**, because the escape gets
eaten twice and a real newline lands inside a cell's string literal, which fails only at
execute time with `SyntaxError: unterminated string literal`. Either edit
`build_notebook.py` directly rather than through a script, or avoid needing `\n` in the
cell at all. Cheap check before spending 90 seconds on an execute:

```bash
.venv/bin/python build_notebook.py >/dev/null && .venv/bin/python -c "
import json
nb = json.load(open('rag.ipynb'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        compile(''.join(c['source']), f'cell{i}', 'exec')
print('all code cells compile')"
```

## Constraints that are deliberate

- **No API keys, ever.** Local models only. The notebook must run on a laptop with no
  account, no billing, and nothing to leak on a shared screen. The generation step
  assembles a prompt and prints it rather than sending it. See section 6.
- **No vector database, no framework.** `E @ q` is the whole search. The point is that
  a vector DB is that line plus persistence, filtering, and an ANN index. Adding
  LangChain or Pinecone here would delete the thing being taught.
- **Must run offline** once the two models are cached. Verified with
  `HF_HUB_OFFLINE=1`. Interview wifi is not to be trusted.
- **Outputs stay committed.** The notebook has to read correctly without being run.

### Where the build-it-yourself line sits

"From scratch" was always a half-truth: the embedding model and the cross-encoder are
`sentence-transformers` calls, because writing a transformer teaches nothing about
retrieval. Roughly 120 lines are hand-written, in functions of 4 to 12 lines.

The test for which side a piece falls on: **does writing it yourself expose a decision you
would otherwise make blindly?**

- *Build it.* `E @ q` (vector search is one matmul), chunking (where you cut, and every
  library hides it behind `chunk_size=`), `rrf` (why you cannot add a cosine to a log
  sum), `softmax`, `evaluate` (the step everyone skips).
- *Import it.* Anything neural. BM25 is the genuine coin-flip, kept hand-written only
  because the live IDF table is one of the clearest moments in the notebook.

A framework version of this notebook would be about 30 lines and would teach nothing about
mechanism, and RAGAS-style evaluation would need an LLM, breaking the no-API-keys rule.

The closing **"What you would actually use"** table maps each hand-written piece to its
production counterpart. Keep it current; it is what makes the from-scratch framing honest
and gives the reader a path out of the notebook.

## Evaluation

`queries.json` holds 18 questions, each with a `marker`: the exact string that must
appear in a retrieved passage for the retrieval to count. Grade at passage level, not
file level. File-level labels scored 1.00 for all four methods and could not
distinguish them at all. That is why the marker scheme exists.

When adding a query, check the marker actually appears in `corpus/` first. A marker
that matches nothing silently scores zero for every method and drags the whole
table down.

Current numbers, which the prose in section 5 quotes directly. **If these change,
the surrounding markdown has to change too**:

| method | recall@5 | MRR |
|---|---|---|
| dense only | 0.78 | 0.69 |
| BM25 only | 0.78 | 0.62 |
| hybrid (RRF) | 0.89 | 0.65 |
| hybrid + rerank | 0.94 | 0.85 |

## Corpus

`corpus/` is 13 markdown files copied out of my own repos, flattened to
`reponame__FILE.md`. Chosen because I wrote all of it, so whether a retrieval is
correct is checkable by reading rather than by guessing.

**Only add docs from repos that are already public on GitHub.** `django-fingerprint`
was removed for this reason: its remote (`seanhelvey/FunDjango`) no longer exists, so
its README had never been published, and it listed third-party sites' admin paths.
`howgood-apply` is excluded too, being an application to a specific company.

Two properties are load-bearing and should survive any corpus change: dense and BM25
must fail on *different* questions (that is the entire argument for hybrid search),
and `:5173` appearing in three separate repos gives a worked example of a question no
retriever can answer without metadata filtering.

## Audience

Written for a technical reader refreshing on RAG, not a specialist, and explicitly
meant to be usable by someone else in the same position. Practical consequence:
**define a term the first time it is used.** PCA, IDF, MRR, bi-encoder, cross-encoder
and the rest all get a sentence in place rather than being assumed. Section 7 is
labelled as vocabulary so it reads as orientation, not a reading list.

**The assumed floor is Python and numpy, and nothing more.** Concretely: `@` is a dot
product, and array slicing is familiar. Everything else gets built. No transformer
background, no softmax on sight.

**Assume nothing beyond that floor.** My own background is classical pre-neural NLP plus
general ML fundamentals, and I do not have BM25, precision/recall, softmax or attention
loaded. Do not infer otherwise from the fact that a topic appeared in a course once. If
it is used, it gets built or re-derived.

**The honest frame is one line: "a refresher and an exploration, filling in gaps as I
go."** That is the whole of it. Do not expand it into a backstory.

Specifically, no narrating my own gaps as setup for having closed them ("my NLP
background predates the neural turn, and I had never worked out what attention does").
that is flexing wearing modesty, and it is not more honest than the thing it replaced. No
effort-boasting either: how long the labelling took, how few lines BM25 needed, how the
pieces were "built from scratch rather than imported." Also drop comparisons that flatter
by contrast ("unlike tutorials that search Wikipedia", "a large reason RAG demos stop
before evaluation").

The line to walk: I made this, and it does not need to be sold as more than a refresher.
Say what a thing *is*, not what it cost or what it beats. Do not convert the frame into
an assumption about the reader either. Headings like "If you learned this before the
neural era" presume a background the reader may not have.

**Layer it, with escape hatches.** Anything re-deriving a fundamental is headed
`### Refresher: <topic>` and opens with an italic *Skip if ...* line. The main line runs
straight through them, so both a newcomer and an experienced reader get a decent read.
The intro explains the convention once. Current refresher: precision/recall.

**Never name a term you are not going to explain.** A bare acronym is not orientation, it
is noise, and stringing several together ("documents as vectors, cosine similarity,
TF-IDF's idea that...") reads as padding no matter how true each one is. Either explain
it where it lands or cut it. Removed for this reason: a front-matter block previewing
TF-IDF (IDF is taught properly in section 3 with a live table, so the preview was worse
than nothing), `HNSW, IVF` as a parenthetical, and `BERT` as a description of the
encoder. Section 7 is the one exception, because it is explicitly labelled as vocabulary
you are not expected to implement.

Watch for jargon that sneaks in through *prose* rather than code. `argmax` appeared in
section 0 as an offhand contrast, defined nowhere and used in no cell, which is the
worst case: it costs the reader something and buys nothing. If a term is not doing
work, cut it; if it is, define it where it lands.

Section 0 is the section that breaks this if anyone is careless with it, because it is
the one place transformer vocabulary appears. It used to define `softmax`, `K`, `V`,
temperature scaling and `softmax(QKᵀ)V` in a single cell without ever saying what a
query, key or value *is*, which read fine to someone who already knew and was a wall
to everyone else. It is now three code cells that go scoring → softmax → query/key/
value, each with visible output, each step usable before the next arrives. Keep that
shape. If a step needs a term the reader has not met, the term comes first.

**Assume the reader does not know what attention is** beyond having heard the paper was
important. So the explanation runs build-then-name, never name-then-build: the section
constructs the scoring machinery first and only afterwards says "that was an attention
head", followed by a plain-English paragraph on what a transformer does with it. Do not
reorder this into "attention is X, and RAG is like attention", which explains the
unfamiliar with the more unfamiliar. The attention material is framed as a bonus
precisely so that a reader who gets nothing from it can still use sections 1-7.

The notebook closes with a three-line "short version" rather than a numbered recap. The
old five-things list restated the whole notebook a third time and was cut; do not
reintroduce it.

## The framing

The intro has been rewritten many times, and each rewrite optimised for one property
(shorter, warmer, fewer dashes) until the concrete motivation was gone. Sean read the
result and asked "what is the problem again, we are trying to search repo readmes?", which
is the signal that it stopped motivating anything.

Two failure modes, both hit in the same session:

1. **Too abstract.** "A language model does not know what is in your files" is true of any
   files and gives the reader no reason to care.
2. **Invented concrete.** Reaching for a story and making one up: "you cannot remember
   which repo the answer is in." Sean's reply was "dude you would know which repo you were
   in." A fake scenario is worse than an abstract one, because it is checkable and wrong.

The honest framing states the problem where it is actually true (a wiki, a support
archive, a codebase nobody has read end to end), keeps the `grep "wipe the database"`
versus `make reset` example because guessing the author's words fails at any size, and
then says plainly that **this corpus is a test bed, not a use case**: 13 files are far too
small to need retrieval. Readers notice that anyway, so the notebook should say it first.

## Scope

Sean's own read (2026-08-21): *"I haven't even made it through more than half of this
thing."* Length is the failure mode that matters here, not incompleteness. A section that
is merely interesting is not worth the reader's remaining attention.

**Cut in 2026-08:** the chunk-size sweep (a whole section for a null result the notebook
itself called "one question wide", and the slowest cell in the file), the six-item
vocabulary list, and the five-things recap. The sweep's one real lesson survives as the
regression demo in section 5, which is a better use of it anyway.

**Evals stay basic.** One clear worked example beats a metrics suite. Section 5 runs:
label a question with a marker, check one question by hand, count over all 18, look at
the failures, then drop chunk size to 200 and watch recall fall. That last cell is the
point of the section. Resist adding latency benchmarks, context precision, or a
faithfulness harness; they are all defensible and they all make the section longer than
its lesson.

## Length

**Prose bloat is the standing risk**, because every fix here is additive and they
accumulate. The front matter reached 776 words before section 0 at one point; it is now
~500 and should not creep back. Whole-notebook prose is ~6,000 words.

Before adding a paragraph, check whether it restates one already present. The usual
failure is three consecutive paragraphs circling the same point in different words. Keep
sentences under about 30 words; the useful check is:

```bash
.venv/bin/python - <<'PY'
import re
src = open('build_notebook.py').read()
for b in re.findall(r"md\('''(.*?)'''\)", src, re.S):
    for p in b.strip().split('\n\n'):
        p = ' '.join(p.split())
        if p[:1] in '#-|': continue
        for s in re.split(r'(?<=[.!?]) ', p):
            if len(s.split()) > 32: print(len(s.split()), s[:90])
PY
```

## Diagrams

Figures are **generated by matplotlib**, never linked or embedded as image files. That
keeps the offline rule intact, keeps them regenerable, and avoids any licensing question
about borrowed images.

**Drawing code lives in `figures.py`, not in the notebook.** Plotting was 39% of all code
lines (174 of 441) and made the cells hard to follow for no teaching benefit. Each figure
is now a one-line call like `figures.softmax_steps(scores, labels, softmax)`. Note that
collapsed-cell metadata would not have worked: GitHub's renderer ignores it, and GitHub is
how most people will read this.

The line to hold: **anything that teaches retrieval stays inline; matplotlib incantations
do not.** The chunking and BM25 cells are still the longest in the notebook, and that is
correct, because they are the subject. Palette lives at the top of `figures.py`:
`GREEN "#2a7"`, `BLUE "#47c"`, `GREY "#999"`, `ORANGE "#e0a300"`. Hide top and right
spines.

A figure has to show the mechanism, not decorate the page. The three explanatory ones:

- **Softmax** (section 0): raw score, after `exp`, after dividing, as grouped bars, plus
  the same scores at x1/x4/x20. Makes "exp stretches the gaps" visible, and calls out
  that a score of 0 becomes 0.37 rather than 0.
- **Attention** (section 0): the head laid out as five labelled columns, key to score to
  weight to value to contribution, with the sum underneath. It is the arithmetic, drawn.
- **Chunking** (section 1): a 2300-char section with the 900-wide windows below it and
  the 150-char overlaps in orange.

**Watch cell ordering when adding one.** A figure cell placed in the markdown block that
*introduces* a function lands before the `def`, and fails with `NameError` only at
execute time. Check with the ordering script under Length, or just confirm the defining
cell comes first.

## Punctuation and framing

**No em dashes or en dashes anywhere.** This is a standing rule of mine, written down in
`corpus/seanhelvey.github.io__CLAUDE.md`, and it applies to this repo too. Ordinary
hyphens in compound words are fine and wanted. Use a colon, a comma, a full stop or
parentheses instead. Check with:

```bash
grep -c "—\|–" build_notebook.py README.md          # must be 0 for both
```

This file is exempt only because the rule above has to quote the characters it forbids.

Bullet definitions take the form `- **Term**: definition`, with a colon rather than a
dash.

**Frame positively.** Stacked negations read cold, and this notebook drifted into them:
"No API keys, no vector database, no framework", "Neither wins", "Nothing is near zero",
"not enough to implement them". Say what a thing *is* and what it *does*. "Everything is
written out here rather than pulled from a framework" carries the same fact and invites
rather than repels. Keep the genuinely negative statements that are load-bearing, like
"None of it is evidence" at the top of section 5, and drop the rest.

**Warmth over coldness.** The intro once ran "on my own docs", "from my own repos", "the
corpus is mine" in three consecutive sentences, which reads as self-absorbed. Prefer
addressing the reader, explain a constraint rather than asserting it, and keep
possessives to one per document where they are actually doing work.

## Style

Comments say *why*, not *what*. The reader is an interviewer deciding whether I
understand retrieval, so explain the reasoning behind a choice rather than narrating
the syntax. The exception is a numpy idiom that hides something: `np.argsort(-scores)`
gets a comment because it returns indices rather than values and negates to sort
descending, and neither is visible from reading it. Honest numbers over flattering
ones: section 5 says out loud that 18 questions cannot referee 0.89 against 0.94, and
that caveat is worth more than the score it qualifies.

**Register: explain, don't rule.** State what the code does and what the numbers show,
and stop there. Specifically avoid:

- verdicts on tools nothing here tested ("X would be the wrong call")
- claims of experience the notebook does not demonstrate ("disappoints in production")
- intensifiers doing the work of evidence: "the whole X", "genuinely", "precisely",
  "actually", "the only reason"
- telling the reader what to conclude ("Two things to notice", "Three things to read
  off that chart") where showing it would do
- judging absent third parties ("the step people skip")

First person is for things that are literally true: I wrote the corpus, I labelled the
queries, the file-level labels were tried first and scored 1.00. Don't invent process
anecdotes to sound humble; dropping the claim is better than dressing it up.

Every code cell should produce something visible. Cells 6-8 used to be three silent
function-definition cells in a row, which reads as broken; each now ends with a small
demo (top-3 dense hits, an IDF table, a toy RRF fusion) that doubles as the
explanation.
