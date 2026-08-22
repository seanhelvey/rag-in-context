# RAG in context: working notes

## What this is, and what it is not

**It is a refresher on modern AI, anchored on RAG, for a reader with a classical ML
background.** The value is in the bridges: representing things as vectors and comparing
with a cosine is decades old, and what changed is that the coordinates are learned. Search
is an old field. Combining rankers is ensembling. Say those connections out loud, because
they are the whole reason this document exists rather than another RAG tutorial.

**It was called "RAG from scratch" until 2026-08-22, and that framing was wrong.** It
turned building things by hand into the goal, so every editing session drifted toward more
low-level implementation, and the notebook grew to 6,200 words that Sean could not get
through. The instruction that matters:

> Write code by hand only where seeing the arithmetic makes an idea click. Import
> everything else without apology.

Importing is the default. Hand-writing needs a reason. That is the inverse of what this
file used to say. Under that rule BM25 stays hand-written, because the live IDF table is
the clearest moment in the notebook, and softmax and attention were cut, because building
them cost 950 words to explain something the notebook only needed to name.

**RAG is the anchor because it is in a lot of job descriptions** and it is the gap Sean
wants closed. Keep it central; the older-ideas framing serves it rather than the reverse.

## Commands

```bash
.venv/bin/jupyter lab rag.ipynb                    # read/run it
.venv/bin/jupytext --sync rag.md                   # after editing either file
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 rag.ipynb   # refresh committed outputs
```

Run everything from the repo root; all paths in the notebook are relative. A full execute
takes ~90s on CPU. The venv exists; fresh setup is `python -m venv .venv && .venv/bin/pip
install -r requirements.txt`.

## Source of truth

**`rag.md` and `rag.ipynb` are paired by jupytext.** Edit either one and `--sync` pushes
the change to the other. Prose is plain markdown and code is in ```python fences, so there
is no string-escaping to get wrong.

This replaced `build_notebook.py`, a script that held the entire notebook inside Python
`'''` strings. It banned hand-editing the notebook, required `\\n` for every literal
newline, broke in confusing ways when edited through a shell heredoc, and made every
change expensive enough that appending a paragraph was easier than restructuring. That is
a large part of how the notebook got long. Do not reintroduce a generator script.

Regenerating from `rag.md` clears the outputs, so re-execute afterwards.

## Constraints that are deliberate

- **No API keys, ever.** Local models only. It must run on a laptop with no account and
  nothing to leak on a shared screen. The generation step prints a prompt rather than
  sending it.
- **Must run offline** once the two models are cached. Verified with `HF_HUB_OFFLINE=1`.
- **Outputs stay committed.** It has to read correctly without being run.
- **No vector database, no framework** in the notebook itself. `E @ q` is the search, and
  the *In production* notes name what you would really use.

## Length

**Prose bloat is the standing failure mode, and it is additive.** Every individual fix
looks reasonable and they accumulate. The notebook went from 3,492 to 4,349 prose words in
a single session of "improvements" while Sean was asking for it to be *clearer*. It is now
~2,500. Treat 3,000 as a ceiling.

Before adding a paragraph, check whether it restates one already present, and prefer
cutting something else. Before adding a figure, the same. Keep sentences under about 30
words:

```bash
.venv/bin/python - <<'PY'
import re
src = open('rag.md').read()
for para in re.split(r'\n\n', re.sub(r'```.*?```', '', src, flags=re.S)):
    p = ' '.join(para.split())
    if p[:1] in '#-|*': continue
    for s in re.split(r'(?<=[.!?]) ', p):
        if len(s.split()) > 32: print(len(s.split()), s[:90])
PY
```

## Production notes go inline, not in a table

Sean asked for this twice. A `*In production:*` line sits directly under the piece it
refers to. A consolidated closing table was tried and removed: it duplicated the inline
notes and landed at the end where a tired reader had already stopped.

## Evaluation

`queries.json` holds 18 questions, each with a `marker`: the exact string that must appear
in a retrieved passage for the retrieval to count. Grade at passage level, not file level.
File-level labels scored 1.00 for all four methods and could not distinguish them.

When adding a query, check the marker actually appears in `corpus/` first. A marker that
matches nothing silently scores zero for every method.

Current numbers, quoted directly by the prose in section 5. **If these change, the
surrounding markdown has to change too**:

| method | recall@5 | MRR |
|---|---|---|
| by meaning | 0.78 | 0.69 |
| by keyword | 0.78 | 0.62 |
| both, fused | 0.89 | 0.65 |
| fused + rerank | 0.94 | 0.85 |

## Corpus

13 markdown files copied out of Sean's own public repos, flattened to `reponame__FILE.md`.
Chosen because he wrote all of it, so whether a retrieval is correct is checkable by
reading rather than guessing.

**Only add docs from repos already public on GitHub.** `django-fingerprint` was removed for
this reason. `howgood-apply` is excluded as an application to a specific company.

**The corpus files are dated snapshots, not live copies.** `seanhelvey__README.md` in
particular tracks a profile README that changes often. Sync it deliberately and rarely
rather than on every edit, and never sync without running `check.py` afterwards.

**Run `.venv/bin/python check.py` after touching `corpus/` or `queries.json`.** Ten of the
eighteen markers appear in exactly one file, so editing that string away makes the query
score zero for every method and drags the results table down without any error. The script
turns that silent failure into a loud one.

Syncing the profile README on 2026-08-22 moved the chunk count from 199 to 201 and left
every score identical, so a sync is cheap. It is the marker disappearing that is expensive.

Two properties are load-bearing: the two search methods must fail on *different* questions
(the entire argument for hybrid search), and `:5173` appearing in two separate projects
gives a worked example of a question no retriever can answer without metadata filtering. It
said "three repos" until 2026-08-22, which counted files. Check the data before restating a
count.

## Audience and voice

Written for a technical reader refreshing on this material, not a specialist. **Define a
term the first time it is used**, and never name one you are not going to explain.

**Watch terms that arrive inside figures.** Three separate bugs came from this: `5173`
appeared in a caption before Vite was explained, `E` was a labelled box before embeddings
were defined, and `dense + BM25` was in the pipeline diagram before section 3. Figures skip
the prose queue, so anything printed in one must be defined earlier than feels necessary.

**Register: explain, do not rule.** Avoid verdicts on tools nothing here tested, claims of
experience the notebook does not demonstrate, intensifiers doing the work of evidence
("genuinely", "precisely", "actually", "the whole"), telling the reader what to conclude
("Three things to notice"), and judging absent third parties ("the step people skip").

**No em dashes or en dashes anywhere.** Standing rule. Ordinary hyphens are fine. Use a
colon, comma, full stop or parentheses. Check with `grep -c "—\|–" rag.md README.md
figures.py`, which must be 0. This file is exempt only because the rule quotes the
characters it forbids.

**Frame positively.** Say what a thing *is* and *does*. Keep possessives to about one per
document.

**The honest frame is one line: a refresher, filling in gaps as I go.** Do not expand it
into a backstory, do not narrate gaps as setup for having closed them, and do not
effort-boast. The README says plainly that Claude helped build it; keep that.

Comments say *why*, not *what*. Every code cell should produce something visible. Honest
numbers over flattering ones: section 5 says out loud that 18 questions cannot referee 0.89
against 0.94.

## Figures

**Generated by matplotlib, never linked or embedded as image files.** That keeps the
offline rule and avoids licensing questions. Drawing code lives in `figures.py`; anything
that teaches retrieval stays inline in the notebook.

Palette: `GREEN "#2a7"`, `BLUE "#47c"`, `GREY "#999"`, `ORANGE "#e0a300"`. Hide top and
right spines. **Each colour means one thing across every figure:**

- **orange**: the one item to look at. The overlap in chunking, the reference arrow in the
  fan, the correct answer in rank movement. Strictest of the four; do not spend it on
  ordinary data.
- **green**: the primary data.
- **grey**: baseline or chance.
- **blue**: a second series or an intermediate step.

The five that survive, each showing a mechanism rather than decorating:

- **Pipeline** (before anything else): two rows of boxes, the indexing loop above and the
  per-question loop below, orange bracket over the retrieval steps. It exists because the
  notebook was built bottom-up and never showed the reader the destination.
- **Cosine fan** (section 2): one chunk as reference, four others at the angle the model
  produced, 90 degrees marked. Beside it, every pair of chunks against random 384-d
  directions, which is what says a cosine of 0.45 is high.
- **Rank movement** (section 4): candidates before and after reranking. Labels carry the
  chunk index because several chunks share a filename, and orange marks the correct
  answers so the reorder can be judged rather than admired.
- **Retrieval quality** (section 5): recall@5 and MRR for all four methods.
- **Chunk windows** (section 6): a 2300-char section with the 900-wide windows below it and
  the 150-char overlaps in orange.

Cut in the 2026-08-22 rewrite: a PCA scatter that showed no clustering, an 8x8 cosine grid
that drew a fabricated 1.00 because two markers resolved to the same chunk, and the softmax
and attention-head figures that supported the from-scratch build.

## History worth not repeating

- The intro was rewritten many times, each pass optimising for one property until the
  concrete motivation was gone. Two failure modes: too abstract ("a language model does not
  know what is in your files") and invented concrete ("you cannot remember which repo the
  answer is in", to which Sean's reply was "dude you would know which repo you were in").
  State the problem where it is actually true, and say plainly that this corpus is a test
  bed rather than a use case.
- Section 1 was chunking for a long time. It is the least motivating possible opener.
- Evaluation sat at section 5 with four sections of technique before any evidence.
- The `[:420]` slice on the sample chunk cut mid-word and looked like broken output, while
  hiding the best example in the notebook: a chunk cut at the 900 limit whose severed
  sentence survives whole in the next chunk because of the overlap.
