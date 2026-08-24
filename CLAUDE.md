# RAG in context: working notes

## What this is, and what it is not

**It is a simple, worked example of RAG for a reader with a classical ML background.**
Simple is a requirement, not a description: if a section cannot be followed on a first
read, it has failed, whatever else it demonstrates. The value on top of that is in the
bridges: representing things as vectors and comparing with a cosine is
decades old, and what changed is that the coordinates are learned. Search is an old field.
Combining rankers is ensembling. Say those connections out loud, because they are the
reason this exists rather than another RAG tutorial.

**It was called "RAG from scratch" until 2026-08-22, and that framing was wrong.** It
turned building things by hand into the goal, so every editing session drifted toward more
low-level implementation, and the notebook grew to 6,200 words that Sean could not get
through. The rename fixed the title. It did not fix the code, which is the subject of the
next section.

**RAG is the anchor because it is in a lot of job descriptions** and it is the gap Sean
wants closed. Keep it central; the older-ideas framing serves it rather than the reverse.

## Use the library, then break it down

**The rule that decides every code cell:**

> Call the standard library to do the work. Break the idea down by inspecting what it
> produced. Hand-write only the one or two lines that *are* the idea.

Three moves, and the middle one is where the teaching happens. Reimplementing a splitter
does not explain chunking; printing the seam between chunk 7 and chunk 8 and watching the
severed sentence reappear does. The breakdown comes from real output, not from source code
the reader has to compile in their head.

Chunking, done all three ways, is the clearest case:

```python
# 1. the library does the work
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
chunks = splitter.split_text(doc)

# 2. break it down by looking at what it did
print(splitter._separators)        # ['\n\n', '\n', ' ', ''] - tried in order, coarse first
print(chunks[7][-160:])            # cut short of the 400 limit, mid-sentence
print(chunks[8][:160])             # the same sentence, whole again, thanks to the overlap
```

Six lines, and the reader sees the algorithm (the separator list *is* "recursive character"),
the parameter, and the consequence. The version this replaced was 37 lines of `re.split` and
`start += max_chars - overlap`, and it showed none of the three.

**The test for keeping code.** Name the specific output a reader would not believe without
seeing it. If you cannot name one, call the library and inspect instead. The IDF table
passes: `the` scores 0.48 and `5433` scores 4.39, and you need real counts to trust that. A
regex that splits on headings fails, because nobody doubts text can be cut into pieces.

**Prefer libraries a reader will meet again.** Well established over clever, familiar over
minimal. Hugging Face's own Advanced RAG cookbook, the reference teaching notebook for this
material, imports `RecursiveCharacterTextSplitter` and hand-writes nothing. A niche package
with fewer dependencies teaches nothing transferable, so do not trade recognition for
install size.

**Take the component, leave the platform.** That same cookbook wraps everything in FAISS and
LangChain vector stores, and copying that would delete the best thing here. `E @ q` being the
entire search is a lesson: it shows there is nothing magic under a vector database.

| step | how | broken down by |
|---|---|---|
| chunking | `RecursiveCharacterTextSplitter` | the seam between two chunks, overlap visible |
| embeddings | `SentenceTransformer` | `E.shape`, and rows having length 1 |
| dense search | `E @ q`, one line | the line is short enough to be its own explanation |
| BM25 scoring | `rank_bm25` | the IDF table, built by hand in ~4 lines |
| fusion | by hand, ~6 lines | short, and it is the ensembling bridge to classical ML |
| reranking | `CrossEncoder` | the rank-movement figure, before against after |
| recall@5, MRR | by hand, short | the definitions are the point of section 5 |
| vector store | none, deliberately | see above |

**Any block that does not run must say so on its first line.** A ```` ```python ```` fence
becomes a real cell that jupytext executes; a fence with no language stays markdown and never
runs. The notebook has one of the latter, the schematic pipeline in *The problem*, and it
opens with `# Pseudocode. No library spells it exactly like this, and nothing here runs it.`
Sean asked for this after `build_index("corpus/")` read as a real API. If an illustration
needs a fence, label it or use a real library.

**Code budget: 15 non-blank lines per cell, 160 total.** The mirror of the 3,000-word prose
ceiling, and enforced the same way. `.venv/bin/python check.py --code` reports per-cell
overages and the total, and fails on any stderr left in a committed output.

The per-cell limit is the one that does the work. The total was set at 160 because that is
where the notebook landed once every cell met the per-cell rule, so its job is to catch the
next drift rather than to be shaved against. Raising it needs the same argument as adding a
figure.

## One library name per place it is used

**Sean raised this five times before it was actually fixed, so do not undo it.** The
complaint each time was the same: the page jumps from LlamaIndex, to a list of unrelated
libraries, to LangChain, before section 1.

The cause was the roadmap table's third column, which named four libraries at the same
moment as the LlamaIndex example. Two frameworks and four packages inside twenty lines, none
of them yet explained. The table now maps jobs to sections and nothing else, and each
library is introduced once, in the section that imports it, with a sentence on what it does.

**Mixing the two frameworks is fine, and the fix was saying why, not picking one.** Sean
landed on this after reading that they occupy different layers and that production teams
routinely combine them, running a LlamaIndex retriever as a tool inside a LangChain loop. The
notebook states the division where the first framework appears: LlamaIndex leads on ingestion
and retrieval, LangChain on chaining and agents, so the opening example is one and the
splitter is the other. Keep that sentence. Without it the mix reads as carelessness.

Reading order is now: LlamaIndex once at the top as the call you would make instead, then
`langchain-text-splitters` in section 2, `sentence-transformers` in 3, `rank_bm25` in 4,
`CrossEncoder` in 6, and both framework names once more at the very end. Check that order
with a grep after any edit to the opening.

**Every library gets the same introduction, in prose, before its import.** The concept first,
then one clause naming the package, then the code. Giving the splitter careful provenance
while `sentence-transformers` and `rank_bm25` only appeared inside import lines made those
two read as arriving from nowhere, and the contrast made it worse rather than better. If one
library is worth placing, they all are.

**Unifying on one framework was considered and rejected on evidence.** LlamaIndex's own
`SentenceSplitter` would remove the second name, but `llama-index-core` pulls 30 packages
including SQLAlchemy, aiohttp, nltk and tiktoken, which breaks both the install-in-seconds
and the offline rules. `langchain-text-splitters` pulls 18 pure-Python packages and the
`langchain` framework is never installed. LangChain has no short RAG form to show at the
top: its current tutorial is agent-based, which is why the opening example is LlamaIndex.

## Keeping the reader located

**The pipeline diagram is the map, and it scrolls out of sight after one screen.** Sean read
to section 7 and said "wtf just happened", then to section 8 and said "wait, we are only now
getting to G". Both were navigation failures rather than content failures.

Two fixes, on 2026-08-23. A full eight-row outline table was tried first and rejected: he
said it did not add much, and it cost 119 words competing with the text it indexed.

1. **A locator strip under every heading from 2 onward.** `figures.locate(step)` reprints the
   spine of the pipeline diagram, six boxes, with the current one lit in orange. Sections 5
   and 7 are not stages, so a lit box would be a lie. **Three marks, one per kind of
   section**, and the distinction is load-bearing:

   | mark | means | used by |
   |---|---|---|
   | lit orange box | this stage moves data | 2, 3, 4, 6, 8 |
   | dotted bracket, `spans=(lo, hi)` | the boxes these choices act on | 7, over chunk to search |
   | dotted return loop, `loop=True` | score, change, score again | 5, over search to rerank |

   The loop was Sean's idea, phrased as "something like a recycling symbol". Evaluation is a
   cycle run over the stages it scores, not a step between them, and a bracket read as static.
   **The loop must span the same boxes the big diagram brackets**, search through rerank. It
   did not at first, and the contradiction between the two pictures was the first thing
   noticed. Section 1 got a strip briefly and it was removed: that section is about where RAG
   sits in ML, so borrowing the pipeline's marks meant nothing.

   **The strip keeps the diagram's box names, its two-phase split and its colours.** Orange is
   the one thing to look at, which on the big diagram is the retrieval bracket and on a strip
   is whatever that section is about. Green dashed is the vectors crossing from indexing into
   querying, the same line the big diagram draws. The strip briefly used green for the eval
   marks, so both colours meant two things at once and Sean asked what they were: a direct
   breach of the one-colour-one-meaning rule below. The green `vectors` arrow also answers a
   question the flat strip raised, whether `embed` and `keyword + meaning` were the same
   thing. They are joined by an artifact, not by sequence, and the picture now says so. It briefly used one flat row and a shortened
   `search` label, which flattened away the diagram's main claim, that indexing runs once and
   querying runs per question, and made the reader translate names. What the strip does drop
   is the boxes carrying no section number: corpus, vectors, question, answer. Divergence
   from the big diagram needs a reason, and "it did not fit" only covers those four.

   **A `locate()` step naming no box lights nothing and still renders**, so the strip looks
   deliberate while saying nothing. That happened when the box was renamed from `search` to
   `keyword + meaning` and the call was not, leaving section 4 blank for two rounds. There is
   now an assert in `locate()` and a `locator steps` check in `check.py`. Renaming a box means
   renaming its call.

   One line per section, excluded from the code budget the same way table rows are excluded
   from the prose ceiling: navigation should not compete with the thing it indexes.
2. **Every section opens by naming what the previous one produced.** Not a topic sentence, a
   handoff. The chain reads: first box on the top row, then 324 passages exist, then `E` is
   built, then none of it is evidence yet, then the eval said they miss different questions,
   then the stack is finished at 0.94, then the G at last. Four of these were cold openers
   that simply announced a subject, and section 7 was the worst.

**The R/G ratio is stated in the opening**, because generation arriving at section 8 reads as
a surprise otherwise: sections 2 to 7 are the R, section 8 alone is the G.

## Measure before you improve

**Evaluation is section 5, before fusion and reranking, and it must stay there.** This is
the structural decision the repo is built around, and it was moved on 2026-08-23 after
research into why RAG tutorials fail. An audit of 30 production RAG systems found the
recurring failure modes were "none about chunking strategy or vector store choice, but all
about engineering discipline: evaluation rigor, retrieval architecture, and observability",
and that the teams who shipped "defined metrics before writing code". Most walkthroughs
optimise for getting an answer out quickly, which is a poor way to learn how these systems
behave.

So the arc is: build one retriever, measure it, discover the two methods miss disjoint sets,
and only then fuse and rerank, measuring each step. `results` is filled in across three
cells rather than one, which is the point rather than an inconvenience.

The cell that earns the structure prints which questions each method alone gets right. Four
are found only by meaning and three only by keyword, so the case for hybrid search is data
in this notebook rather than an assertion.

**This had been recorded as fixed once before and was not.** Evaluation drifted back to
section 6 of 8, after four sections of technique. Check its position before shipping any
restructure.

## Two bugs that made the notebook teach the wrong lesson

**Found on 2026-08-24 by a reviewer agent, verified before acting on. Both had been in the
notebook for its whole life and neither broke anything visibly.**

1. **RRF was degenerate.** `damp=60` came from the original paper, which fused lists of a
   thousand. At `pool=25` the best a single list can offer is 1/60 while anything on both
   lists scores at least 2/84, so every intersection beat every non-intersection and rank
   stopped counting. It was a set-intersection vote wearing rank fusion's name. The sweep now
   lives in the notebook: 0.94 at `damp=1` against 0.83 at 60.
2. **BM25 scored the stopwords.** `rank_bm25` floors negative IDF to 1.18 here, a quarter of
   what `5433` earns, so in "how do I wipe my local database" the words `how` and `do` nearly
   outweighed `database`. Queries now drop a stopword list, as every production text index
   does. By-keyword recall went 0.72 to 0.89.

Together these produced the notebook's one non-obvious finding, that fusion raises recall and
lowers MRR, and the prose explained it with a sentence that was wrong twice over ("it cannot
tell first place from fifth", of an algorithm that uses nothing but rank). **A borrowed
constant is a claim like any other. Sweep it or do not print a lesson about it.**

## Cold reads find what checks cannot

A subagent read the notebook on 2026-08-23 as the target reader, in order, reporting only
where comprehension broke. It found thirteen problems, several of them things no script
catches:

- **The pipeline figure's section labels were stale**, four of five wrong, and it is the
  first thing on the page. `check.py` only looked at `rag.md`; it now checks `figures.py`
  too. Figure text lives outside the prose and skips every review.
- **The rank-movement paragraph described three chunks at ranks 3, 4 and 7** moving to the
  top. There are two, at ranks 2 and 5, moving to 1 and 3. Written against an older chunking
  and never re-derived, sitting directly under the figure that disproves it.
- **The two-caveats paragraph appeared twice, verbatim**, and the first copy cited "150"
  from an experiment two sections later.
- **"Recall went up and MRR did not"** hid the interesting result: fusion pushes MRR *below*
  dense alone.
- **RAG was never expanded**, yet section 8 says "the G in RAG". **Context window** and
  **transformer** were both load-bearing and undefined, the second doing the work of
  explaining why any of it works.
- The staleness cell's claim was disproved by its own output, and the `queries.json`
  illustration used key names the code below it does not read.

**Run a cold read after any restructure.** Give a subagent the reader's background, tell it
to read in order and report only comprehension breaks, and tell it not to suggest rewrites.
Findings 1 through 4 were all introduced by edits that passed every check.

## Reader questions are the bug report

**When Sean asks a question mid-session, that is the page failing, not the reader.** Four
questions came up on 2026-08-23 and three pointed at the same section:

| the question | what was missing |
|---|---|
| "make embeddings from an LLM?" | the page led with "it is a transformer, as is nearly every LLM", asserting sameness before difference |
| "do elements correspond to chunks or words?" | it never said a row is one chunk, or that the 384 numbers mean nothing individually |
| "what does a negative cosine mean?" | it defined the term with its own geometry, circularly |
| "are we picking LangChain then using other libs?" | a LlamaIndex example sat beside a LangChain package with no stated roles |

All four are fixed. Prefer this signal over any other when deciding what to change, because
it is the only evidence here from someone reading cold. Section 2 was split into 2 and 3 on
the strength of it: 609 words carrying two ideas was where every misreading landed.

## What the code rule cost, 2026-08-23

Applying the rule above took the notebook from 231 code lines to 159, and no cell is over
15. What moved:

- **Chunking**: 37 hand-written lines became `RecursiveCharacterTextSplitter` plus a
  three-line cell that prints the seam between two chunks. The separator list is passed
  explicitly rather than left to the default, because that list is what "recursive
  character" means.
- **BM25**: the scoring formula came from `rank_bm25`, and the frequency table stayed. The
  displayed IDF now reads out of `bm25.idf` rather than a reimplemented formula, which
  surfaced something worth saying: the library floors negative IDF, so `the` reads 1.18
  instead of the -0.45 the textbook formula gives.
- **Eval, prompt assembly, the comparison printer**: trimmed, nothing conceptual removed.

## Commands

```bash
.venv/bin/python check.py                          # every invariant, run this always
.venv/bin/jupyter lab rag.ipynb                    # read/run it
.venv/bin/jupytext --sync rag.md                   # after editing either file
MPLCONFIGDIR=.mplcache .venv/bin/jupyter nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.timeout=1800 rag.ipynb   # refresh outputs
```

`check.py` verifies the markers, the figure's file count, the results table against the
notebook's real output in both doc files, section cross-references, the code budget,
committed stderr, missing outputs, the prose ceiling, sentence length and the dash rule.
Every one of those exists because it broke silently once. The procedure around it is the
`rag-notebook` skill.

`MPLCONFIGDIR` is not optional. Executing without it in a restricted environment writes a
matplotlib cache warning into cell 1 and commits a temp path into the notebook.

Run everything from the repo root; all paths in the notebook are relative. A full execute
takes about 30s on CPU. The venv exists; fresh setup is `python -m venv .venv && .venv/bin/pip
install -r requirements.txt`.

## The edit loop

1. **Edit `rag.md`.** Never edit `rag.ipynb` in the same session as `rag.md`. They are a
   jupytext pair and sync direction is decided by mtime, so touching both loses one side.
2. `.venv/bin/jupytext --sync rag.md`
3. Re-execute, but only if code changed. Prose-only edits do not need it, because `--sync`
   preserves outputs for unchanged cells.
4. `.venv/bin/python check.py` must print `all clear`. Not optional, and not slow.

**Changing one parameter moves every number, in four places at once.** `chunk_size` is the
worst: it changes the chunk count, the IDF table, both searches, all eight cells of the
results table, the cosine values in section 3, and which questions each method gets right.
Those numbers live in `rag.md` prose, `README.md`, `CLAUDE.md` and two figures, and nothing
crashes when they disagree.

So after a parameter change: re-execute, read the real output, then correct the prose from
what it printed rather than editing toward what you expected. `check.py` compares both doc
tables against the notebook, but it cannot check a sentence reading "lifts recall to 0.89".
Renumbering sections has the same shape: `check.py` catches dangling references in `rag.md`,
not in `README.md`, so grep that by hand.

**Both ceilings are already reached, so adding requires cutting first.** Look for duplication
before padding, since the usual cause is one point made in two sections. Do not raise a
budget to fit a change.

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
- **No vector database and no orchestration framework.** `E @ q` is the search, and the
  *In production* notes name what you would really use. Single-purpose libraries are a
  different thing and are preferred: `langchain-text-splitters` is fine, `langchain` is not.
- **Dependencies stay pure Python and install in seconds.** Nothing here needs a compiler.

## Length

**Prose bloat is the standing failure mode, and it is additive.** Every individual fix
looks reasonable and they accumulate. The notebook went from 3,492 to 4,349 prose words in
a single session of "improvements" while Sean was asking for it to be *clearer*. Treat 3,000
as the ceiling.

**Markdown table rows do not count toward it, and have their own 150-word budget.** That
split was made on 2026-08-23, after the roadmap grew into a full eight-section outline and
several rounds of trimming went into deleting explanation to make room for navigation. A
table of contents should not compete with the text it indexes. This is not licence to raise
the prose ceiling: that one has never moved.

Code bloats the same way and went uncounted for longer. Its budget lives in *Use the
library, then break it down* above, with `check.py --code` behind it.

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

Current numbers, quoted by the prose in sections 5 and 6. **If these change, the
surrounding markdown has to change too**:

| method | recall@5 | MRR |
|---|---|---|
| by meaning | 0.78 | 0.61 |
| by keyword | 0.89 | 0.52 |
| both, fused | 0.94 | 0.60 |
| fused + rerank | 0.94 | 0.77 |

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

Syncing the profile README on 2026-08-22 moved the chunk count by two and left
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
were defined, and `dense + BM25` was in the pipeline diagram before section 4. Figures skip
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
numbers over flattering ones: section 6 says out loud that 18 questions cannot referee 0.83
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
- **Cosine fan** (section 3): one chunk as reference, four others at the angle the model
  produced, 90 degrees marked. Beside it, every pair of chunks against random 384-d
  directions, which is what says a cosine of 0.45 is high.
- **Rank movement** (section 5): candidates before and after reranking. Labels carry the
  chunk index because several chunks share a filename, and orange marks the correct
  answers so the reorder can be judged rather than admired.
- **Retrieval quality** (section 6): recall@5 and MRR for all four methods, drawn only
  once all four exist.
- **Locator** (under every heading from 2): the pipeline's six steps at a glance, marked one
  of three ways. The only figure that repeats, and the only one whose job is navigation.

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
- Evaluation sat late twice, with four sections of technique before any evidence. See
  *Measure before you improve*.
- The `[:420]` slice on the sample chunk cut mid-word and looked like broken output, while
  hiding the best example in the notebook: a chunk cut at the 900 limit whose severed
  sentence survives whole in the next chunk because of the overlap.
