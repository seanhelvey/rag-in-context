# RAG from scratch — working notes

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

The venv already exists. Fresh setup is `python -m venv .venv && .venv/bin/pip install
-r requirements.txt`.

## The one rule that matters

**Never hand-edit `rag.ipynb`.** It is generated. Edit `build_notebook.py`, re-run it,
then re-execute the notebook to refresh the outputs. Editing the JSON directly means
the next `build_notebook.py` run silently throws the change away.

Cells are defined by `md(...)` and `code(...)` calls in order, each holding a `'''`
string. Cell sources therefore must not contain `'''`, and a literal `\n` inside a
cell needs to be written `\\n` so it survives into the notebook.

## Constraints that are deliberate

- **No API keys, ever.** Local models only. The notebook must run on a laptop with no
  account, no billing, and nothing to leak on a shared screen. The generation step
  assembles a prompt and prints it rather than sending it — see section 7.
- **No vector database, no framework.** `E @ q` is the whole search. The point is that
  a vector DB is that line plus persistence, filtering, and an ANN index. Adding
  LangChain or Pinecone here would delete the thing being taught.
- **Must run offline** once the two models are cached. Verified with
  `HF_HUB_OFFLINE=1`. Interview wifi is not to be trusted.
- **Outputs stay committed.** The notebook has to read correctly without being run.

## Evaluation

`queries.json` holds 18 questions, each with a `marker`: the exact string that must
appear in a retrieved passage for the retrieval to count. Grade at passage level, not
file level — file-level labels scored 1.00 for all four methods and could not
distinguish them at all. That is why the marker scheme exists.

When adding a query, check the marker actually appears in `corpus/` first. A marker
that matches nothing silently scores zero for every method and drags the whole
table down.

Current numbers, which the prose in sections 5–6 quotes directly — **if these change,
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
`howgood-apply` is excluded too — an application to a specific company.

Two properties are load-bearing and should survive any corpus change: dense and BM25
must fail on *different* questions (that is the entire argument for hybrid search),
and `:5173` appearing in three separate repos gives a worked example of a question no
retriever can answer without metadata filtering.

## Audience

Written for a technical reader refreshing on RAG, not a specialist — and explicitly
meant to be usable by someone else in the same position. Practical consequence:
**define a term the first time it is used.** PCA, IDF, MRR, bi-encoder, cross-encoder
and the rest all get a sentence in place rather than being assumed. Section 8 is
labelled as vocabulary so it reads as orientation, not a reading list.

The closing "if you remember five things" recap exists so the notebook can be
summarized out loud from memory. Keep it in sync with the sections.

## Style

Comments say *why*. The reader is an interviewer deciding whether I understand
retrieval, not someone who needs `argsort` explained. Honest numbers over flattering
ones: section 6 says out loud that its peak is one question wide, and that is the
most valuable paragraph in the notebook.
