# RAG in context

A refresher on **retrieval-augmented generation**, written to connect it to ideas that
have been around a lot longer. RAG appears in a great many job descriptions, and most of
what makes it work turns out to be search, which is a well-worn subject.

One notebook, about a twenty minute read. Everything runs on a laptop with no account and
no API key. Two small models download once (~170 MB) and after that it works offline, in
about a minute.

It is written to be *read* as much as run. Outputs and figures are committed, so it makes
sense start to finish without executing a cell.

## What it covers

| | |
|---|---|
| **The problem** | why a language model cannot answer from your documents, and what to do about it |
| **1. Where this sits** | RAG involves no training. Where its pieces land against supervised, unsupervised, traditional and deep learning |
| **2. From text to vectors** | documents into chunks, chunks into embeddings, and why cosine similarity is the idea you already know with the coordinates learned rather than chosen |
| **3. Two ways to search** | by meaning and by keyword, shown failing on different questions |
| **4. Combining and reordering** | fusing two ranked lists, then a slower model fixing the order |
| **5. Does any of it work?** | 18 labelled questions, the scores, and the eval catching a plausible change that quietly breaks things |
| **6. The decisions that matter** | where you cut, questions with no single answer, and keeping embeddings fresh |
| **7. The generation half** | assembling the prompt, and what each instruction prevents |

Where a piece has an obvious production counterpart, an *In production* note names it on
the spot: LangChain, pgvector, `rank_bm25`, Cohere Rerank, RAGAS and the rest.

## Results

| method | recall@5 | MRR |
|---|---|---|
| by meaning (embeddings) | 0.78 | 0.69 |
| by keyword (BM25) | 0.78 | 0.62 |
| both, fused | 0.89 | 0.65 |
| fused + rerank | **0.94** | **0.85** |

The two search methods tie, and fail on *different* questions. Embeddings cannot retrieve
the exact token `5433`; keyword search cannot get from "wipe my local database" to
`make reset`. Fusing them fixes recall, and only the reranker fixes the order.

One question defeats all four, and section 5 works through why. The ceiling is more
instructive than the score.

## Run it

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab rag.ipynb
```

Run it from the repo root, since all paths are relative.

## How it is built

`rag.md` is the source and `rag.ipynb` is generated from it. The two are paired with
[jupytext](https://jupytext.readthedocs.io/), so editing either updates the other:

```sh
.venv/bin/jupytext --sync rag.md                   # after editing either file
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 rag.ipynb   # refresh the committed outputs
```

| | |
|---|---|
| `rag.md` | the source. Plain markdown, code in fences. Edit this |
| `rag.ipynb` | generated, with outputs committed |
| `figures.py` | matplotlib drawing, kept out of the notebook so cells stay about retrieval |
| `corpus/` | 13 markdown files, ~12,000 words: READMEs and CLAUDE.md files from 8 public repos of mine |
| `queries.json` | 18 hand-labelled questions, each with the exact string a correct passage must contain |

## Why this corpus

It is far too small to actually need retrieval, which is the point: a test bed, not a use
case. Evaluating retrieval means knowing whether a result is *right*, and on an unfamiliar
corpus that is the expensive part, which is why so many RAG demos stop before the
evaluation.

It also has a property I didn't plan: two of these projects run a dev server on port 5173,
so "start the frontend dev server" is unanswerable without knowing which one you meant. It
is a worked example of a question no retriever can fix, only metadata filtering can.

## A note on how it was made

Yes, I got help from Claude building this. Agents can be useful not just for coding but
for creating and sharing materials that help people understand key concepts.
