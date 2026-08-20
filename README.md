# RAG from scratch

A single notebook that builds retrieval-augmented generation up from numpy, on a
corpus of my own README and CLAUDE.md files.

No API keys. No vector database. No framework. Two small models download once
(~170 MB total) and everything after that runs offline on a laptop CPU in about
a minute.

## Run it

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab rag.ipynb
```

The notebook ships with its outputs, so it reads fine without running anything.

## What's here

| | |
|---|---|
| `rag.ipynb` | the notebook — read this |
| `build_notebook.py` | generates `rag.ipynb`; edit here and re-run, not the JSON |
| `corpus/` | 14 markdown files, ~14,000 words, vendored from my own repos |
| `queries.json` | 18 hand-labelled questions, each with the exact string a retrieved passage must contain |

## The arc

1. **Attention is already retrieval** — `softmax(QKᵀ)V` in numpy, and why RAG is the
   same operation over a store that doesn't fit in the context window.
2. **Chunking** — split on headings, then window. The least glamorous decision and
   often the one that decides whether any of it works.
3. **Embedding** — `all-MiniLM-L6-v2`, a PCA scatter of the corpus, and a similarity
   heatmap that shows why a fixed similarity threshold is a bad idea.
4. **Three ways to search** — dense (`E @ q`, one line), BM25 from scratch, and
   reciprocal rank fusion over both.
5. **Reranking** — bi-encoder vs cross-encoder, and the cheap-then-expensive cascade.
6. **Evaluation** — recall@5 and MRR over the labelled set, plus the failure list,
   which is the half that's actually useful.
7. **Chunk size** — a sweep, and an honest note about what 18 questions can and
   cannot resolve.
8. **The generation half** — assembling the prompt, and what the three instructions
   in it are each preventing.

## Results

| method | recall@5 | MRR |
|---|---|---|
| dense only | 0.83 | 0.73 |
| BM25 only | 0.83 | 0.68 |
| hybrid (RRF) | **1.00** | 0.72 |
| hybrid + cross-encoder rerank | **1.00** | **0.91** |

Dense and BM25 tie, and fail on different questions — dense misses the exact token
`5433`, BM25 can't get from "wipe my local database" to `make reset`. Fusing them
fixes recall; only the cross-encoder fixes ranking.

## Why this corpus

Because I wrote it. Evaluating retrieval means knowing whether a result is right,
and on someone else's corpus that is the expensive part. It also has a property I
didn't plan: three of these repos run a dev server on port 5173, so
"start the frontend dev server" is genuinely unanswerable without knowing which
project you meant. That's in there as a worked example of a question no retriever
can fix.
