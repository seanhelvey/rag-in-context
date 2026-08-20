# RAG from scratch

A single notebook that builds **retrieval-augmented generation** up from numpy, on a
corpus of my own project documentation.

No API keys. No vector database. No framework. Two small models download once
(~170 MB) and everything after that runs offline on a laptop CPU in about a minute.

It is written to be *read*, not just run — the outputs and figures are committed, so
it makes sense start to finish without executing a cell.

## What RAG is, in one paragraph

A language model can't see your documents. RAG is the workaround: when someone asks a
question, search your documents for the handful of paragraphs most likely to contain
the answer, paste those into the prompt, and ask the question with that context
attached. The "generation" half is one ordinary API call. Essentially all of the
engineering is in the search — which is why this notebook spends eight sections there
and one on the prompt.

## The vocabulary

Enough to read the notebook. Each is introduced again in context.

| term | meaning |
|---|---|
| **corpus** | the pile of documents being searched — here, 13 markdown files |
| **chunk** | one searchable piece of a document. Retrieval returns chunks, not files |
| **embedding** | a list of numbers a neural net produces for text, arranged so similar *meaning* lands on similar numbers |
| **cosine similarity** | how close two embeddings point in the same direction; the standard way to compare them |
| **dense retrieval** | search by embedding similarity — matches meaning, misses exact tokens |
| **BM25** | classic keyword search — matches exact words, misses paraphrases |
| **hybrid search** | running both and merging the two ranked lists |
| **reranking** | a slower, more accurate model re-ordering the top candidates |
| **recall@5** | did the right passage land in the top 5? |
| **MRR** | mean reciprocal rank — how *high* did the first correct hit land? |

## Run it

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab rag.ipynb
```

## What's here

| | |
|---|---|
| `rag.ipynb` | the notebook — read this |
| `build_notebook.py` | generates `rag.ipynb`; edit here and re-run, never the JSON |
| `corpus/` | 13 markdown files, ~12,000 words, from my own repos |
| `queries.json` | 18 hand-labelled questions, each with the exact string a correct passage must contain |

## The arc

0. **Attention is already retrieval** — `softmax(QKᵀ)V` in numpy, and why RAG is the
   same operation over a store too big for the context window.
1. **Chunking** — split on headings, then window. The least glamorous decision and
   usually the one that decides whether any of it works.
2. **Embedding** — what 199 chunks look like in vector space, and why a raw
   similarity score means less than people assume.
3. **Three ways to search** — dense (`E @ q`, one line), BM25 from scratch, and
   reciprocal rank fusion over both.
4. **Reranking** — bi-encoder vs cross-encoder, and the cheap-then-expensive cascade.
5. **Evaluation** — recall@5 and MRR over the labelled set, then the failure list,
   which is the half that actually teaches you something.
6. **Chunk size** — a sweep, and an honest note on what 18 questions can't resolve.
7. **The generation half** — assembling the prompt, and what each instruction in it
   prevents.
8. **What's missing** — contextual retrieval, ColBERT, Matryoshka embeddings and the
   rest, as vocabulary rather than homework.

## Results

| method | recall@5 | MRR |
|---|---|---|
| dense only | 0.78 | 0.69 |
| BM25 only | 0.78 | 0.62 |
| hybrid (RRF) | 0.89 | 0.65 |
| hybrid + cross-encoder rerank | **0.94** | **0.85** |

Dense and BM25 tie — and fail on *different* questions. Dense can't retrieve the exact
token `5433`; BM25 can't get from "wipe my local database" to `make reset`. Fusing
them fixes recall. Only the cross-encoder fixes ranking.

One question defeats all four methods, and section 5 spends a page on it. That is
deliberate: the ceiling is more instructive than the score.

## Why this corpus

Because I wrote it. Evaluating retrieval means knowing whether a result is *right*,
and on an unfamiliar corpus that's the expensive part — which is why so many RAG demos
skip evaluation entirely.

It also has a property I didn't plan: three of these repos run a dev server on port
5173, so "start the frontend dev server" is genuinely unanswerable without knowing
which project you meant. That's in there as a worked example of a question no
retriever can fix, only metadata filtering can.
