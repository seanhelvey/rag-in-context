# RAG in context

A notebook about retrieval-augmented generation: getting a language model to answer
questions from documents it was never trained on.

The trick is search. Find the few paragraphs most likely to hold the answer, paste only
those into the prompt. Nearly all the engineering lives in that finding step, and most of
it predates the models. This is written as a refresher, and section 1 places each piece
against the ML you already know.

Here is the problem in one example. Ask *"how do I wipe my local database and start
over?"* about a project whose README says `make reset`. Keyword search misses it, because
you did not guess the author's words. Search by meaning finds it, because "wipe" and
"reset" sit near each other once words are vectors. Now ask which port local Postgres
listens on. The answer is the bare token `5433`, so keyword search goes straight to it
and search by meaning never surfaces it. A real system runs both, and section 3 shows
each one failing where the other works.

About a twenty minute read. It is written to be *read* as much as run: outputs and figures
are committed, so it makes sense start to finish without executing a cell.

Everything runs on a laptop with no account and no API key. Two small models download once
(176 MB) and after that it works offline. A full run takes under a minute.

## Does any of it work?

Four methods, scored on 18 hand-labelled questions:

| method | recall@5 | MRR |
|---|---|---|
| by meaning (dense embeddings) | 0.78 | 0.69 |
| by keyword (BM25) | 0.78 | 0.62 |
| both, fused (RRF) | 0.89 | 0.65 |
| fused + rerank (cross-encoder) | **0.94** | **0.85** |

**recall@5** is how often the right passage appeared in the top five. **MRR** is how high
it landed. Grading is at passage level: a labelled question names the exact string a
correct passage has to contain.

The two search methods tie, and fail on different questions, which is the reason to run
both. Fusing them lifts recall. Only the reranker fixes the order.

With 18 questions each one is worth 0.06 recall, so this can say "that change was a bad
idea" and cannot referee 0.89 against 0.94. Section 5 says so rather than rounding up, and
spends a cell watching the eval catch a plausible chunking change that quietly breaks
two of the eighteen questions.

One question defeats all four. Two of these projects run a dev server on port 5173, so
*"start the frontend dev server"* has no single right answer. No retriever fixes that; a
metadata filter does. The ceiling turns out to be more instructive than the score.

## What it covers

| | |
|---|---|
| **The problem** | why a language model cannot answer from your documents, and what to do about it |
| **1. Where this sits** | RAG involves no training. Where its pieces land against supervised, unsupervised, traditional and deep learning |
| **2. From text to vectors** | documents into chunks, chunks into embeddings, and why cosine similarity is the idea you already know with the coordinates learned rather than chosen |
| **3. Two ways to search** | by meaning and by keyword, shown failing on different questions |
| **4. Combining and reordering** | fusing two ranked lists, then a slower model fixing the order |
| **5. Does any of it work?** | the 18 questions, the scores, and the eval catching a change that looks like an improvement |
| **6. The decisions that matter** | where you cut, questions with no single answer, and keeping embeddings fresh |
| **7. The generation half** | assembling the prompt, and what each instruction prevents. It prints the prompt rather than sending it, which is what keeps the notebook keyless |

Where a piece has an obvious production counterpart, an *In production* note names it on
the spot: LangChain, `rank_bm25`, Elasticsearch, Cohere Rerank and the rest.

## Run it

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab rag.ipynb
```

Run it from the repo root, since all paths are relative.

## Why this corpus

13 markdown files, about 12,000 words: READMEs and CLAUDE.md files from 8 public repos of
mine. Far too small to need retrieval, which is the point. It is a test bed
rather than a use case.

Evaluating retrieval means knowing whether a result is *right*, and on an unfamiliar
corpus that labelling is the expensive part. It is why a lot of RAG demos stop before the
evaluation. Here it is cheap, because I wrote every file in the corpus and can check an
answer by reading it.

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
| `corpus/` | the 13 markdown files |
| `queries.json` | 18 hand-labelled questions, each with the exact string a correct passage must contain |
| `check.py` | verifies every marker still appears in the corpus. Run it after changing either |

## A note on how it was made

I got help from Claude building this. Agents can be useful not just for coding but for
creating and sharing materials that help people understand key concepts.

Sean Helvey. [github.com/seanhelvey](https://github.com/seanhelvey)
