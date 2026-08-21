# RAG from scratch

**The problem:** somewhere in a company wiki, a support archive, or a codebase nobody has
read end to end, there is a paragraph that answers your question. Keyword search makes you
guess the author's words, so `grep "wipe the database"` misses the file that says
`make reset`. And a language model has never seen those documents.

A single notebook that builds **retrieval-augmented generation** up from numpy, one piece
at a time, on 13 markdown files of real project documentation.

No framework, no vector database, no API keys. The embedding model and the reranker are
ordinary `sentence-transformers` calls, since writing those yourself teaches nothing;
everything between them is written out so you can read it, and the last section maps each
piece to what you would use in production. Two small models download once (~170 MB);
everything after that runs offline on a laptop CPU in about a minute.

It is written to be *read*, not just run. The outputs and figures are committed, so it
makes sense start to finish without executing a cell.

## What RAG is, in one paragraph

A language model can't see your documents. RAG is the workaround: when someone asks a
question, search your documents for the handful of paragraphs most likely to contain the
answer, paste those into the prompt, and ask the question with that context attached. The
"generation" half is one ordinary API call. Essentially all of the engineering is in the
search, which is why this notebook spends six sections there and one on the prompt.

**Assumed background:** Python and numpy, specifically that `@` does a dot product.
Softmax, attention and BM25 are built as they come up.

A refresher and an exploration, filling in gaps as I go. If you are somewhere similar,
it should work for you too: sections that re-derive a fundamental are headed
**Refresher** and open with a line telling you to skip if it's already fresh, so the
notebook reads either way.

## The vocabulary

Enough to read the notebook. Each is introduced again in context.

| term | meaning |
|---|---|
| **corpus** | the pile of documents being searched. Here, 13 markdown files |
| **chunk** | one searchable piece of a document. Retrieval returns chunks, not files |
| **embedding** | a list of numbers a neural net produces for text, arranged so similar *meaning* lands on similar numbers |
| **cosine similarity** | how close two embeddings point in the same direction; the standard way to compare them |
| **dense retrieval** | search by embedding similarity. Matches meaning, misses exact tokens |
| **BM25** | classic keyword search. Matches exact words, misses paraphrases |
| **hybrid search** | running both and merging the two ranked lists |
| **reranking** | a slower, more accurate model re-ordering the top candidates |
| **recall@5** | did the right passage land in the top 5? |
| **MRR** | mean reciprocal rank. How *high* did the first correct hit land? |

## Run it

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab rag.ipynb
```

## What's here

| | |
|---|---|
| `rag.ipynb` | the notebook, read this |
| `build_notebook.py` | generates `rag.ipynb`; edit here and re-run, never the JSON |
| `corpus/` | 13 markdown files, ~13,000 words, from my own repos |
| `queries.json` | 18 hand-labelled questions, each with the exact string a correct passage must contain |

## The arc

0. **Scoring with vectors**: dot products, softmax, query/key/value, built from numpy
   alone. Ends by explaining what attention is, having just built one.
1. **Chunking**: split on headings, then window. The least glamorous decision and usually
   the one that decides whether any of it works.
2. **Embedding**: what 199 chunks look like in vector space, and why a raw similarity
   score means less than people assume.
3. **Three ways to search**: dense (`E @ q`, one line), BM25 from scratch, and reciprocal
   rank fusion over both.
4. **Reranking**: bi-encoder vs cross-encoder, and the cheap-then-expensive cascade.
5. **Evaluation**: label 18 questions, count how often each method finds the answer, then
   watch the eval catch a plausible change that quietly breaks things.
6. **The generation half**: assembling the prompt, and what each instruction prevents.
7. **What's next**: contextual retrieval, metadata filtering, query rewriting, ColBERT.

## Results

| method | recall@5 | MRR |
|---|---|---|
| dense only | 0.78 | 0.69 |
| BM25 only | 0.78 | 0.62 |
| hybrid (RRF) | 0.89 | 0.65 |
| hybrid + cross-encoder rerank | **0.94** | **0.85** |

Dense and BM25 tie, and fail on *different* questions. Dense can't retrieve the exact
token `5433`; BM25 can't get from "wipe my local database" to `make reset`. Fusing
them fixes recall. Only the cross-encoder fixes ranking.

One question defeats all four methods, and section 5 works through why. The ceiling is
more instructive than the score.

## Why this corpus

It is far too small to actually need retrieval, which is the point: a test bed, not a use
case. Evaluating retrieval means knowing whether a result is *right*, and on an unfamiliar
corpus that is the expensive part, which is why so many RAG demos stop before the
evaluation.

It also has a property I didn't plan: three of these repos run a dev server on port 5173,
so "start the frontend dev server" is unanswerable without knowing which project you
meant. It's a worked example of a question no retriever can fix, only metadata filtering
can.
