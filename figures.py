"""Plotting for rag.ipynb.

Drawing code lives here so the notebook cells stay short. Nothing in this file
teaches anything about retrieval; it is presentation only. The interesting code
is all inline in the notebook.
"""
import numpy as np
import matplotlib.pyplot as plt

GREEN, BLUE, GREY, ORANGE = "#2a7", "#47c", "#999", "#e0a300"


def _clean(ax, sides=("top", "right")):
    for s in sides:
        ax.spines[s].set_visible(False)


def softmax_steps(scores, labels, softmax):
    """Raw score, after exp, after dividing, plus the same scores sharpened."""
    x, bw = np.arange(len(labels)), 0.26
    ex = np.exp(scores - scores.max())
    nrm = ex / ex.sum()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar(x - bw, scores, bw, color=GREY, label="raw score")
    a1.bar(x, ex, bw, color=BLUE, label="after exp")
    a1.bar(x + bw, nrm, bw, color=GREEN, label="divided by total")
    for i in range(len(labels)):
        a1.text(i + bw, nrm[i] + .03, f"{nrm[i]:.2f}", ha="center", fontsize=8, color=GREEN)
        if scores[i] == 0:                  # a zero bar has no height, so draw a stub
            a1.plot([i - bw - .12, i - bw + .12], [0, 0], color=GREY, lw=2.5)
            a1.text(i - bw, .05, "0", ha="center", fontsize=8, color=GREY)
    a1.annotate("a score of 0 becomes 0.37,\nnot 0", xy=(2, ex[2]), xytext=(1.75, .72),
                fontsize=8, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
    a1.set_title("Softmax, one step at a time", fontsize=11)
    a1.legend(fontsize=8, frameon=False, loc="upper right")
    a1.set_ylim(0, 1.25); a1.set_ylabel("value", fontsize=9)

    for mult, colour, mark in ((1, GREY, "o"), (4, BLUE, "s"), (20, GREEN, "^")):
        a2.plot(x, softmax(scores * mult), marker=mark, color=colour, lw=2,
                label=f"scores x{mult}")
    a2.set_title("The same scores, sharpened", fontsize=11)
    a2.set_ylim(-.05, 1.05); a2.set_ylabel("weight", fontsize=9)
    a2.legend(fontsize=8, frameon=False)

    for ax in (a1, a2):
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); _clean(ax)
    plt.tight_layout(); plt.show()


def attention_head(labels, scores, weights, V, answer):
    """One attention head as five labelled columns: key, score, weight, value, product."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    cx = {"key": 0.6, "score": 2.5, "weight": 4.1, "val": 6.2, "contrib": 8.1}

    for header, xx in (("key (K)", cx["key"]), ("score = K @ q", cx["score"]),
                       ("weight = softmax", cx["weight"]), ("value (V)", cx["val"]),
                       ("weight x value", cx["contrib"])):
        ax.text(xx, 3.85, header, fontsize=9, fontweight="bold", color="#444")

    for y, label, score, wt, v in zip([3, 2, 1, 0], labels, scores, weights, V):
        ax.text(cx["key"], y, label, fontsize=9, va="center", color="#333")
        ax.annotate("", xy=(cx["score"] - .25, y), xytext=(cx["key"] + 1.15, y),
                    arrowprops=dict(arrowstyle="->", color="#ccc", lw=1))
        ax.text(cx["score"], y, f"{score:.2f}", fontsize=9, va="center", color=BLUE)
        ax.barh(y, wt * 1.7, left=cx["weight"], height=.34, color=GREEN, alpha=.85)
        ax.text(cx["weight"] + .05 + wt * 1.7, y, f"{wt:.3f}", fontsize=8, va="center",
                color=GREEN)
        ax.text(cx["val"], y, f"[{v[0]:>5.1f}, {v[1]:>5.1f}]", fontsize=9, va="center",
                family="monospace", color="#333")
        part = wt * v                      # fade the rows that barely contribute
        ax.text(cx["contrib"], y, f"[{part[0]:>5.2f}, {part[1]:>5.2f}]", fontsize=9,
                va="center", family="monospace", color="#333",
                alpha=.35 + .65 * min(1, wt * 1.7))

    ax.plot([cx["contrib"] - .1, cx["contrib"] + 1.55], [-.55, -.55], color="#444", lw=1)
    ax.text(cx["contrib"], -.95, f"sum  [{answer[0]:.2f}, {answer[1]:.2f}]", fontsize=10,
            family="monospace", color=GREEN, fontweight="bold", va="center")
    ax.text(cx["key"], -.95, "the blended output, mostly made of the rows that matched",
            fontsize=9, color=GREEN, va="center")
    ax.set_xlim(0, 10.2); ax.set_ylim(-1.4, 4.3); ax.axis("off")
    ax.set_title("One attention head, column by column", fontsize=11, loc="left")
    plt.tight_layout(); plt.show()


def chunk_windows(sec_len=2300, max_chars=900, overlap=150):
    """A long section with the windows drawn underneath and the overlaps marked."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.barh(1.75, sec_len, height=.34, color="#dde6f2", edgecolor=BLUE)
    ax.text(sec_len / 2, 1.75, f"one section, {sec_len} chars", ha="center", va="center",
            fontsize=9, color=BLUE)

    start, row = 0, 0
    while start < sec_len:                 # same loop as window() in the notebook
        end = min(start + max_chars, sec_len)
        y = 1.0 - row * .42
        ax.barh(y, end - start, left=start, height=.3, color=GREEN, alpha=.8,
                edgecolor="white")
        if start > 0:
            ax.barh(y, min(overlap, end - start), left=start, height=.3, color=ORANGE,
                    edgecolor="white")
        if end - start > 260:
            ax.text((start + end) / 2, y, f"chunk {row + 1}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        else:
            ax.text(end + 55, y, f"chunk {row + 1} ({end - start} chars)", va="center",
                    fontsize=8, color="#666")
        start += max_chars - overlap
        row += 1

    ax.text(0, 1.42, f"orange marks the {overlap}-char overlap, so a fact sitting on a "
            "cut line survives whole in at least one chunk", fontsize=8.5,
            color="#8a6100", va="center")
    ax.set_xlim(-40, sec_len + 430); ax.set_ylim(1.0 - row * .42 - .35, 2.15)
    ax.set_yticks([]); ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xlabel("characters", fontsize=9)
    ax.set_title(f"Windowing a long section: {max_chars} wide, {overlap} of overlap",
                 fontsize=11, loc="left")
    _clean(ax, ("top", "right", "left"))
    plt.tight_layout(); plt.show()


def embedding_space(E, chunks):
    """Chunks flattened to 2D with PCA, coloured by source repo."""
    from sklearn.decomposition import PCA

    xy = PCA(n_components=2, random_state=0).fit_transform(E)
    repos = sorted({c["repo"] for c in chunks})
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, repo in enumerate(repos):
        m = np.array([c["repo"] == repo for c in chunks])
        ax.scatter(xy[m, 0], xy[m, 1], s=42, alpha=0.75, color=cmap(i % 10), label=repo)
    ax.set_title("Chunks in embedding space (PCA to 2D), colored by source repo")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout(); plt.show()


def similarity_matrix(names, S):
    """Cosine similarity between a handful of hand-picked chunks."""
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(S, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for a in range(len(names)):
        for b in range(len(names)):
            ax.text(b, a, f"{S[a, b]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if S[a, b] < 0.6 else "black")
    ax.set_title("Cosine similarity between hand-picked chunks")
    fig.colorbar(im, shrink=0.8); plt.tight_layout(); plt.show()


def rank_movement(query, before, after, chunks):
    """How the cross-encoder reorders the candidates hybrid retrieval returned."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, idx in enumerate(before):
        y0, y1 = i, after.index(idx)
        moved_up = y1 < y0
        ax.plot([0, 1], [y0, y1], marker="o", color=GREEN,
                lw=2.0 if moved_up else 1.0, alpha=0.95 if moved_up else 0.4)
        ax.text(-0.03, y0, chunks[idx]["file"][:32], ha="right", va="center", fontsize=8)
        ax.text(1.03, y1, chunks[idx]["file"][:32], ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.75, 1.75); ax.invert_yaxis(); ax.set_xticks([0, 1])
    ax.set_xticklabels(["hybrid retrieval", "after cross-encoder rerank"])
    ax.set_yticks([]); ax.set_title(f'Rank movement: "{query}"')
    _clean(ax, ("top", "right", "left", "bottom"))
    plt.tight_layout(); plt.show()


def retrieval_quality(results, n_queries):
    """recall@5 and MRR side by side for every method."""
    names = list(results)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.19, [results[n][0] for n in names], 0.38, label="recall@5", color=GREEN)
    ax.bar(x + 0.19, [results[n][1] for n in names], 0.38, label="MRR", color=BLUE)
    for i, n in enumerate(names):
        ax.text(i - 0.19, results[n][0] + .015, f"{results[n][0]:.2f}", ha="center",
                fontsize=9)
        ax.text(i + 0.19, results[n][1] + .015, f"{results[n][1]:.2f}", ha="center",
                fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.08)
    ax.set_title(f"Retrieval quality over {n_queries} hand-labelled questions")
    ax.legend(); _clean(ax)
    plt.tight_layout(); plt.show()
