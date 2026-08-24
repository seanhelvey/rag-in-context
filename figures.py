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


def chunk_windows(sec_len=1100, max_chars=400, overlap=80):
    """A long section with the windows drawn underneath and the overlaps marked."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.barh(1.75, sec_len, height=.34, color="#dde6f2", edgecolor=BLUE)
    ax.text(sec_len / 2, 1.75, f"one section, {sec_len} chars", ha="center", va="center",
            fontsize=9, color=BLUE)

    start, row = 0, 0
    while start < sec_len:                 # the stride the splitter uses
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
    step = 250 if sec_len <= 1500 else 500
    ax.set_yticks([]); ax.set_xticks(range(0, sec_len + 1, step))
    ax.set_xlabel("characters", fontsize=9)
    ax.set_title(f"Windowing a long section: {max_chars} wide, {overlap} of overlap",
                 fontsize=11, loc="left")
    _clean(ax, ("top", "right", "left"))
    plt.tight_layout(); plt.show()


def rank_movement(query, before, after, chunks, marker=None):
    """How the cross-encoder reorders the candidates hybrid retrieval returned.

    Labels carry the chunk index, because several chunks share a filename and would
    otherwise be indistinguishable. If a marker is given, chunks containing it are the
    correct answers and get highlighted, so a reorder can be judged rather than admired.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, idx in enumerate(before):
        y0, y1 = i, after.index(idx)
        hit = marker is not None and marker in chunks[idx]["text"]
        up = y1 < y0
        ax.plot([0, 1], [y0, y1], marker="o",
                color=ORANGE if hit else GREEN,
                lw=2.6 if hit else (1.8 if up else 1.0),
                alpha=1.0 if hit else (0.85 if up else 0.32),
                zorder=3 if hit else 2)
        tag = f'{chunks[idx]["file"][:24]}#{idx}'
        colour, weight = ("#8a6100", "bold") if hit else ("#444", "normal")
        ax.text(-0.03, y0, f'{y0 + 1}.  {tag}', ha="right", va="center", fontsize=8,
                color=colour, fontweight=weight)
        ax.text(1.03, y1, f'{y1 + 1}.  {tag}', ha="left", va="center", fontsize=8,
                color=colour, fontweight=weight)

    if marker is not None:
        ax.text(0.5, len(before) - 0.3,
                f'orange: the chunk contains "{marker}", so it is a correct answer',
                fontsize=8.5, color="#8a6100", ha="center")
    ax.set_xlim(-0.85, 1.85); ax.invert_yaxis(); ax.set_xticks([0, 1])
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


def cosine_fan(ref_label, labels, cosines, E):
    """One chunk as a reference, others drawn at their true angle from it, plus the
    spread of every pair in the corpus against random directions."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))

    a1.add_patch(plt.matplotlib.patches.Wedge((0, 0), 1.6, 90, 118, color="#f6f6f6"))
    a1.text(-1.42, 0.72, "past 90 degrees:\nnegative cosine", fontsize=7.5, color=GREY)
    a1.annotate("", xy=(1.0, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=3))
    a1.text(1.04, 0, f"  {ref_label}\n  (measured from)", fontsize=8.5,
            color=ORANGE, va="center", fontweight="bold")

    # Labels sit at the arrow tips and run rightwards, so two arrows only 10 degrees
    # apart put their text at nearly the same height and overprint. Staggering the
    # radius by rank separates them vertically whatever the angles happen to be.
    order = np.argsort([np.arccos(np.clip(c, -1, 1)) for c in cosines])
    for rank, k in enumerate(order):
        label, cos = labels[k], cosines[k]
        deg = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        r = np.radians(deg)
        x, y = np.cos(r), np.sin(r)
        a1.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.2))
        lr = 1.07 + 0.23 * rank
        a1.plot([x, x * lr], [y, y * lr], color="#cfe6dd", lw=.9, zorder=0)
        a1.text(x * lr, y * lr, f" {label}\n cos {cos:+.2f}, {deg:.0f} deg",
                fontsize=8, color="#1a6a55", va="center",
                ha="left" if x > -0.1 else "right")

    th = np.radians(np.linspace(0, 118, 200))
    a1.plot(np.cos(th) * .99, np.sin(th) * .99, color="#e8e8e8", lw=1, zorder=0)
    a1.plot([0, 0], [0, 1.62], color=GREY, lw=1.1, ls="--")
    a1.set_xlim(-1.6, 2.05); a1.set_ylim(-0.2, 2.15)
    a1.set_aspect("equal"); a1.axis("off")
    a1.set_title("Chunks as arrows, at their real angles", fontsize=11, loc="left")

    S = E @ E.T
    real = S[np.triu_indices(len(E), k=1)]
    rng = np.random.default_rng(0)
    r = rng.normal(size=(4000, E.shape[1]))
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    rand = (r[:2000] * r[2000:]).sum(1)

    a2.hist(rand, bins=70, density=True, color=GREY, alpha=.75,
            label=f"two random directions in {E.shape[1]}-d")
    a2.hist(real, bins=70, density=True, color=GREEN, alpha=.75,
            label="two actual chunks")
    a2.axvline(float(real.mean()), color=GREEN, lw=1.2, ls="--")
    a2.text(float(real.mean()) + .015, a2.get_ylim()[1] * .70,
            f"chunk average {real.mean():.2f}", fontsize=8, color=GREEN)
    a2.set_xlim(-.35, 1)
    a2.set_xlabel("cosine similarity", fontsize=9); a2.set_yticks([])
    a2.set_title("Every pair of chunks, against pure chance", fontsize=11, loc="left")
    a2.legend(fontsize=8, frameon=False, loc="upper right")
    _clean(a2, ("top", "right", "left"))
    plt.tight_layout(); plt.show()


def pipeline():
    """The whole notebook on one screen: what runs once, and what runs per question."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    TOP, BOT, H = 2.55, 0.75, 0.60

    def box(x, y, w, label, sub, face, edge, bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, H, boxstyle="round,pad=0.015",
                                    facecolor=face, edgecolor=edge, lw=1.4))
        ax.text(x + w / 2, y + (0.38 if sub else 0.30), label, ha="center", va="center",
                fontsize=9, color="#222", fontweight="bold" if bold else "normal")
        if sub:
            ax.text(x + w / 2, y + 0.16, sub, ha="center", va="center", fontsize=7.5,
                    color="#777")

    def arrow(x0, y, x1):
        ax.annotate("", xy=(x1, y + H / 2), xytext=(x0, y + H / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#c4c4c4", lw=1.5))

    pale, mid = "#eef6f2", "#dceee6"
    row1 = [("corpus", "13 markdown files", 1.55, "#f2f2f2", "#ccc"),
            ("chunk", "section 2", 1.35, pale, GREEN),
            ("embed", "section 3", 1.35, pale, GREEN),
            ("vectors", "one per chunk", 1.35, mid, GREEN)]
    x = 0.35
    for label, sub, w, face, edge in row1:
        box(x, TOP, w, label, sub, face, edge)
        if x > 0.35:
            arrow(x - 0.32, TOP, x - 0.04)
        x += w + 0.32
    e_centre = x - row1[-1][2] - 0.32 + row1[-1][2] / 2

    row2 = [("question", None, 1.35, "#f2f2f2", "#ccc"),
            ("keyword + meaning", "section 4", 2.05, pale, ORANGE),
            ("fuse", "section 6", 1.15, pale, ORANGE),
            ("rerank", "section 6", 1.25, pale, ORANGE),
            ("prompt", "section 8", 1.35, pale, GREEN),
            ("answer", None, 1.35, "#f2f2f2", "#ccc")]
    x, spans = 0.35, []
    for label, sub, w, face, edge in row2:
        box(x, BOT, w, label, sub, face, edge)
        spans.append((x, x + w))
        if x > 0.35:
            arrow(x - 0.30, BOT, x - 0.04)
        x += w + 0.30

    # Route the E-to-search connector through the empty band between the rows, so it
    # does not graze the tops of the boxes it passes over.
    search_centre = sum(spans[1]) / 2
    band = (TOP + BOT + H) / 2
    ax.plot([e_centre, e_centre, search_centre], [TOP - 0.02, band, band],
            color=GREEN, lw=1.5, ls="--")
    ax.annotate("", xy=(search_centre, BOT + H + 0.02), xytext=(search_centre, band),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5, ls="--"))
    ax.text(search_centre + 0.12, band + 0.14, "searched, never rebuilt per question",
            fontsize=7.5, color=GREEN, va="center")

    ax.text(0.35, TOP + H + 0.30, "ONCE, whenever the documents change",
            fontsize=9, color="#555", fontweight="bold")
    ax.text(0.35, BOT + H + 0.30, "EVERY question", fontsize=9, color="#555",
            fontweight="bold")

    lo, hi = spans[1][0], spans[3][1]
    ax.plot([lo, lo, hi, hi], [BOT - 0.16, BOT - 0.28, BOT - 0.28, BOT - 0.16],
            color=ORANGE, lw=1.4)
    ax.text((lo + hi) / 2, BOT - 0.52, "retrieval: nearly all the engineering, "
            "and what section 5 measures", fontsize=8.5, color="#8a6100", ha="center")

    ax.set_xlim(0, 11.2); ax.set_ylim(0, 3.75); ax.axis("off")
    plt.tight_layout(); plt.show()


def locate(step=None, spans=None, loop=False, note=None):
    """A thin callback to the pipeline figure, marking where the reader is now.

    Keeps that figure's box names, its two-phase split and its colours. Orange is
    whatever this section is about, exactly as orange is the retrieval bracket on
    the big diagram: the one thing to look at. Green dashed is the vectors leaving
    the indexing phase and being searched, which is the same green dashed line the
    big diagram draws, so a reader is not asked to learn a second scheme.

    Dropped from the big diagram: corpus, vectors, question and answer, the boxes
    carrying no section number. Everything else is deliberately identical.
    """
    from matplotlib.patches import FancyBboxPatch

    steps = [("chunk", 2, 1.5), ("embed", 3, 1.5),
             ("keyword + meaning", 4, 2.3), ("fuse", 6, 1.2),
             ("rerank", 6, 1.35), ("prompt", 8, 1.5)]
    ONCE = 2                       # boxes before this run once, the rest per question
    gap, phase_gap = 0.26, 1.30
    names = [n for n, _, _ in steps]
    # A step naming no box lights nothing and says nothing, which is worse than
    # an error because the strip still renders and looks deliberate.
    for one in ((step,) if isinstance(step, str) else step or ()):
        assert one in names, f"locate({one!r}): expected one of {names}"

    fig, ax = plt.subplots(figsize=(11.5, 1.5 if spans else 1.05))
    x, at = 0.0, []
    for i, (label, sec, w) in enumerate(steps):
        if i == ONCE:
            x += phase_gap - gap
        at.append((x, w))
        here = label == step or (isinstance(step, tuple) and label in step)
        ax.add_patch(FancyBboxPatch((x, 0), w, 0.5, boxstyle="round,pad=0.015",
                                    facecolor=ORANGE if here else "#f5f5f5",
                                    edgecolor=ORANGE if here else "#dedede", lw=1.2))
        ax.text(x + w / 2, 0.31, label, ha="center", va="center", fontsize=8.5,
                color="white" if here else "#9a9a9a",
                fontweight="bold" if here else "normal")
        ax.text(x + w / 2, 0.13, f"section {sec}", ha="center", va="center",
                fontsize=6.5, color="#fff3dc" if here else "#c4c4c4")
        if i and i != ONCE:
            ax.annotate("", xy=(x - 0.04, 0.25), xytext=(x - gap + 0.04, 0.25),
                        arrowprops=dict(arrowstyle="-|>", color="#dedede", lw=1.1))
        x += w + gap
    end = x - gap

    # The vectors crossing from the once phase into the per-question phase. Same
    # green dashed line as the big diagram, so the two boxes do not read as equal.
    x0 = at[ONCE - 1][0] + at[ONCE - 1][1]
    x1 = at[ONCE][0]
    ax.annotate("", xy=(x1 - 0.04, 0.25), xytext=(x0 + 0.04, 0.25),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.3,
                                ls=(0, (3, 3))))
    ax.text((x0 + x1) / 2, 0.40, "vectors", ha="center", fontsize=6.5, color=GREEN)
    ax.text(at[0][0], 0.66, "ONCE", fontsize=7, color="#aaa", fontweight="bold")
    ax.text(at[ONCE][0], 0.66, "EVERY question", fontsize=7, color="#aaa",
            fontweight="bold")

    if spans:
        lo, hi = spans
        a, b = at[lo][0], at[hi][0] + at[hi][1]
        rail = -0.30
        dots = dict(color=ORANGE, lw=1.4, ls=(0, (2.5, 2.5)))
        if loop:
            ax.plot([b, b, a], [-0.04, rail, rail], **dots)
            ax.annotate("", xy=(a, -0.04), xytext=(a, rail),
                        arrowprops=dict(arrowstyle="-|>", **dots))
        else:
            ax.plot([a, a, b, b], [-0.04, rail, rail, -0.04], **dots)
        if note:
            ax.text((a + b) / 2, rail - 0.12, note, ha="center", va="top",
                    fontsize=7.5, color="#8a6100", fontweight="bold")
        ax.set_ylim(-0.72, 0.84)
    else:
        ax.set_ylim(-0.05, 0.84)
    ax.set_xlim(-0.1, end + 0.1)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
