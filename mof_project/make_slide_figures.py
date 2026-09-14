"""
make_slide_figures.py
=======================
Purpose-built static figures for the presentation, generated directly
from the real viewer_data.json (same numbers as the interactive demo),
NOT screenshots of the scrollable web viewer. A browser screenshot of
a 5x22 table can only show what fits one screen width -- that's why
your two screenshots showed different, both-incomplete slices of the
same matrix. A matplotlib figure has no such limit: we choose exactly
which columns to show and render them all in one frame, legibly.

Produces:
  mof_step2_costmatrix.png  -- the Stage-1 cost matrix, print-ready
  mof_step3_optimization.png -- X before/after, print-ready
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

with open("viewer_data.json") as f:
    DATA = json.load(f)

CAND = DATA["candidates"][0]  # qmof-0bbf75f.cif -- the clean MATCH example
MAROON = "#A4123F"
TEAL = "#1f6b62"

# a print-friendly diverging colormap: teal (identical) -> white -> maroon
# (very different) -- ties the figure's own palette back to the deck's
# amritaMaroon brand color, so it reads as designed-for-this-deck, not
# a generic export.
cmap = LinearSegmentedColormap.from_list("cost", [TEAL, "#f5f1ee", MAROON])


# =====================================================================
# Figure 1: Stage-1 cost matrix C, a legible 8-column slice
# =====================================================================
def make_cost_matrix_figure():
    row_labels = CAND["matrix_row_labels"]
    all_cols = CAND["matrix_col_labels"]
    matched_mask = CAND["matrix_col_matched"]
    full_matrix = np.array(CAND["cost_matrix"])

    # pick 8 columns: every matched column (the ones GNCCP actually
    # selected), plus a few unmatched ones for contrast -- real
    # selection, not cherry-picked to look good.
    matched_idx = [i for i, m in enumerate(matched_mask) if m]
    unmatched_idx = [i for i, m in enumerate(matched_mask) if not m]
    show_idx = sorted(matched_idx + unmatched_idx[:8 - len(matched_idx)])

    cols = [all_cols[i] for i in show_idx]
    col_matched = [matched_mask[i] for i in show_idx]
    matrix = full_matrix[:, show_idx]

    fig, ax = plt.subplots(figsize=(7.4, 3.0), dpi=220)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=6, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9, rotation=0)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9.5)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt_color = "white" if v > 4.2 or v < 1.2 else "#2a2a2a"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=8.6, color=txt_color)

    for j, m in enumerate(col_matched):
        if m:
            ax.get_xticklabels()[j].set_color(MAROON)
            ax.get_xticklabels()[j].set_fontweight("bold")

    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="#ddd6d0", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("dissimilarity  (0 = identical)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        f"Stage 1 cost matrix C — {CAND['name']}  "
        f"(8 of {CAND['n_candidate_atoms_in_cost_matrix']} candidate atoms shown; "
        f"maroon labels = atoms GNCCP actually matched)",
        fontsize=9.3, pad=10, loc="left", color="#2a2a2a",
    )
    fig.tight_layout()
    fig.savefig("mof_step2_costmatrix.png", bbox_inches="tight", facecolor="white")
    print("wrote mof_step2_costmatrix.png")


# =====================================================================
# Figure 2: X before (uniform) vs X after (0/1) -- Stage 1 optimization
# =====================================================================
def make_optimization_figure():
    row_labels = CAND["matrix_row_labels"]
    all_cols = CAND["matrix_col_labels"]
    matched_mask = CAND["matrix_col_matched"]
    matched_idx = [i for i, m in enumerate(matched_mask) if m]
    unmatched_idx = [i for i, m in enumerate(matched_mask) if not m]
    show_idx = sorted(matched_idx + unmatched_idx[:8 - len(matched_idx)])
    cols = [all_cols[i] for i in show_idx]

    x_init = np.array(CAND["x_initial"])[:, show_idx]
    x_final = np.array(CAND["x_final"])[:, show_idx]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.9), dpi=220)

    for ax, mat, title, vmax, fmt in [
        (axes[0], x_init, f"Before — uniform init  (every cell = {CAND['x_initial_value']:.5f})", x_init.max()*1.4, "{:.4f}"),
        (axes[1], x_final, "After — discretized X*  (exactly one 1 per row)", 1.0, "{:.0f}"),
    ]:
        cmap2 = LinearSegmentedColormap.from_list("x", ["#f5f1ee", MAROON])
        im = ax.imshow(mat, cmap=cmap2, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8, rotation=0)
        ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                         color="white" if v > vmax*0.55 else "#2a2a2a")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
        ax.grid(which="minor", color="#ddd6d0", linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    fig.savefig("mof_step3_optimization.png", bbox_inches="tight", facecolor="white")
    print("wrote mof_step3_optimization.png")


if __name__ == "__main__":
    make_cost_matrix_figure()
    make_optimization_figure()