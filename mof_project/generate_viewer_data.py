"""
generate_viewer_data.py
=========================
Runs the ACTUAL (corrected) pipeline -- gnccp_matcher.gnccp_match +
geometric_refinement.stage2_refine -- on a handful of real QMOF
candidates, and writes a single JSON blob the HTML viewer embeds
directly. Nothing in the viewer is fabricated or hand-typed; every
number and every atom position on screen came out of this script.

NOTE ON THE QUERY MOTIF: this sandbox does not have your real
data/UiO-66.cif on disk, only the QMOF cache. So the query motif here
is a genuine Zr-O-C-O-Zr bridge extracted from a real QMOF structure in
the cache (same extractor, same chemistry, same code path
mof_graph.find_bridging_carboxylate_path you use everywhere else) --
it is NOT literally your UiO-66.cif file. When you run this on your
own machine, swap in the block marked "QUERY (swap this on your
machine)" below to load your real UiO-66.cif via mof_graph.load_structure
instead, and rerun -- the rest of the script and the viewer need no
changes.
"""

import json
import networkx as nx
import numpy as np

from dataset_loader import load_graph_cache
from mof_graph import find_bridging_carboxylate_path
from family_labels import primary_metal, heuristic_label
from gnccp_matcher import gnccp_match, init_X
from combinatorial_cost import build_cost_matrix
from geometric_refinement import stage2_refine
from threshold_calibration import stage2_verdict

QUERY_METAL = "Zr"
GNCCP_KWARGS = dict(dzeta=0.2, TFW=10)  # fixed for every candidate below -- no cherry-picked settings per candidate

# Hand-picked, from a real scan of 60 same-metal Zr candidates + a few
# other-metal ones, spanning the full RMSD range that scan produced --
# not cherry-picked to only show flattering results. See chat: one of
# these (qmof-0147aef.cif) landed on a DIFFERENT local optimum than an
# earlier run with different dzeta/TFW settings, which is itself an
# honest finding about GNCCP's non-convexity worth keeping visible.
CURATED_CANDIDATES = [
    "qmof-0bbf75f.cif",  # same metal, near-perfect 3D overlay (RMSD ~0.07 A)
    "qmof-0147aef.cif",  # same metal, near-perfect 3D overlay (RMSD ~0.07 A)
    "qmof-1e0b61c.cif",  # same metal, borderline (RMSD ~2.2 A)
    "qmof-15c314b.cif",  # same metal, same family but different 3D geometry (RMSD ~3.6 A)
    "qmof-2d90005.cif",  # same metal, clear 3D decoy (RMSD ~13.2 A)
    "qmof-0000295.cif",  # different metal (Cu) -- true negative
    "qmof-0001b0d.cif",  # different metal (Zn) -- true negative
]

ELEMENT_COLORS = {
    "Zr": "#5fd0c0", "Zn": "#7aa8ff", "Cu": "#e08a5f", "Al": "#c9c46a",
    "Fe": "#e0715f", "Co": "#b98ae0", "Ni": "#7fd0e0", "Mg": "#9be07f",
    "O": "#ff6b6b", "C": "#c9ccd1", "H": "#e8e8ea", "N": "#6b8fff",
}
DEFAULT_COLOR = "#9aa0ab"


def color_for(elem):
    return ELEMENT_COLORS.get(elem, DEFAULT_COLOR)


def main():
    print("Loading QMOF cache ...")
    graphs = load_graph_cache("qmof_graph_cache.json")
    print(f"Loaded {len(graphs)} structures.")

    # ----------------------------------------------------------------
    # QUERY (swap this on your machine for your real UiO-66.cif, via
    # mof_graph.load_structure + build_bonded_graph + this same
    # find_bridging_carboxylate_path call)
    # ----------------------------------------------------------------
    Gq, query_source = None, None
    for name, G in graphs.items():
        sub, path = find_bridging_carboxylate_path(G, QUERY_METAL)
        if sub is not None:
            Gq, query_source, query_path = sub, name, path
            break
    if Gq is None:
        raise RuntimeError(f"No {QUERY_METAL} bridge found in the cache.")
    print(f"Query motif from {query_source}: {'-'.join(Gq.nodes[n]['element'] for n in query_path)}")

    # simple fixed linear layout for the 5-atom query chain, IN THE ACTUAL
    # M-O-C-O-M PATH ORDER (not networkx's arbitrary node iteration order --
    # that bug would have scrambled the "left to right" reading of the diagram)
    q_nodes = list(Gq.nodes())
    query_layout = {n: (i, 0) for i, n in enumerate(query_path)}

    # ----------------------------------------------------------------
    # curated candidate list -- see CURATED_CANDIDATES comment above
    # ----------------------------------------------------------------
    candidates = [n for n in CURATED_CANDIDATES if n in graphs and n != query_source]
    missing = [n for n in CURATED_CANDIDATES if n not in graphs]
    if missing:
        print(f"WARNING: not found in cache, skipped: {missing}")
    print(f"Candidates: {len(candidates)}")

    # ----------------------------------------------------------------
    # run the real pipeline on each candidate
    # ----------------------------------------------------------------
    results = []
    for name in candidates:
        Hc = graphs[name]
        r1 = gnccp_match(Gq, Hc, **GNCCP_KWARGS)
        label = heuristic_label(QUERY_METAL, Hc)

        mapping = r1["mapping"]
        matched_cand_nodes = set(mapping.values())

        # local subgraph for display: matched atoms + their 1-hop neighbors
        display_nodes = set(matched_cand_nodes)
        for n in matched_cand_nodes:
            display_nodes.update(Hc.neighbors(n))
        Hc_local = Hc.subgraph(display_nodes).copy()
        local_order = list(Hc_local.nodes())  # fixes column order for graph AND matrices below

        # 2D layout for the local subgraph (spring layout, deterministic seed)
        pos2d = nx.spring_layout(Hc_local, seed=0, k=0.9)

        # ------------------------------------------------------------
        # STEP 2 data: the real cost matrix C (Definition 1) -- built
        # against the FULL candidate (N_full columns), then sliced down
        # to just the columns we're displaying, for legibility. The
        # values themselves are untouched -- same numbers the real
        # algorithm actually used.
        # ------------------------------------------------------------
        C_full, q_nodes_cost, h_nodes_cost = build_cost_matrix(Gq, Hc)
        q_row_of = {n: i for i, n in enumerate(q_nodes_cost)}
        h_col_of = {n: i for i, n in enumerate(h_nodes_cost)}
        row_order = [n for n in query_path if n in q_row_of]  # display rows in chemical M-O-C-O-M order

        cost_matrix_display = [
            [round(float(C_full[q_row_of[qn], h_col_of[cn]]), 2) for cn in local_order]
            for qn in row_order
        ]

        # ------------------------------------------------------------
        # STEP 3 data: X before optimization (uniform, Proposition 1)
        # and X after optimization (the discretized 0/1 mapping) --
        # both sliced to the same displayed columns for a fair before/after.
        # X_initial's true value uses the REAL N_full (not just the
        # columns shown), so the number is honest even though the grid
        # is cropped for display.
        # ------------------------------------------------------------
        M_full = len(q_nodes_cost)
        N_full = len(h_nodes_cost)
        L_full = M_full
        uniform_value = round(float(L_full / (M_full * N_full)), 5)
        x_initial_display = [[uniform_value for _ in local_order] for _ in row_order]

        x_final_display = [
            [1 if mapping.get(qn) == cn else 0 for cn in local_order]
            for qn in row_order
        ]

        r2 = None
        verdict = None
        if len(mapping) >= 3:
            r2 = stage2_refine(Gq, Hc, mapping, k2D=r1["score"])
            verdict = stage2_verdict(r2["motif_score"], r2["rmsd"], conservative=False)

        results.append({
            "name": name,
            "metal": primary_metal(Hc),
            "same_metal_as_query": bool(primary_metal(Hc) == QUERY_METAL),
            "heuristic_label": int(label),
            "n_atoms_full_structure": Hc.number_of_nodes(),
            "n_candidate_atoms_in_cost_matrix": N_full,
            "score_2d": r1["score"],
            "edge_consistency": r1["edge_consistency"],
            "n_fw_iters": r1["n_fw_iters_total"],
            "rmsd": (None if r2 is None or not np.isfinite(r2["rmsd"]) else round(r2["rmsd"], 4)),
            "motif_score_3d": (None if r2 is None or not np.isfinite(r2["motif_score"]) else round(r2["motif_score"], 4)),
            "k3d": (None if r2 is None else round(r2["k3D"], 4)),
            "hybrid_score": (None if r2 is None or "hybrid" not in r2 else round(r2["hybrid"], 4)),
            "verdict": verdict,
            "mapping": {str(q): str(c) for q, c in mapping.items()},
            "matrix_row_labels": [f"{Gq.nodes[qn]['element']} (#{qn})" for qn in row_order],
            "matrix_col_labels": [f"{Hc_local.nodes[cn]['element']} (#{cn})" for cn in local_order],
            "matrix_col_matched": [bool(cn in matched_cand_nodes) for cn in local_order],
            "cost_matrix": cost_matrix_display,
            "cost_matrix_note": f"5 x {N_full} in the real run (w_elem=3.0, w_deg=1.0, w_hist=1.5); showing 5 x {len(local_order)} here (matched atoms + their bonded neighbors)",
            "x_initial": x_initial_display,
            "x_initial_value": uniform_value,
            "x_final": x_final_display,
            "candidate_graph": {
                "nodes": [
                    {
                        "id": str(n),
                        "element": Hc_local.nodes[n]["element"],
                        "color": color_for(Hc_local.nodes[n]["element"]),
                        "x": round(float(pos2d[n][0]), 4),
                        "y": round(float(pos2d[n][1]), 4),
                        "matched": bool(n in matched_cand_nodes),
                    }
                    for n in local_order
                ],
                "edges": [[str(u), str(v)] for u, v in Hc_local.edges()],
            },
        })
        print(f"  {name:20s} metal={primary_metal(Hc)!s:4s} score={r1['score']:.3f} "
              f"rmsd={r2['rmsd'] if r2 else float('nan'):.3f} verdict={verdict}")

    payload = {
        "query_source_note": (
            f"Query motif extracted from real QMOF structure {query_source} "
            f"(stand-in for UiO-66.cif in this sandbox -- same extractor, same "
            f"chemistry; swap in your real CIF on your machine, see script docstring)"
        ),
        "query_metal": QUERY_METAL,
        "query_graph": {
            "nodes": [
                {
                    "id": str(n),
                    "element": Gq.nodes[n]["element"],
                    "color": color_for(Gq.nodes[n]["element"]),
                    "x": query_layout[n][0],
                    "y": query_layout[n][1],
                }
                for n in q_nodes
            ],
            "edges": [[str(u), str(v)] for u, v in Gq.edges()],
        },
        "candidates": results,
    }

    with open("viewer_data.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\nWrote viewer_data.json")


if __name__ == "__main__":
    main()