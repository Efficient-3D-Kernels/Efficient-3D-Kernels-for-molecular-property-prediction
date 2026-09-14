"""
combinatorial_cost.py
=======================
This is NOT a matcher. It is the small, shared piece of both
gnccp_matcher.py and gnccp_naive_baseline.py that builds the atom-feature
dissimilarity matrix C (Definition 1 of the theory paper: "a cost matrix
C encodes pairwise atom-feature dissimilarity"). The paper deliberately
leaves C's construction to the implementer -- this is the same 2D
feature set (element mismatch + degree difference + neighbor-element-
histogram mismatch) used throughout this project, kept in one place so
both matchers score chemical dissimilarity identically and any
score/mapping difference between them is attributable ONLY to the
optimizer, not to a different notion of "similar atom."

(This file used to live inside combinatorial_matcher.py, which also
contained a standalone one-shot-Hungarian matcher that is NOT part of
your proposed method and has been removed. Only the cost-matrix and
edge-consistency utilities below survive that removal.)
"""

from __future__ import annotations
import numpy as np
import networkx as nx


def _neighbor_histogram(G: nx.Graph, node) -> dict:
    hist = {}
    for nb in G.neighbors(node):
        e = G.nodes[nb]["element"]
        hist[e] = hist.get(e, 0) + 1
    return hist


def _hist_mismatch(h1: dict, h2: dict) -> float:
    keys = set(h1) | set(h2)
    if not keys:
        return 0.0
    diff = sum(abs(h1.get(k, 0) - h2.get(k, 0)) for k in keys)
    denom = sum(h1.values()) + sum(h2.values())
    return diff / denom if denom else 0.0


def build_cost_matrix(
    Gq: nx.Graph, Hc: nx.Graph, w_elem=3.0, w_deg=1.0, w_hist=1.5
) -> tuple:
    """C_ij = w_elem*element_mismatch + w_deg*degree_diff + w_hist*neighbor_hist_mismatch,
    for every query atom i and candidate atom j. Pure 2D/topological
    information -- no coordinates used anywhere here."""
    q_nodes = list(Gq.nodes())
    h_nodes = list(Hc.nodes())
    max_deg = max(
        max((d for _, d in Gq.degree()), default=1),
        max((d for _, d in Hc.degree()), default=1),
        1,
    )
    q_hist = {u: _neighbor_histogram(Gq, u) for u in q_nodes}
    h_hist = {v: _neighbor_histogram(Hc, v) for v in h_nodes}

    C = np.zeros((len(q_nodes), len(h_nodes)))
    for a, u in enumerate(q_nodes):
        eu = Gq.nodes[u]["element"]
        du = Gq.degree(u)
        for b, v in enumerate(h_nodes):
            ev = Hc.nodes[v]["element"]
            dv = Hc.degree(v)
            elem_cost = 0.0 if eu == ev else 1.0
            deg_cost = abs(du - dv) / max_deg
            hist_cost = _hist_mismatch(q_hist[u], h_hist[v])
            C[a, b] = w_elem * elem_cost + w_deg * deg_cost + w_hist * hist_cost
    return C, q_nodes, h_nodes


def structural_consistency(Gq: nx.Graph, Hc: nx.Graph, mapping: dict) -> float:
    """Fraction of query edges whose matched image is also an edge in
    the candidate graph -- a simple, cheap sanity diagnostic reported
    alongside the GNCCP score, not part of the score itself."""
    if Gq.number_of_edges() == 0:
        return 1.0
    preserved = 0
    for u, v in Gq.edges():
        if u in mapping and v in mapping:
            if Hc.has_edge(mapping[u], mapping[v]):
                preserved += 1
    return preserved / Gq.number_of_edges()