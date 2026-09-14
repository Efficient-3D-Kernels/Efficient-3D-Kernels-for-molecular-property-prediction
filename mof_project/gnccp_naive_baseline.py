"""
gnccp_naive_baseline.py
=========================
The other half of the ablation. This file implements EXACTLY the same
Algorithm 1 (GNCCP + Frank-Wolfe) as gnccp_matcher.py -- same zeta
schedule, same Frank-Wolfe cap, same duality-gap early stopping, same
Sinkhorn projection, same Hungarian LMO and discretization -- but with
the two optimizations from your paper switched OFF:

  - U is built the naive way: U = X @ ones(N,N) @ X.T
    (O(MN^2 + M^2N), the cost Lemma 1 says this eliminates)
  - The structural gradient is NOT the closed form from Theorem 1.
    Two "no closed form" variants are provided:
      (a) naive_structural_gradient() -- same mathematical formula as
          Theorem 1, but built through the naive O(MN^2+M^2N) U instead
          of the rank-one shortcut. This isolates Lemma 1 ONLY, and is
          fast enough to run end-to-end on real QMOF candidates.
      (b) finite_difference_gradient() -- what you would have to do
          with NO derived closed form at all: central differences,
          2*M*N evaluations of f(X) per gradient. This isolates
          Theorem 1's contribution, but is only practical on small
          matrices (a handful of atoms per side) -- see the docstring
          on that function for why.

WHY THIS FILE MATTERS FOR YOUR DEFENSE: your paper's claim is about
REDUCING GNCCP's per-iteration cost, not replacing GNCCP with something
else. Running the SAME loop through (a) the naive path and (b) the
optimized path in gnccp_matcher.py, on the SAME candidates, is the only
comparison that actually measures Lemma 1 / Theorem 1's contribution.
Since both paths compute the same mathematical quantities, their outputs
(mapping, score) should agree -- if they don't, one of the two
implementations has a bug, and that needs to be found before either
number is trusted.
"""

from __future__ import annotations
import time
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment, minimize_scalar

from combinatorial_cost import build_cost_matrix, structural_consistency
from gnccp_matcher import (
    build_adjacency,
    normalize_adjacency,
    J_zeta,
    grad_J_zeta,
    init_X,
    fw_lmo,
    fw_gap,
    sinkhorn_project,
    similarity_kernel,
)


# =====================================================================
# Naive U (no Lemma 1)
# =====================================================================

def naive_U(X: np.ndarray) -> np.ndarray:
    """U = X 1_{NxN} X^T, built literally as the paper's Eq. (4) is
    written, with no rank-one shortcut. O(MN^2 + M^2N): X @ ones(N,N)
    costs O(M*N*N), the result @ X.T costs O(M*N*M). This is exactly
    the operation Lemma 1's Corollary 1 says is eliminated."""
    N = X.shape[1]
    return X @ np.ones((N, N)) @ X.T


# =====================================================================
# Naive objective / gradient -- same formula as Theorem 1, but routed
# through naive_U (isolates Lemma 1's contribution only)
# =====================================================================

def naive_wcs_objective(X, AG, AH, C, alpha):
    U = naive_U(X)
    S = X @ AH @ X.T
    R = (U * AG) - S
    f = float(np.sum(R * R))
    F = alpha * f + (1 - alpha) * float(np.sum(C * X))
    return F, R


def naive_structural_gradient(X, AG, AH, R):
    """Same closed-form formula as Theorem 1 (Eq. 7-8) -- the gradient
    DERIVATION is still correct and still used -- but the rank-one
    shortcut inside it is not applied: the first term is built as the
    literal (R∘AG) X 1_{NxN} matrix product rather than the row-sum
    broadcast Eq. (8) reduces it to. This isolates Lemma 1's cost ONLY;
    Theorem 1's derivation itself is not being ablated here (see
    finite_difference_gradient for that)."""
    M, N = X.shape
    T = (R * AG) @ X                    # (M, N)
    term1 = T @ np.ones((N, N))         # naive: full N x N ones matrix, O(MN^2)
    term2 = R @ X @ AH
    return 4.0 * term1 - 4.0 * term2


def naive_grad_F(X, AG, AH, C, alpha, R):
    return alpha * naive_structural_gradient(X, AG, AH, R) + (1 - alpha) * C


# =====================================================================
# Finite-difference gradient -- "what if Theorem 1 didn't exist"
# =====================================================================

def finite_difference_gradient(X, AG, AH, C, alpha, h=1e-5):
    """
    Central-difference numerical gradient of F(X) w.r.t. every entry of
    X. This is what an implementation WITHOUT Theorem 1's derived
    closed form would have to fall back to.

    COST WARNING: this evaluates wcs_objective (an O(MN^2+M^2N) or
    O(MN+M^2+M^2N) operation depending on which U it uses) 2*M*N times
    -- once per matrix entry, forward and backward. For a 5x100 query-
    candidate pair that is already 1000 full objective evaluations PER
    GRADIENT, PER Frank-Wolfe iteration, PER zeta step. Use this only
    for the small-scale microbenchmark in benchmark_optimizations.py,
    never inside a full pipeline run over thousands of candidates --
    that is precisely the point being demonstrated: without Theorem 1,
    GNCCP is not practical at dataset scale.
    """
    from gnccp_matcher import wcs_objective  # optimized U, so we're isolating ONLY the gradient method
    grad = np.zeros_like(X)
    M, N = X.shape
    for i in range(M):
        for j in range(N):
            Xp = X.copy(); Xp[i, j] += h
            Xm = X.copy(); Xm[i, j] -= h
            Fp, _ = wcs_objective(Xp, AG, AH, C, alpha)
            Fm, _ = wcs_objective(Xm, AG, AH, C, alpha)
            grad[i, j] = (Fp - Fm) / (2 * h)
    return grad


# =====================================================================
# Naive line search (identical to gnccp_matcher's, just calling the
# naive objective so timing is apples-to-apples)
# =====================================================================

def _naive_line_search(X, Y, AG, AH, C, alpha, zeta):
    D = Y - X

    def obj(lam):
        Xl = X + lam * D
        F_val, _ = naive_wcs_objective(Xl, AG, AH, C, alpha)
        return J_zeta(F_val, Xl, zeta)

    res = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded",
                           options={"xatol": 1e-3, "maxiter": 25})
    return float(np.clip(res.x, 0.0, 1.0))


def _mapping_to_X(mapping, q_nodes, h_nodes):
    q_idx = {n: i for i, n in enumerate(q_nodes)}
    h_idx = {n: i for i, n in enumerate(h_nodes)}
    X = np.zeros((len(q_nodes), len(h_nodes)))
    for u, v in mapping.items():
        X[q_idx[u], h_idx[v]] = 1.0
    return X


# =====================================================================
# The naive Algorithm 1 -- structurally IDENTICAL to gnccp_matcher.gnccp_match
# =====================================================================

def gnccp_match_naive(
    Gq: nx.Graph,
    Hc: nx.Graph,
    alpha: float = 0.5,
    gamma: float = 0.5,
    dzeta: float = 0.1,
    TFW: int = 15,
    fw_eps: float = 1e-4,
    sinkhorn_iters: int = 20,
    w_elem: float = 3.0,
    w_deg: float = 1.0,
    w_hist: float = 1.5,
    verbose: bool = False,
):
    """
    Line-for-line the same driver as gnccp_matcher.gnccp_match (copy the
    two side by side to confirm) -- same init, same zeta schedule, same
    FW cap, same duality-gap stopping, same Sinkhorn projection, same
    final Hungarian discretization. The ONLY substitutions are
    naive_wcs_objective / naive_grad_F in place of the optimized
    versions. Any timing difference between this function and
    gnccp_matcher.gnccp_match is attributable ONLY to Lemma 1's rank-one
    U trick (Theorem 1's derivation is used in both).
    """
    t0 = time.perf_counter()

    q_nodes = list(Gq.nodes())
    h_nodes = list(Hc.nodes())
    M, N = len(q_nodes), len(h_nodes)
    L = M
    if M > N:
        raise ValueError(f"Query has {M} atoms but candidate only has {N}.")

    AG = normalize_adjacency(build_adjacency(Gq, q_nodes))
    AH = normalize_adjacency(build_adjacency(Hc, h_nodes))

    C_raw, _, _ = build_cost_matrix(Gq, Hc, w_elem=w_elem, w_deg=w_deg, w_hist=w_hist)
    c_min, c_max = C_raw.min(), C_raw.max()
    C = (C_raw - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(C_raw)

    X = init_X(M, N, L)
    zeta = 1.0

    n_outer, n_fw_total = 0, 0
    while zeta >= -1.0 - 1e-9:
        n_outer += 1
        for t in range(TFW):
            n_fw_total += 1
            F_val, R = naive_wcs_objective(X, AG, AH, C, alpha)
            gF = naive_grad_F(X, AG, AH, C, alpha, R)
            gJ = grad_J_zeta(gF, X, zeta)

            Y = fw_lmo(gJ, L)
            gap = fw_gap(gJ, X, Y)
            if gap < fw_eps:
                break

            lam = _naive_line_search(X, Y, AG, AH, C, alpha, zeta)
            X = sinkhorn_project(X + lam * (Y - X), L, iters=sinkhorn_iters)

        if verbose:
            print(f"  [naive] zeta={zeta:+.2f}  FW iters={t+1:2d}  gap={gap:.3e}")
        zeta -= dzeta

    row_ind, col_ind = linear_sum_assignment(-X)
    mapping = {q_nodes[r]: h_nodes[c] for r, c in zip(row_ind, col_ind)}

    F_star, _ = naive_wcs_objective(_mapping_to_X(mapping, q_nodes, h_nodes), AG, AH, C, alpha)
    k = similarity_kernel(F_star, L, gamma)
    edge_score = structural_consistency(Gq, Hc, mapping)

    runtime = time.perf_counter() - t0
    return {
        "mapping": mapping,
        "score": k,
        "F_star": F_star,
        "edge_consistency": edge_score,
        "n_outer_steps": n_outer,
        "n_fw_iters_total": n_fw_total,
        "runtime_sec": runtime,
    }