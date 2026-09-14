"""
gnccp_matcher.py
=================
THIS IS YOUR PROPOSED METHOD -- the optimized GNCCP described in your
"Mathematical Foundations of the WCS-GNCCP Molecular Graph Kernel" paper:
the same GNCCP continuation (Graduated Non-Convexity and Concavity
Procedure, zeta swept from +1 to -1, Frank-Wolfe inner solve at every zeta
step, Hungarian as the FW linear oracle) as the original 2018 WCS paper,
but with two algebraic optimizations that reduce the cost of EVERY
iteration:

  - Lemma 1: the co-matching matrix U is built from its rank-one identity
    (r r^T) instead of via the full N x N all-ones matrix -- this alone
    drops U's cost from O(MN^2 + M^2N) to O(MN + M^2).
  - Theorem 1: the structural gradient has a closed form (using Lemma 1's
    rank-one trick inside it too) instead of needing numerical / symbolic
    differentiation of the Hadamard + bilinear objective every iteration.

This module is a faithful, checkable transcription of the paper -- every
function below is tagged with the exact Definition / Proposition / Theorem
it implements, so you (and your panel) can put the paper and this file
side by side.

--------------------------------------------------------------------
How to verify "more optimized than GNCCP" with THIS codebase
--------------------------------------------------------------------
Your claim is an ablation, not a different-algorithm comparison: run the
SAME Algorithm 1 loop (same zeta schedule, same TFW cap, same duality-gap
stopping) once with these optimizations and once without them
(gnccp_naive_baseline.py implements the "without" side -- naive U, no
closed-form gradient). Because it's the same algorithm, naive and
optimized should produce matching (or near-identical) mappings/scores --
only the wall-clock time should differ. benchmark_optimizations.py runs
exactly that ablation, on real QMOF candidates.

There is no more "one-shot Hungarian vs full GNCCP" comparison in this
codebase -- that was comparing two structurally different algorithms and
does not support your actual claim. combinatorial_matcher.py has been
removed for this reason.

--------------------------------------------------------------------
What is implemented, mapped to the paper (document: Mathematical
Foundations of the WCS-GNCCP Molecular Graph Kernel)
--------------------------------------------------------------------
  Definition 1  -> PL / DL encoded implicitly via the row/col-cap + total-sum
                   Sinkhorn projection (Prop. 5)
  Definition 2  -> WCS objective F(X)                         -> wcs_objective()
  Lemma 1       -> U = X 1_{NxN} X^T = r r^T (rank-one)        -> co_matching_U()
  Theorem 1     -> closed-form structural gradient, using the
                   rank-one identity (8) to avoid ever forming
                   the full N x N all-ones matrix              -> grad_wcs_objective()
  Proposition 1 -> GNCCP endpoint behaviour (used to justify
                   the uniform-matrix initialization X0)       -> init_X()
  Proposition 2 -> gradient of J_zeta                          -> grad_J_zeta()
  Definition 4  -> Frank-Wolfe linear minimization oracle,
                   solved by (rectangular) Hungarian            -> fw_lmo()
  Theorem 2     -> Frank-Wolfe duality gap, used for early
                   termination of the inner loop                -> fw_gap()
  Proposition 4 -> Frobenius normalization of AG, AH             -> normalize_adjacency()
  Proposition 5 -> Sinkhorn-Knopp projection onto DL             -> sinkhorn_project()
  Proposition 3 -> exponential similarity kernel k(G,H)          -> similarity_kernel()
  Theorem 4 /
  Corollary 3   -> Stage-2 Kabsch alignment reduced to O(L)      -> (delegated to
                                                                     geometric_refinement.kabsch_rmsd,
                                                                     reused unchanged since Stage 1's
                                                                     bijective map is exactly what
                                                                     Theorem 4 requires)
  Algorithm 1   -> the full outer/inner loop                     -> gnccp_match()
  Theorem 3 /
  Corollary 2   -> O(Tzeta * TFW * n^3) total cost -- verified
                   EMPIRICALLY (not just asserted) by
                   run_gnccp_vs_combinatorial.py, which times
                   this function against combinatorial_matcher's
                   one-shot Hungarian on the same graph pairs.

No step here is a shortcut around the paper: the rank-one trick (Lemma 1)
and the closed-form gradient (Theorem 1) are exactly what let GNCCP itself
run in O(n^3) per Frank-Wolfe iteration instead of O(n^4)-ish if you built
U and the gradient naively -- but GNCCP still needs Tzeta*TFW such
iterations, one full Hungarian LMO each, which is the whole point of the
comparison.
"""

from __future__ import annotations
import time
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment, minimize_scalar

from combinatorial_cost import build_cost_matrix, structural_consistency


# =====================================================================
# Graph -> matrix plumbing
# =====================================================================

def build_adjacency(G: nx.Graph, nodes: list) -> np.ndarray:
    """
    Weighted adjacency matrix A_G in the paper's sense (Sec. I). Our MOF
    graphs (mof_graph.py) do not carry bond order/type (single/double/
    aromatic), only bonded/not-bonded from a covalent-radius cutoff, so the
    honest weight here is binary presence -- exactly the same information
    Stage 1 of your combinatorial matcher is allowed to see (2D topology
    only, no coordinates). If you later add bond-order perception, plug the
    weight in here; nothing else in this file needs to change.
    """
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))
    for u, v in G.edges():
        if u in idx and v in idx:
            i, j = idx[u], idx[v]
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Proposition 4: A_hat = A / ||A||_F. Keeps f_hat(X) <= 2 for all
    X in D_L regardless of bond count, so the structural term (which
    otherwise scales with the number of bonds -- see Remark 6) stays
    comparable in magnitude to the O(L)-bounded appearance term
    tr(C^T X), and alpha actually does the balancing job it's meant to."""
    norm = np.linalg.norm(A, ord="fro")
    if norm < 1e-12:
        return A.copy()
    return A / norm


# =====================================================================
# Lemma 1 -- rank-one factorization of U
# =====================================================================

def co_matching_U(X: np.ndarray) -> np.ndarray:
    """
    Lemma 1: U = X 1_{NxN} X^T = r r^T, r = X 1_N.

    We NEVER materialize the N x N all-ones matrix (that would be the
    O(MN^2 + M^2N) naive route the paper explicitly flags as the thing
    Lemma 1 avoids) -- we go straight to the O(MN + M^2) route via the
    row-sum vector r.
    """
    r = X.sum(axis=1)  # O(MN)
    return np.outer(r, r)  # O(M^2)


# =====================================================================
# Definition 2 / Theorem 1 -- WCS objective and its gradient
# =====================================================================

def wcs_objective(X: np.ndarray, AG: np.ndarray, AH: np.ndarray, C: np.ndarray, alpha: float):
    """F(X) = alpha * f(X) + (1-alpha) * tr(C^T X),  f(X) = ||U∘AG - X AH X^T||_F^2
    (Definition 2, Eq. 2-3). Returns (F, R) where R is the residual
    U∘AG - X AH X^T, reused by the gradient so we don't recompute it."""
    U = co_matching_U(X)
    S = X @ AH @ X.T  # the O(M^2 N) bilinear form -- Theorem 3's dominant term (c)
    R = (U * AG) - S
    f = float(np.sum(R * R))
    F = alpha * f + (1 - alpha) * float(np.sum(C * X))
    return F, R


def grad_wcs_objective(X: np.ndarray, AG: np.ndarray, AH: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Theorem 1 (Eq. 7-8): grad_X f = 4 (R∘AG) X 1_{NxN} - 4 R X A_H,
    where the first term is computed via its rank-one form (Eq. 8):

        (R∘AG) X 1_{NxN} = [ (R∘AG) X ] 1_N 1_N^T

    i.e. each row i of the result is the constant  sum_j [(R∘AG)X]_{ij} ,
    repeated across all N columns. This is exactly the same rank-one
    trick as Lemma 1 applied to the gradient, and it's what keeps this
    step at O(M^2 N) instead of needing the literal N x N all-ones matrix.
    """
    M, N = X.shape
    T = (R * AG) @ X  # (M, N), O(M^2 N)... actually O(M*M*N)=O(M^2 N)
    row_sums = T.sum(axis=1, keepdims=True)  # (M, 1)
    term1 = np.repeat(row_sums, N, axis=1)  # rank-one broadcast, Eq. (8)
    term2 = R @ X @ AH  # (M, N)
    return 4.0 * term1 - 4.0 * term2


def grad_F(X, AG, AH, C, alpha, R):
    return alpha * grad_wcs_objective(X, AG, AH, R) + (1 - alpha) * C


# =====================================================================
# Proposition 2 -- gradient of J_zeta ; Definition 3 -- J_zeta itself
# =====================================================================

def J_zeta(F_val: float, X: np.ndarray, zeta: float) -> float:
    """Definition 3 (Eq. 13)."""
    xx = float(np.sum(X * X))
    if zeta >= 0:
        return (1 - zeta) * F_val + zeta * xx
    else:
        return (1 + zeta) * F_val + zeta * xx


def grad_J_zeta(gradF: np.ndarray, X: np.ndarray, zeta: float) -> np.ndarray:
    """Proposition 2 (Eq. 14)."""
    if zeta >= 0:
        return (1 - zeta) * gradF + 2 * zeta * X
    else:
        return (1 + zeta) * gradF + 2 * zeta * X


# =====================================================================
# Proposition 1 -- initialization at zeta = +1
# =====================================================================

def init_X(M: int, N: int, L: int) -> np.ndarray:
    """Proposition 1(1): the unique minimizer of J_{+1} = ||X||_F^2 over
    D_L is the uniform matrix X0 = (L / (M*N)) * 1_{MxN}."""
    return np.full((M, N), L / (M * N))


# =====================================================================
# Definition 4 -- Frank-Wolfe linear minimization oracle (Hungarian)
# =====================================================================

def fw_lmo(grad: np.ndarray, L: int) -> np.ndarray:
    """
    Y = argmin_{Y in D_L} <grad, Y>.

    The minimum of a linear form over a polytope is attained at a vertex,
    i.e. Y is a genuine partial permutation matrix (Y in P_L, not just
    D_L) -- found by the Hungarian algorithm. For our use, L = M (every
    query atom must be matched, M <= N), so this is the plain rectangular
    Hungarian assignment your paper cites as the L = M <= N special case
    (Sec. II-D): O(M^2 N).
    """
    M, N = grad.shape
    row_ind, col_ind = linear_sum_assignment(grad)  # minimizes sum of grad[row,col]
    Y = np.zeros_like(grad)
    Y[row_ind[:L], col_ind[:L]] = 1.0
    return Y


# =====================================================================
# Theorem 2 -- Frank-Wolfe duality gap
# =====================================================================

def fw_gap(grad: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    """g_t = <grad J_zeta(X_t), X_t - Y_t>  (Eq. 16)."""
    return float(np.sum(grad * (X - Y)))


# =====================================================================
# Proposition 5 -- Sinkhorn-Knopp projection onto D_L
# =====================================================================

def sinkhorn_project(X: np.ndarray, L: int, iters: int = 20, tol: float = 1e-9) -> np.ndarray:
    """Steps 1-4 of Proposition 5, iterated to convergence (or `iters`
    rounds, whichever first)."""
    X = np.clip(X, 0, None)
    for _ in range(iters):
        total = X.sum()
        if total > 1e-12:
            X = X * (L / total)
        row_sums = X.sum(axis=1, keepdims=True)
        over_r = row_sums[:, 0] > 1.0
        if np.any(over_r):
            X[over_r] = X[over_r] / row_sums[over_r]
        col_sums = X.sum(axis=0, keepdims=True)
        over_c = col_sums[0, :] > 1.0
        if np.any(over_c):
            X[:, over_c] = X[:, over_c] / col_sums[:, over_c]
        # convergence check
        row_ok = np.all(X.sum(axis=1) <= 1 + tol)
        col_ok = np.all(X.sum(axis=0) <= 1 + tol)
        sum_ok = abs(X.sum() - L) < tol * max(1, L)
        if row_ok and col_ok and sum_ok:
            break
    return X


# =====================================================================
# Line search (paper: "inexact line search, e.g. backtracking", Eq. 8
# of the IEEE WCS paper / Eq. 12 discussion in your theory paper)
# =====================================================================

def _line_search(X, Y, AG, AH, C, alpha, zeta):
    D = Y - X

    def obj(lam):
        Xl = X + lam * D
        F_val, _ = wcs_objective(Xl, AG, AH, C, alpha)
        return J_zeta(F_val, Xl, zeta)

    res = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded",
                           options={"xatol": 1e-3, "maxiter": 25})
    return float(np.clip(res.x, 0.0, 1.0))


# =====================================================================
# Proposition 3 -- exponential similarity kernel
# =====================================================================

def similarity_kernel(F_star: float, L: int, gamma: float = 0.5) -> float:
    """k(G,H) = exp(-gamma * F*/L)  (Eq. 19)."""
    return float(np.exp(-gamma * F_star / max(L, 1)))


# =====================================================================
# Algorithm 1 -- full GNCCP + Frank-Wolfe driver
# =====================================================================

def gnccp_match(
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
    Runs Algorithm 1 of the theory paper end to end: zeta swept from +1 to
    -1 in steps of `dzeta`, each zeta solved by up to `TFW` Frank-Wolfe
    iterations (early-terminated via the duality gap, Theorem 2), final
    discretization by Hungarian(-X).

    Returns a dict with the SAME shape as combinatorial_matcher.stage1_match
    (mapping / score / runtime_sec / ...) plus GNCCP-specific diagnostics
    (n_outer_steps, n_fw_iters_total), so the two can be swapped into
    run_mof_screening.py / run_threshold_calibration.py interchangeably,
    and so their iteration counts can be reported side by side.
    """
    t0 = time.perf_counter()

    q_nodes = list(Gq.nodes())
    h_nodes = list(Hc.nodes())
    M, N = len(q_nodes), len(h_nodes)
    L = M  # PIW case: every query atom must be matched, M <= N (Sec. II-D)
    if M > N:
        raise ValueError(
            f"Query motif has {M} atoms but candidate only has {N}; "
            f"GNCCP here assumes L = M <= N (the case your pipeline uses)."
        )

    # --- adjacency (Sec. I) + Frobenius normalization (Proposition 4) ---
    AG = normalize_adjacency(build_adjacency(Gq, q_nodes))
    AH = normalize_adjacency(build_adjacency(Hc, h_nodes))

    # --- cost matrix (reuse the exact same 2D features as Stage 1's
    #     one-shot matcher, so score differences reflect the OPTIMIZER,
    #     not a different notion of atom dissimilarity) ---
    C_raw, _, _ = build_cost_matrix(Gq, Hc, w_elem=w_elem, w_deg=w_deg, w_hist=w_hist)
    c_min, c_max = C_raw.min(), C_raw.max()
    C = (C_raw - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(C_raw)

    # --- Proposition 1: initialize at the zeta=+1 minimizer ---
    X = init_X(M, N, L)
    zeta = 1.0

    n_outer, n_fw_total = 0, 0
    while zeta >= -1.0 - 1e-9:
        n_outer += 1
        for t in range(TFW):
            n_fw_total += 1
            F_val, R = wcs_objective(X, AG, AH, C, alpha)
            gF = grad_F(X, AG, AH, C, alpha, R)
            gJ = grad_J_zeta(gF, X, zeta)

            Y = fw_lmo(gJ, L)  # Definition 4, via Hungarian -- the expensive step
            gap = fw_gap(gJ, X, Y)  # Theorem 2

            if gap < fw_eps:
                break

            lam = _line_search(X, Y, AG, AH, C, alpha, zeta)
            X = sinkhorn_project(X + lam * (Y - X), L, iters=sinkhorn_iters)  # Proposition 5

        if verbose:
            print(f"  zeta={zeta:+.2f}  FW iters={t+1:2d}  gap={gap:.3e}")
        zeta -= dzeta

    # --- discretize: X* = Hungarian(-X), Algorithm 1 line 16 ---
    row_ind, col_ind = linear_sum_assignment(-X)
    mapping = {q_nodes[r]: h_nodes[c] for r, c in zip(row_ind, col_ind)}

    F_star, _ = wcs_objective(
        _mapping_to_X(mapping, q_nodes, h_nodes), AG, AH, C, alpha
    )
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


def _mapping_to_X(mapping: dict, q_nodes: list, h_nodes: list) -> np.ndarray:
    q_idx = {n: i for i, n in enumerate(q_nodes)}
    h_idx = {n: i for i, n in enumerate(h_nodes)}
    X = np.zeros((len(q_nodes), len(h_nodes)))
    for u, v in mapping.items():
        X[q_idx[u], h_idx[v]] = 1.0
    return X