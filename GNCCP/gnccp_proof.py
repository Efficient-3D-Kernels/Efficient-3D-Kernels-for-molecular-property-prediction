"""
Weighted Common Subgraph (WCS) matching via GNCCP
==================================================
Faithful implementation of:

  Xu Yang, Hong Qiao, Zhi-Yong Liu,
  "An Algorithm for Finding the Most Similar Given Sized Subgraphs
   in Two Weighted Graphs", IEEE TNNLS, 2018.

Implements exactly:
  - Objective F(X) = alpha * H1(X) + (1-alpha) * tr(C^T X)      (paper eq. 2, 9)
  - H1 relaxation of H0                                          (eq. 9)
  - Gradient of H1                                                (eq. 11)
  - GNCCP homotopy  J_zeta(X)                                     (eq. 4, 6)
  - Frank-Wolfe inner loop with the *fast* LP step from Sec. II-D
    (rectangular Hungarian assignment + drop the M-L worst matches)
  - Optimal step size by 1-D line search (backtracking style)     (eq. 8)

Then demonstrates a PROPOSITION (proved below) that WCS matching is
invariant to a relabeling of the graph vertices: if you permute the
vertex names of G by P and of H by Q, the optimal partial-permutation
matrix simply gets permuted the same way, X' = P X Q^T, and the
objective value H0(X) is *exactly* unchanged.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------
# 1. Core WCS / GNCCP algorithm  (Algorithm 1 in the paper)
# ----------------------------------------------------------------------

def H1(X, AG, AH):
    """Eq. (9): H1(X) = || U o AG - X AH X^T ||_F^2 , U = X 1_{NxN} X^T."""
    N = AH.shape[0]
    J = np.ones((N, N))
    U = X @ J @ X.T
    diff = (U * AG) - (X @ AH @ X.T)
    return np.sum(diff * diff)


def grad_H1(X, AG, AH):
    """Eq. (11): gradient of H1 w.r.t. X."""
    M, N = X.shape
    J = np.ones((N, N))
    term1 = (AG.T * AG.T + AG * AG) @ X @ J
    term2 = -2.0 * (AG.T @ X @ AH + AG @ X @ AH.T)
    term3 = 2.0 * (X @ AH @ X.T @ X @ AH.T + X @ AH.T @ X.T @ X @ AH)
    return term1 + term2 + term3


def F(X, AG, AH, C, alpha):
    """Eq. (2): F(X) = alpha*H1(X) + (1-alpha)*tr(C^T X)."""
    return alpha * H1(X, AG, AH) + (1 - alpha) * np.sum(C * X)


def grad_F(X, AG, AH, C, alpha):
    """Eq. (7) (with H1 in place of H0)."""
    return alpha * grad_H1(X, AG, AH) + (1 - alpha) * C


def J_zeta(X, zeta, AG, AH, C, alpha):
    """Eq. (4): GNCCP homotopy objective."""
    reg = np.sum(X * X)  # tr(X^T X)
    Fx = F(X, AG, AH, C, alpha)
    if zeta >= 0:
        return (1 - zeta) * Fx + zeta * reg
    else:
        return (1 + zeta) * Fx + zeta * reg


def grad_J_zeta(X, zeta, AG, AH, C, alpha):
    """Eq. (6): gradient of the GNCCP homotopy objective."""
    gF = grad_F(X, AG, AH, C, alpha)
    if zeta >= 0:
        return (1 - zeta) * gF + 2 * zeta * X
    else:
        return (1 + zeta) * gF + 2 * zeta * X


def lp_step_fast(grad, L):
    """
    Fast approximate solution of eq. (5):
        Y = argmax tr(-grad^T Y)  s.t. Y in D  (M ones-per-row/col <=1, sum=L)

    Section II-D 'fast method': solve the full MxN rectangular assignment
    (Hungarian) minimizing sum(grad[i,j]) over a full matching of the M rows,
    then keep only the L cheapest (best) pairs and drop the (M-L) worst ones.
    Returns a genuine 0/1 partial-permutation matrix Y in P (subset of D).
    """
    M, N = grad.shape
    row_ind, col_ind = linear_sum_assignment(grad)     # minimizes sum(grad[row,col])
    costs = grad[row_ind, col_ind]
    keep = np.argsort(costs)[:L]                       # L smallest (best) costs
    Y = np.zeros((M, N))
    Y[row_ind[keep], col_ind[keep]] = 1.0
    return Y


def line_search(X, Y, zeta, AG, AH, C, alpha, n_grid=25):
    """Eq. (8): 1-D minimization of J_zeta(X + lambda*(Y-X)) over lambda in [0,1]."""
    D = Y - X
    best_lam, best_val = 0.0, J_zeta(X, zeta, AG, AH, C, alpha)
    for lam in np.linspace(0, 1, n_grid):
        val = J_zeta(X + lam * D, zeta, AG, AH, C, alpha)
        if val < best_val:
            best_val, best_lam = val, lam
    # local refinement (golden-section-ish polish) around best_lam
    lo, hi = max(0.0, best_lam - 1.0 / n_grid), min(1.0, best_lam + 1.0 / n_grid)
    for lam in np.linspace(lo, hi, n_grid):
        val = J_zeta(X + lam * D, zeta, AG, AH, C, alpha)
        if val < best_val:
            best_val, best_lam = val, lam
    return best_lam


def wcs_gnccp(AG, AH, L, alpha=1.0, C=None, d_zeta=0.05,
              fw_iters=8, verbose=False):
    """
    Algorithm 1 of the paper: GNCCP-based WCS matching.

    AG : (M,M) weighted adjacency matrix of graph G
    AH : (N,N) weighted adjacency matrix of graph H,  M <= N
    L  : number of vertices to match, L <= M <= N
    alpha : balance between structure term H1 and unary cost term C
    C  : (M,N) unary dissimilarity cost matrix (default: zeros)
    """
    M, N = AG.shape[0], AH.shape[0]
    assert L <= M <= N
    if C is None:
        C = np.zeros((M, N))

    # Initialization: X <- 1_{MxN} * L/(M*N),  zeta <- 1
    X = np.full((M, N), L / (M * N))
    zeta = 1.0

    while zeta >= -1.0:
        for _ in range(fw_iters):
            g = grad_J_zeta(X, zeta, AG, AH, C, alpha)
            Y = lp_step_fast(g, L)
            lam = line_search(X, Y, zeta, AG, AH, C, alpha)
            X = X + lam * (Y - X)
        if verbose:
            print(f"zeta={zeta:+.2f}  F(X)={F(X, AG, AH, C, alpha):.5f}")
        zeta -= d_zeta

    # Snap the final (already near-binary) X onto P exactly, for a clean output
    g_final = X  # use -X as "reward": prefer entries closest to 1
    row_ind, col_ind = linear_sum_assignment(-X)
    scores = X[row_ind, col_ind]
    keep = np.argsort(-scores)[:L]
    X_bin = np.zeros((M, N))
    X_bin[row_ind[keep], col_ind[keep]] = 1.0
    return X_bin


# ----------------------------------------------------------------------
# 2. Example graphs from Fig. 1 / Fig. 2 of the paper: M=5, N=6, L=4
#    (The paper's figure is a schematic drawing without printed numeric
#    weights, so a representative symmetric weighted adjacency matrix of
#    the same size is used here -- swap in your real AG/AH if you have them.)
# ----------------------------------------------------------------------

M, N, L = 5, 6, 4

AG = np.array([
    [0.0, 0.4, 0.0, 0.7, 0.2],
    [0.4, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.3, 0.6],
    [0.7, 0.0, 0.3, 0.0, 0.0],
    [0.2, 0.0, 0.6, 0.0, 0.0],
])

AH = np.array([
    [0.0, 0.3, 0.8, 0.0, 0.0, 0.6],
    [0.3, 0.0, 0.0, 0.2, 0.0, 0.0],
    [0.8, 0.0, 0.0, 0.7, 0.1, 0.0],
    [0.0, 0.2, 0.7, 0.0, 0.4, 0.0],
    [0.0, 0.0, 0.1, 0.4, 0.0, 0.9],
    [0.6, 0.0, 0.0, 0.0, 0.9, 0.0],
])

alpha = 1.0                 # pure structural (no vertex-label) term, as in
C = np.zeros((M, N))        # the paper's synthetic-graph experiments


# ----------------------------------------------------------------------
# 3. Run the algorithm on the ORIGINAL graphs
# ----------------------------------------------------------------------

print("=" * 70)
print("STEP 1: run GNCCP-based WCS matching on the original G, H")
print("=" * 70)
X_orig = wcs_gnccp(AG, AH, L, alpha=alpha, C=C)
print("\nX (original labeling):\n", X_orig)
h0_orig = H1(X_orig, AG, AH)
print(f"\nH0(X) on original graphs = {h0_orig:.6f}")


# ----------------------------------------------------------------------
# 4. PROPOSITION: relabeling vertices permutes X but does not change H0
# ----------------------------------------------------------------------
#
#  Let P (MxM) and Q (NxN) be permutation matrices encoding a relabeling
#  of G's and H's vertices respectively:  AG' = P AG P^T,  AH' = Q AH Q^T.
#  Claim: for any X,
#       H0( P X Q^T ; AG', AH' ) = H0( X ; AG, AH ).
#  Proof sketch (uses only that P, Q are permutation matrices, so
#  P P^T = I, Q Q^T = I, and that conjugating by a permutation matrix
#  commutes with the Hadamard product / preserves the Frobenius norm):
#
#     U' = (PXQ^T) 1_{NxN} (PXQ^T)^T
#        = P X (Q^T 1_{NxN} Q) X^T P^T = P X 1_{NxN} X^T P^T = P U P^T
#        [since Q^T 1_{NxN} Q = 1_{NxN} for any permutation Q]
#
#     U' o AG' = (P U P^T) o (P AG P^T) = P (U o AG) P^T
#        [Hadamard product commutes with simultaneous row/col permutation]
#
#     (PXQ^T) AH' (PXQ^T)^T = P X (Q^T Q) AH (Q^T Q) X^T P^T = P (X AH X^T) P^T
#
#     => U' o AG' - (PXQ^T)AH'(PXQ^T)^T = P [ (U o AG) - X AH X^T ] P^T
#     => || ... ||_F^2 is unchanged, since Frobenius norm is invariant
#        under conjugation by an orthogonal (here: permutation) matrix.
#     Hence H0(PXQ^T) = H0(X).  QED.
#
#  Consequence: if X* is optimal for (AG,AH), then P X* Q^T is optimal
#  for (AG',AH') -- the algorithm should recover exactly this permuted
#  matching (up to the usual local-optimum / tie-breaking caveats of any
#  non-convex solver).
# ----------------------------------------------------------------------

# --- build the permutation matrices from your relabeling ---
# G (5 vertices, 1-indexed labels 1..5): 1->2, 2->3, 3->5, 5->1, (4->4 unchanged)
# H (6 vertices, 1-indexed labels 1..6): 6->5, 4->2, 5->4, 2->6, (1->1, 3->3 unchanged)

# perm_G[old_index] = new_index   (0-indexed)
perm_G = np.array([1, 2, 4, 3, 0])   # label1->2, label2->3, label3->5, label4->4, label5->1
# label1->1, label2->6, label3->3, label4->2, label5->4, label6->5
perm_H = np.array([0, 5, 2, 1, 3, 4])

def perm_matrix(perm):
    """Build permutation matrix P with P[perm[i], i] = 1."""
    n = len(perm)
    P = np.zeros((n, n))
    for old, new in enumerate(perm):
        P[new, old] = 1.0
    return P

P_G = perm_matrix(perm_G)
P_H = perm_matrix(perm_H)

AG_relabel = P_G @ AG @ P_G.T
AH_relabel = P_H @ AH @ P_H.T

print("\n" + "=" * 70)
print("STEP 2: relabel vertices of G and H, run GNCCP again from scratch")
print("=" * 70)
X_relabel = wcs_gnccp(AG_relabel, AH_relabel, L, alpha=alpha, C=np.zeros((M, N)))
print("\nX (after relabeling, solved independently):\n", X_relabel)
h0_relabel = H1(X_relabel, AG_relabel, AH_relabel)
print(f"\nH0(X) on relabeled graphs = {h0_relabel:.6f}")

# --- prediction from the proposition ---
X_predicted = P_G @ X_orig @ P_H.T

print("\n" + "=" * 70)
print("STEP 3: compare -- does relabeling-then-solving equal")
print("        solving-then-relabeling, and is H0 unchanged?")
print("=" * 70)
print("\nP_G X_orig P_H^T  (prediction from the proposition):\n", X_predicted)
print("\nX obtained by re-running GNCCP on the relabeled graphs:\n", X_relabel)

same_matching = np.array_equal(X_predicted, X_relabel)
print(f"\nMatching matrices identical (X_relabel == P_G X_orig P_H^T)? {same_matching}")
print(f"H0 original   = {h0_orig:.10f}")
print(f"H0 relabeled  = {h0_relabel:.10f}")
print(f"H0(P X_orig P_H^T) evaluated on relabeled graphs = "
      f"{H1(X_predicted, AG_relabel, AH_relabel):.10f}")
print(f"|H0_orig - H0_relabel| = {abs(h0_orig - h0_relabel):.2e}  (must be ~0)")

assert abs(h0_orig - h0_relabel) < 1e-8, "Objective value must be invariant!"
print("\n>>> CONFIRMED: relabeling vertices only permutes the matching matrix;")
print(">>> the graph structure and the optimal objective value H0(X) are unchanged.")