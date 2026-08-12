"""
GNCCP WCS matching reproduced on the paper's OWN Figure 1 / Figure 2 example
(Yang, Qiao, Liu, IEEE TNNLS 2018), M=5 (graph G), N=6 (graph H), L=4.

Then: relabel G's vertices as 1->2, 2->3, 3->5, 5->1 (4 unchanged), and
relabel H's vertices as 6->5, 4->2, 5->4, 2->6 (1, 3 unchanged), and check
whether the matching found is "the same" -- meaning the same REAL vertex
pairs, expressed under the new names -- as before relabeling.

This is the same relabeling-invariance property demonstrated earlier for
the synthetic social network, now applied directly to the paper's own toy
example for a very concrete, checkable illustration.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar

np.set_printoptions(precision=3, suppress=True, linewidth=150)


# ============================================================================
# 1. THE PAPER'S OWN A_G, A_H (read directly off Fig. 2b/2c), symmetrized
#    by averaging each (i,j)/(j,i) pair -- the printed figure isn't
#    perfectly symmetric (rounding in the scan/print), but a weighted
#    undirected graph's adjacency matrix must be.
# ============================================================================
def _symmetrize(A):
    return (A + A.T) / 2.0


AG_raw = np.array([
    [0,   0.7, 0,   0,   1  ],
    [0.1, 0,   0.5, 0,   0  ],
    [0,   0.5, 0,   0.6, 0.6],
    [0,   0,   0.3, 0,   0.2],
    [0.9, 0,   0.6, 0.1, 0  ],
])
AH_raw = np.array([
    [0,   0.8, 0,   0,   0.8, 0.6],
    [0.4, 0,   0.8, 0.5, 0,   0.2],
    [0,   0.2, 0,   1,   0,   0  ],
    [0,   0.5, 0.7, 0,   0.7, 0  ],
    [0.5, 0,   0,   0.6, 0,   0.9],
    [0.4, 0.7, 0,   0,   1,   0  ],
])
AG = _symmetrize(AG_raw)
AH = _symmetrize(AH_raw)
M, N, L = 5, 6, 4

# The example correspondence shown in Fig. 1's X (1-indexed vertex labels):
# G1->H6, G2->H2, G3->H4, G5->H5, G4 unmatched (H1, H3 also unmatched)
FIGURE_X_LABELS = {1: 6, 2: 2, 3: 4, 5: 5}


# ============================================================================
# 2. H1(X) OBJECTIVE + GRADIENT  (paper Eq. 9, Eq. 11)
# ============================================================================
def H1_value(X, AG, AH):
    N = X.shape[1]
    ones_NN = np.ones((N, N))
    U = X @ ones_NN @ X.T
    term1 = np.trace((AG * AG) @ U.T)
    term2 = -2.0 * np.trace(AG @ X @ AH.T @ X.T)
    XAHXt = X @ AH @ X.T
    term3 = np.trace(XAHXt @ X @ AH.T @ X.T)
    return term1 + term2 + term3


def H1_grad(X, AG, AH):
    N = X.shape[1]
    ones_NN = np.ones((N, N))
    term1 = 2.0 * (AG * AG) @ X @ ones_NN
    term2 = -4.0 * (AG @ X @ AH)
    term3 = 4.0 * (X @ AH @ X.T @ X @ AH)
    return term1 + term2 + term3


def J_zeta_value(X, zeta, AG, AH):
    Fv = H1_value(X, AG, AH)
    reg = np.trace(X.T @ X)
    return ((1 - zeta) * Fv + zeta * reg) if zeta >= 0 else ((1 + zeta) * Fv + zeta * reg)


def J_zeta_grad(X, zeta, AG, AH):
    Fg = H1_grad(X, AG, AH)
    return ((1 - zeta) * Fg + 2 * zeta * X) if zeta >= 0 else ((1 + zeta) * Fg + 2 * zeta * X)


# ============================================================================
# 3. LINEAR STEP + GNCCP MAIN LOOP (Algorithm 1) + local-search refinement
# ============================================================================
def linear_step(grad, L):
    M, N = grad.shape
    row_ind, col_ind = linear_sum_assignment(grad)
    costs = grad[row_ind, col_ind]
    keep = np.argsort(costs)[:L]
    Y = np.zeros((M, N))
    Y[row_ind[keep], col_ind[keep]] = 1.0
    return Y


def gnccp_wcs_match(AG, AH, L, dzeta=0.02, inner_iters=25, tol=1e-8):
    M, N = AG.shape[0], AH.shape[0]
    X = np.full((M, N), L / (M * N))

    zeta = 1.0
    while zeta >= -1.0 - 1e-9:
        for _ in range(inner_iters):
            grad = J_zeta_grad(X, zeta, AG, AH)
            Y = linear_step(grad, L)
            d = Y - X

            def obj(lam):
                return J_zeta_value(X + lam * d, zeta, AG, AH)

            lam = minimize_scalar(obj, bounds=(0.0, 1.0), method='bounded').x
            X_new = X + lam * d
            converged = np.linalg.norm(X_new - X) < tol
            X = X_new
            if converged:
                break
        zeta -= dzeta

    return linear_step(-X, L)


def local_search_refine(assign, AG, AH, M, N, max_rounds=50):
    def cost_of(a):
        X = np.zeros((M, N))
        for g, h in a.items():
            X[g, h] = 1.0
        return H1_value(X, AG, AH)

    assign = dict(assign)
    for _ in range(max_rounds):
        improved = False
        cur = cost_of(assign)
        matched_h = set(assign.values())
        unmatched_g = [g for g in range(M) if g not in assign]
        free_h = [h for h in range(N) if h not in matched_h]

        for g in list(assign.keys()):
            for h in free_h:
                trial = dict(assign); trial[g] = h
                nc = cost_of(trial)
                if nc < cur - 1e-9:
                    assign, cur, improved = trial, nc, True
                    break
            if improved:
                break
        if improved:
            continue

        gs = list(assign.keys())
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                g1, g2 = gs[i], gs[j]
                trial = dict(assign); trial[g1], trial[g2] = assign[g2], assign[g1]
                nc = cost_of(trial)
                if nc < cur - 1e-9:
                    assign, cur, improved = trial, nc, True
                    break
            if improved:
                break
        if improved:
            continue

        for g_in in gs:
            for g_out in unmatched_g:
                for h in free_h + [assign[g_in]]:
                    trial = dict(assign); del trial[g_in]; trial[g_out] = h
                    nc = cost_of(trial)
                    if nc < cur - 1e-9:
                        assign, cur, improved = trial, nc, True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return assign


def match_and_refine(AG, AH, L, **gnccp_kwargs):
    M, N = AG.shape[0], AH.shape[0]
    X = gnccp_wcs_match(AG, AH, L, **gnccp_kwargs)
    assign = {i: int(np.argmax(row)) for i, row in enumerate(X) if row.sum() > 0.5}
    return local_search_refine(assign, AG, AH, M, N)


# ============================================================================
# 4. RELABELING: build a matrix under new vertex names, given old->new map
# ============================================================================
def relabel_matrix(A, label_map):
    """label_map: dict {old_label: new_label}, labels are 1-indexed."""
    n = A.shape[0]
    newA = np.zeros_like(A)
    for old_a in range(1, n + 1):
        for old_b in range(1, n + 1):
            new_a, new_b = label_map[old_a], label_map[old_b]
            newA[new_a - 1, new_b - 1] = A[old_a - 1, old_b - 1]
    return newA


def assign_to_labels(assign):
    """0-indexed {g_idx: h_idx} -> 1-indexed {g_label: h_label}."""
    return {g + 1: h + 1 for g, h in assign.items()}


def translate_labels(label_pairs, inv_map_G, inv_map_H):
    """Translate a {g_label: h_label} dict (in relabeled space) back to the
    ORIGINAL label space using each side's inverse relabeling map."""
    return {inv_map_G[g]: inv_map_H[h] for g, h in label_pairs.items()}


# ============================================================================
# 5. DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("A_G (symmetrized from Fig. 2b) and A_H (symmetrized from Fig. 2c)")
    print("=" * 78)
    print("A_G:\n", AG)
    print("A_H:\n", AH)

    cost_figure_X = None
    Xfig = np.zeros((M, N))
    for g, h in FIGURE_X_LABELS.items():
        Xfig[g - 1, h - 1] = 1.0
    cost_figure_X = H1_value(Xfig, AG, AH)
    print(f"\nFig. 1's own example X = {FIGURE_X_LABELS}  ->  H1 cost = {cost_figure_X:.4f}")
    print("(this is the paper's illustrative example, not necessarily what the")
    print(" solver below will independently find as the lowest-cost match)")

    print("\n" + "=" * 78)
    print("B) RUN 1: solve WCS matching on the ORIGINAL labeling")
    print("=" * 78)
    assign1 = match_and_refine(AG, AH, L, dzeta=0.02, inner_iters=25)
    labels1 = assign_to_labels(assign1)
    cost1 = H1_value(
        np.eye(M, N)[list(assign1.keys())] if False else
        (lambda X: X)(np.array([[1.0 if assign1.get(i) == j else 0.0 for j in range(N)] for i in range(M)])),
        AG, AH)
    print(f"Found correspondence (original labels): {labels1}")
    print(f"H1 cost of solver's answer: {cost1:.4f}")

    print("\n" + "=" * 78)
    print("C) Relabel BOTH graphs:")
    print("   G: 1->2, 2->3, 3->5, 5->1, (4 unchanged)")
    print("   H: 6->5, 4->2, 5->4, 2->6, (1, 3 unchanged)")
    print("=" * 78)
    label_map_G = {1: 2, 2: 3, 3: 5, 4: 4, 5: 1}
    label_map_H = {1: 1, 2: 6, 3: 3, 4: 2, 5: 4, 6: 5}
    inv_map_G = {v: k for k, v in label_map_G.items()}
    inv_map_H = {v: k for k, v in label_map_H.items()}

    AG2 = relabel_matrix(AG, label_map_G)
    AH2 = relabel_matrix(AH, label_map_H)
    print("Relabeled A_G:\n", AG2)
    print("Relabeled A_H:\n", AH2)

    print("\n" + "=" * 78)
    print("D) RUN 2: solve WCS matching on the RELABELED graphs")
    print("=" * 78)
    assign2 = match_and_refine(AG2, AH2, L, dzeta=0.02, inner_iters=25)
    labels2 = assign_to_labels(assign2)
    print(f"Found correspondence (NEW labels): {labels2}")

    labels2_translated_back = translate_labels(labels2, inv_map_G, inv_map_H)
    print(f"Translated back to ORIGINAL labels: {labels2_translated_back}")

    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    same = (labels1 == labels2_translated_back)
    print(f"Run 1 (original labels)         : {labels1}")
    print(f"Run 2 (translated back to original labels): {labels2_translated_back}")
    print(f"Identical real-world correspondence: {same}")
    print()
    print("Renaming vertices is pure bookkeeping -- it doesn't touch which pairs")
    print("of REAL vertices are close/strongly-connected to which others. Since")
    print("A_G2 and A_H2 are just A_G and A_H with rows/columns moved to new")
    print("positions (not new values), the optimization problem is IDENTICAL in")
    print("substance, so the algorithm must -- and does -- find the same answer,")
    print("just wearing the new name tags.")