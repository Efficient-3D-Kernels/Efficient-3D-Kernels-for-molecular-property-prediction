"""
Weighted Common Subgraph (WCS) Matching via GNCCP
==================================================
Implements: Yang, Qiao, Liu, "An Algorithm for Finding the Most Similar
Given Sized Subgraphs in Two Weighted Graphs", IEEE TNNLS 2018.

Faithful to the paper:
  - Objective            F(X) = alpha*H1(X) + (1-alpha)*tr(C^T X)      [Eq. 2, relaxed via Eq. 9]
  - Continuation          J_zeta(X), zeta: 1 -> -1                     [Eq. 4]
  - Inner solver           Frank-Wolfe (linear step + exact line search)[Eq. 5-8]
  - Partial matching       L < M <= N via rectangular-Hungarian +
                            "drop worst (M-L)" fast method              [Sec. II-D]

Works for 2D or 3D point-cloud graphs (dim=2 or dim=3). Distances (not raw
coordinates) are used as edge weights, so the whole pipeline is provably
invariant to rigid rotation/translation of a graph -- verified directly below.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=3, suppress=True, linewidth=150)


# ============================================================================
# 1. SYNTHETIC DATA (2D or 3D point-cloud graphs, with outliers)
# ============================================================================
def generate_synthetic_pair(M=12, N=16, L=9, dim=2, noise_std=0.005,
                             rotate_deg=0.0, translate=None, seed=0):
    """
    Graph G: M random points in `dim` dimensions -> pairwise-distance matrix A_G.
    Graph H: L of G's points (correspondence) + (N-L) random outliers,
             then the WHOLE point cloud (inliers+outliers together) is
             rigidly rotated/translated -> pairwise-distance matrix A_H.

    Because a rigid transform preserves every pairwise distance, A_H is
    numerically IDENTICAL regardless of rotate_deg/translate (verified in
    verify_rigid_invariance() below) -- that is what "rotation invariant"
    means for a distance-based graph representation.

    Returns: AG (MxM), AH (NxN), ground_truth {g_idx: h_idx}, coords for plotting/debug.
    """
    rng = np.random.RandomState(seed)

    G_points = rng.uniform(0, 1, size=(M, dim))

    inlier_G = np.sort(rng.choice(M, size=L, replace=False))
    order = rng.permutation(L)

    H_points = np.zeros((N, dim))
    ground_truth = {}
    for k in range(L):
        H_points[order[k]] = G_points[inlier_G[k]]
        ground_truth[int(inlier_G[k])] = int(order[k])

    if N > L:
        H_points[L:] = rng.uniform(0, 1, size=(N - L, dim))

    # noise is part of the physical point cloud -> gets carried through the rotation
    H_points_pre_rigid = H_points + rng.normal(0, noise_std, size=H_points.shape)

    H_points_rigid = _apply_rigid_transform(H_points_pre_rigid, dim, rotate_deg, translate)

    AG = _dist_matrix(G_points)
    AH = _dist_matrix(H_points_rigid)
    return AG, AH, ground_truth, G_points, H_points_pre_rigid, H_points_rigid


def _apply_rigid_transform(pts, dim, rotate_deg, translate):
    center = pts.mean(axis=0)
    if dim == 2:
        theta = np.deg2rad(rotate_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
    elif dim == 3:
        # rotate_deg: either a scalar (applied about a fixed default axis mix,
        # for parity with simple "tilt" demos) or a 3-vector of Euler angles.
        if np.isscalar(rotate_deg):
            euler = [rotate_deg, rotate_deg * 0.5, rotate_deg * 0.25]
        else:
            euler = rotate_deg
        R = Rotation.from_euler('xyz', euler, degrees=True).as_matrix()
    else:
        raise ValueError("dim must be 2 or 3")

    t = np.zeros(dim) if translate is None else np.array(translate)
    return (pts - center) @ R.T + center + t


def _dist_matrix(pts):
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def verify_rigid_invariance(M=12, N=16, L=9, dim=3, seed=0):
    """Directly proves AH is unchanged by rotation, independent of the solver."""
    _, AH0, _, _, _, _ = generate_synthetic_pair(M, N, L, dim=dim, noise_std=0.0,
                                                   rotate_deg=0.0, seed=seed)
    _, AH1, _, _, _, _ = generate_synthetic_pair(M, N, L, dim=dim, noise_std=0.0,
                                                   rotate_deg=91.0 if dim == 3 else 91.0,
                                                   translate=[5, -3, 2][:dim], seed=seed)
    max_diff = np.max(np.abs(AH0 - AH1))
    print(f"[invariance check] max |AH(unrotated) - AH(rotated)| = {max_diff:.2e}  "
          f"-> {'PASS' if max_diff < 1e-9 else 'FAIL'}")
    return max_diff < 1e-9


# ============================================================================
# 2. OBJECTIVE H1(X) AND ITS GRADIENT  (Eq. 9, Eq. 11 -- simplified for
#    symmetric/undirected A_G, A_H, which distance matrices always are)
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


def F_value(X, AG, AH, alpha=1.0, C=None):
    val = alpha * H1_value(X, AG, AH)
    if C is not None:
        val += (1 - alpha) * np.trace(C.T @ X)
    return val


def F_grad(X, AG, AH, alpha=1.0, C=None):
    g = alpha * H1_grad(X, AG, AH)
    if C is not None:
        g = g + (1 - alpha) * C
    return g


def J_zeta_value(X, zeta, AG, AH, alpha=1.0, C=None):
    Fv = F_value(X, AG, AH, alpha, C)
    reg = np.trace(X.T @ X)
    return ((1 - zeta) * Fv + zeta * reg) if zeta >= 0 else ((1 + zeta) * Fv + zeta * reg)


def J_zeta_grad(X, zeta, AG, AH, alpha=1.0, C=None):
    Fg = F_grad(X, AG, AH, alpha, C)
    return ((1 - zeta) * Fg + 2 * zeta * X) if zeta >= 0 else ((1 + zeta) * Fg + 2 * zeta * X)


# ============================================================================
# 3. LINEAR STEP:  Y = argmin tr(grad^T Y),  Y in D (L ones, row/col <= 1)
#    "fast" method (Sec. II-D): full rectangular Hungarian (M matches),
#    then keep only the L cheapest of those M matches.
# ============================================================================
def linear_step(grad, L):
    M, N = grad.shape
    row_ind, col_ind = linear_sum_assignment(grad)
    costs = grad[row_ind, col_ind]
    keep = np.argsort(costs)[:L]
    Y = np.zeros((M, N))
    Y[row_ind[keep], col_ind[keep]] = 1.0
    return Y


# ============================================================================
# 4. GNCCP MAIN LOOP (Algorithm 1) with Frank-Wolfe inner solver
# ============================================================================
def gnccp_wcs_match(AG, AH, L, alpha=1.0, C=None,
                     dzeta=0.05, inner_iters=15, tol=1e-6, verbose=False):
    M, N = AG.shape[0], AH.shape[0]
    X = np.full((M, N), L / (M * N))   # Algorithm 1 initialization

    zeta = 1.0
    log = []
    while zeta >= -1.0 - 1e-9:
        for _ in range(inner_iters):
            grad = J_zeta_grad(X, zeta, AG, AH, alpha, C)
            Y = linear_step(grad, L)
            d = Y - X

            def obj(lam):
                return J_zeta_value(X + lam * d, zeta, AG, AH, alpha, C)

            lam = minimize_scalar(obj, bounds=(0.0, 1.0), method='bounded').x
            X_new = X + lam * d
            converged = np.linalg.norm(X_new - X) < tol
            X = X_new
            if converged:
                break

        if verbose:
            log.append((zeta, J_zeta_value(X, zeta, AG, AH, alpha, C)))
        zeta -= dzeta

    X_final = linear_step(-X, L)   # final binarization: keep the L largest entries
    return X_final, log


# ============================================================================
# 5. EVALUATION / REPORTING
# ============================================================================
def matching_accuracy(X_final, ground_truth):
    pred = {i: int(np.argmax(row)) for i, row in enumerate(X_final) if row.sum() > 0.5}
    correct = sum(1 for g, h in ground_truth.items() if pred.get(g) == h)
    total = len(ground_truth)
    return correct, total, pred


def print_breakdown(pred, ground_truth):
    print(f"{'G node':<8} | {'True H':<8} | {'Pred H':<8} | Status")
    print("-" * 40)
    for g, true_h in sorted(ground_truth.items()):
        pred_h = pred.get(g, None)
        status = "OK" if pred_h == true_h else "MISS"
        print(f"{g:<8} | {true_h:<8} | {str(pred_h):<8} | {status}")


# ============================================================================
# 6. DEMO: 3D tilted WCS matching + rigorous rotation-invariance test
# ============================================================================
if __name__ == "__main__":
    M, N, L, DIM, SEED = 10, 11, 9, 3, 3

    print("=" * 70)
    print("A) Direct proof that rotation/translation cannot change AH")
    print("=" * 70)
    verify_rigid_invariance(M, N, L, dim=DIM, seed=SEED)

    print("\n" + "=" * 70)
    print("B) RUN 1: no rotation")
    print("=" * 70)
    AG, AH, gt, Gp, Hp_pre, Hp = generate_synthetic_pair(
        M, N, L, dim=DIM, noise_std=0.002, rotate_deg=0.0, seed=SEED)
    X1, _ = gnccp_wcs_match(AG, AH, L, alpha=1.0, dzeta=0.05, inner_iters=20)
    c1, t1, pred1 = matching_accuracy(X1, gt)
    print_breakdown(pred1, gt)
    print(f"Accuracy: {c1}/{t1} = {100*c1/t1:.1f}%")

    print("\n" + "=" * 70)
    print("C) RUN 2: whole H graph rigidly tilted (3D rotation) + translated")
    print("=" * 70)
    AG2, AH2, gt2, Gp2, Hp2_pre, Hp2 = generate_synthetic_pair(
        M, N, L, dim=DIM, noise_std=0.002, rotate_deg=65.0,
        translate=[4.0, -2.0, 1.5], seed=SEED)
    print("AG identical across runs:", np.allclose(AG, AG2))
    print("AH identical across runs:", np.allclose(AH, AH2, atol=1e-9))
    X2, _ = gnccp_wcs_match(AG2, AH2, L, alpha=1.0, dzeta=0.05, inner_iters=20)
    c2, t2, pred2 = matching_accuracy(X2, gt2)
    print_breakdown(pred2, gt2)
    print(f"Accuracy: {c2}/{t2} = {100*c2/t2:.1f}%")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    same_matrix = np.allclose(AH, AH2, atol=1e-9)
    same_result = (pred1 == pred2)
    print(f"AH bit-identical before/after rotation: {same_matrix}")
    print(f"Predicted matching identical before/after rotation: {same_result}")
    print("The solver never sees coordinates -- only pairwise distances -- so a")
    print("tilted graph and an untilted graph are literally the same input to it.")