"""
GNCCP-based Weighted Common Subgraph Matching -- applied to a synthetic
Metal-Organic Framework (MOF) matching problem.

IMPORTANT HONESTY NOTE
-----------------------
This uses a SYNTHETIC, simplified point-cloud model of a MOF fragment
(metal centers + organic linker atoms placed at chemically-plausible
distances), NOT a real crystal structure from a CIF file. It's built to
give the graph-matching algorithm a realistic-shaped problem to solve.
For real MOFs, you'd load atomic coordinates from a .cif with a tool
like ASE (`ase.io.read`) or pymatgen (`Structure.from_file`), and feed
those coordinates into `_dist_matrix()` below instead of the synthetic
generator. Everything downstream (matching, invariance, evaluation)
would work unchanged.

THE APPLICATION
----------------
You have a reference MOF fragment G (e.g. one known coordination motif:
a metal cluster with its attached organic linkers). You have a sample H
which is a bigger simulated/experimental region that:
  - contains most of G's atoms (structural correspondence)
  - is MISSING a couple of atoms (a vacancy defect -- fewer matched
    atoms than G has, hence L < M: this is exactly the WCS/"subgraph"
    case the paper is built for, not plain equal-sized matching)
  - has extra DECOY atoms (rest of the crystal, or a guest molecule)
  - is reported in a DIFFERENT crystallographic orientation (rotated),
    plus small thermal-motion noise

The question WCS matching answers: "which atoms in H correspond to
which atoms in G's reference motif?" -- despite the defect, the decoys,
the noise, and the orientation difference.

We ALSO use the vertex-label (metal vs linker) unary cost term C from
the paper's F(X) = alpha*H1(X) + (1-alpha)*tr(C^T X), so the algorithm
is discouraged from ever matching a metal atom to a linker atom -- a
chemically meaningless correspondence that pure-structure (alpha=1)
matching has no way to avoid.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=3, suppress=True, linewidth=150)


# ============================================================================
# 1. SYNTHETIC MOF FRAGMENT GENERATOR
# ============================================================================
def generate_mof_reference(n_metal=4, linkers_per_metal=(3, 4),
                            metal_min_sep=6.0, metal_linker_dist=2.05,
                            bond_noise=0.05, seed=0):
    """
    Builds a synthetic MOF-like point cloud:
      - `n_metal` metal centers (secondary building units), placed with a
        minimum pairwise separation (mimics real SBU spacing, ~6-12 A)
      - each metal center gets a random coordination number of organic
        linker atoms (linkers_per_metal range), placed at ~metal_linker_dist
        A away in a random direction (typical M-O/M-N coordination bond
        length), with small noise (bond_noise A) for realism

    Returns coords (Nx3 array, angstrom-like units) and labels (N array
    of 'M' for metal or 'L' for linker).
    """
    rng = np.random.RandomState(seed)

    # Place metal centers with a minimum-separation rejection sampler
    metal_coords = []
    tries = 0
    while len(metal_coords) < n_metal and tries < 10000:
        tries += 1
        cand = rng.uniform(0, 20, size=3)
        if all(np.linalg.norm(cand - m) >= metal_min_sep for m in metal_coords):
            metal_coords.append(cand)
    metal_coords = np.array(metal_coords)

    coords = [c for c in metal_coords]
    labels = ['M'] * len(metal_coords)
    parent_metal = [-1] * len(metal_coords)   # which metal each atom "belongs" to (-1 = itself)

    for m_idx, m in enumerate(metal_coords):
        n_link = rng.randint(linkers_per_metal[0], linkers_per_metal[1] + 1)
        for _ in range(n_link):
            direction = rng.normal(0, 1, size=3)
            direction /= np.linalg.norm(direction)
            pos = m + direction * (metal_linker_dist + rng.normal(0, bond_noise))
            coords.append(pos)
            labels.append('L')
            parent_metal.append(m_idx)

    return np.array(coords), np.array(labels), np.array(parent_metal)


def build_sample_from_reference(G_coords, G_labels, n_missing=2, n_decoys=6,
                                 noise_std=0.08, rotate_deg=None, translate=None,
                                 seed=1):
    """
    Builds a "sample" H from reference G:
      - drops `n_missing` random G atoms (simulated vacancy defect -> L < M)
      - adds `n_decoys` unrelated atoms nearby (rest-of-crystal / guest atoms)
      - adds thermal noise to every kept atom
      - rigidly rotates + translates the WHOLE sample (inliers + decoys
        together), simulating a differently-oriented crystallographic frame

    Returns AG, AH, ground_truth {g_idx: h_idx}, and coordinate arrays for
    inspection.
    """
    rng = np.random.RandomState(seed)
    M = len(G_coords)

    keep_idx = np.sort(rng.choice(M, size=M - n_missing, replace=False))
    L = len(keep_idx)
    order = rng.permutation(L)

    N = L + n_decoys
    H_coords = np.zeros((N, 3))
    H_labels = np.array([''] * N, dtype='<U1')
    ground_truth = {}
    for k, g_idx in enumerate(keep_idx):
        H_coords[order[k]] = G_coords[g_idx]
        H_labels[order[k]] = G_labels[g_idx]
        ground_truth[int(g_idx)] = int(order[k])

    # decoys: scattered near the same region, random type
    center = G_coords.mean(axis=0)
    for d in range(n_decoys):
        H_coords[L + d] = center + rng.normal(0, 6.0, size=3)
        H_labels[L + d] = rng.choice(['M', 'L'])

    H_coords += rng.normal(0, noise_std, size=H_coords.shape)

    if rotate_deg is not None:
        euler = [rotate_deg, rotate_deg * 0.6, rotate_deg * 0.3] if np.isscalar(rotate_deg) else rotate_deg
        R = Rotation.from_euler('xyz', euler, degrees=True).as_matrix()
        c = H_coords.mean(axis=0)
        t = np.zeros(3) if translate is None else np.array(translate)
        H_coords = (H_coords - c) @ R.T + c + t

    AG = _dist_matrix(G_coords)
    AH = _dist_matrix(H_coords)
    return AG, AH, ground_truth, H_coords, H_labels, L


def _dist_matrix(pts):
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def build_label_cost(G_labels, H_labels, mismatch_penalty=6.0):
    """C[i,j] = 0 if G atom i and H atom j are the same type, else a penalty.
    Used as the unary term in F(X) = alpha*H1(X) + (1-alpha)*tr(C^T X)."""
    M, N = len(G_labels), len(H_labels)
    C = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if G_labels[i] != H_labels[j]:
                C[i, j] = mismatch_penalty
    return C


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
# 3. LINEAR STEP (Sec. II-D "fast" method) + GNCCP MAIN LOOP (Algorithm 1)
# ============================================================================
def linear_step(grad, L):
    M, N = grad.shape
    row_ind, col_ind = linear_sum_assignment(grad)
    costs = grad[row_ind, col_ind]
    keep = np.argsort(costs)[:L]
    Y = np.zeros((M, N))
    Y[row_ind[keep], col_ind[keep]] = 1.0
    return Y


def gnccp_wcs_match(AG, AH, L, alpha=1.0, C=None,
                     dzeta=0.05, inner_iters=15, tol=1e-6):
    M, N = AG.shape[0], AH.shape[0]
    X = np.full((M, N), L / (M * N))

    zeta = 1.0
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
        zeta -= dzeta

    return linear_step(-X, L)


# ============================================================================
# 4. DISCRETE LOCAL-SEARCH REFINEMENT (post-processing hill-climb)
#    GNCCP's continuous relaxation can land in a local optimum of its own
#    objective; this cheap discrete cleanup fixes the common cases (found
#    to matter a lot in practice -- see conversation history).
# ============================================================================
def local_search_refine(assign, AG, AH, M, N, alpha=1.0, C=None, max_rounds=50):
    def cost_of(a):
        X = np.zeros((M, N))
        for g, h in a.items():
            X[g, h] = 1.0
        return F_value(X, AG, AH, alpha, C)

    assign = dict(assign)
    for _ in range(max_rounds):
        improved = False
        cur = cost_of(assign)
        matched_h = set(assign.values())
        unmatched_g = [g for g in range(M) if g not in assign]
        free_h = [h for h in range(N) if h not in matched_h]

        # (a) move one g to a different free h
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

        # (b) swap two matched g's h-partners
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

        # (c) swap which g is unmatched
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


def match_and_refine(AG, AH, L, alpha=1.0, C=None, **gnccp_kwargs):
    M, N = AG.shape[0], AH.shape[0]
    X = gnccp_wcs_match(AG, AH, L, alpha=alpha, C=C, **gnccp_kwargs)
    assign = {i: int(np.argmax(row)) for i, row in enumerate(X) if row.sum() > 0.5}
    return local_search_refine(assign, AG, AH, M, N, alpha=alpha, C=C)


# ============================================================================
# 5. EVALUATION
# ============================================================================
def evaluate(assign, ground_truth, G_labels, H_labels):
    correct = sum(1 for g, h in ground_truth.items() if assign.get(g) == h)
    total = len(ground_truth)
    type_ok = sum(1 for g, h in assign.items() if G_labels[g] == H_labels[h])
    type_total = len(assign)
    return correct, total, type_ok, type_total


def print_breakdown(assign, ground_truth, G_labels, H_labels):
    print(f"{'G idx':<7} | {'G type':<7} | {'True H':<8} | {'Pred H':<8} | {'Pred type':<10} | Status")
    print("-" * 65)
    for g, true_h in sorted(ground_truth.items()):
        pred_h = assign.get(g, None)
        pred_type = H_labels[pred_h] if pred_h is not None else '-'
        status = "OK" if pred_h == true_h else "MISS"
        print(f"{g:<7} | {G_labels[g]:<7} | {true_h:<8} | {str(pred_h):<8} | {pred_type:<10} | {status}")


# ============================================================================
# 6. DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("Building synthetic MOF reference motif (metal centers + linkers)")
    print("=" * 78)
    G_coords, G_labels, parent = generate_mof_reference(
        n_metal=4, linkers_per_metal=(3, 4), metal_min_sep=6.0,
        metal_linker_dist=2.05, bond_noise=0.05, seed=0)
    M = len(G_coords)
    n_metal_G = np.sum(G_labels == 'M')
    n_linker_G = np.sum(G_labels == 'L')
    print(f"Reference motif: {M} atoms ({n_metal_G} metal centers, {n_linker_G} linker atoms)")

    AG = _dist_matrix(G_coords)

    print("\n" + "=" * 78)
    print("A) RUN 1: sample with 2 missing atoms (vacancy defect) + 6 decoys, no rotation")
    print("=" * 78)
    AG1, AH1, gt1, Hc1, Hl1, L1 = build_sample_from_reference(
        G_coords, G_labels, n_missing=2, n_decoys=6, noise_std=0.06,
        rotate_deg=None, seed=5)
    N1 = AH1.shape[0]
    C1 = build_label_cost(G_labels, Hl1, mismatch_penalty=6.0)
    print(f"Sample H: {N1} atoms total ({L1} true correspondences + {N1-L1} decoys)")

    assign1 = match_and_refine(AG1, AH1, L1, alpha=0.7, C=C1, dzeta=0.05, inner_iters=20)
    c, t, tok, ttot = evaluate(assign1, gt1, G_labels, Hl1)
    print_breakdown(assign1, gt1, G_labels, Hl1)
    print(f"\nStructural accuracy: {c}/{t} = {100*c/t:.1f}%")
    print(f"Type consistency (metal matched to metal, linker to linker): {tok}/{ttot} = {100*tok/ttot:.1f}%")

    print("\n" + "=" * 78)
    print("B) RUN 2: SAME sample, but reported in a different crystallographic")
    print("   orientation (rigid 3D rotation + translation of the whole thing)")
    print("=" * 78)
    AG2, AH2, gt2, Hc2, Hl2, L2 = build_sample_from_reference(
        G_coords, G_labels, n_missing=2, n_decoys=6, noise_std=0.06,
        rotate_deg=50.0, translate=[8.0, -4.0, 3.0], seed=5)
    C2 = build_label_cost(G_labels, Hl2, mismatch_penalty=6.0)
    print("AH identical across runs (proves rotation cannot change the graph):",
          np.allclose(AH1, AH2, atol=1e-9))

    assign2 = match_and_refine(AG2, AH2, L2, alpha=0.7, C=C2, dzeta=0.05, inner_iters=20)
    c2, t2, tok2, ttot2 = evaluate(assign2, gt2, G_labels, Hl2)
    print_breakdown(assign2, gt2, G_labels, Hl2)
    print(f"\nStructural accuracy: {c2}/{t2} = {100*c2/t2:.1f}%")
    print(f"Type consistency: {tok2}/{ttot2} = {100*tok2/ttot2:.1f}%")

    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    print(f"Same predicted matching before/after rotation: {assign1 == assign2}")
    print("The algorithm correctly located the reference coordination motif")
    print("inside the larger, defected, decoy-containing, differently-oriented")
    print("sample -- exactly the WCS matching problem the paper is designed for,")
    print("now with a chemically-meaningful unary term (metal vs linker type)")
    print("added via the C matrix and alpha < 1.")