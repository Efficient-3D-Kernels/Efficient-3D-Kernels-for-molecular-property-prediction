import argparse

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment, minimize_scalar
import time
_start_time = time.time()
np.set_printoptions(precision=3, suppress=True, linewidth=150)


# ============================================================================
# 1. CONNECTIVITY EDGE BINNING
# ============================================================================

BIN_EDGES = np.array([0, 10, 60, 110, 160, 210, 260, 280, np.inf], dtype=float)


def weight_to_bin(weight):
    weight = float(weight)
    if weight < 0:
        raise ValueError("Edge weight cannot be negative.")
    bin_id = np.digitize(weight, BIN_EDGES[1:-1], right=False)
    return int(bin_id)


def bin_all_weights(weights):
    return np.array([weight_to_bin(w) for w in weights], dtype=int)


# ============================================================================
# 2. LOAD CONNECTIVITY DATA
# ============================================================================

def load_connectivity_csv(filename, source_col="source", target_col="target",
                           weight_col="weight", region_col="region"):
    df = pd.read_csv(filename)
    required = [source_col, target_col, weight_col, region_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable columns: {list(df.columns)}")

    df = df[[source_col, target_col, weight_col, region_col]].copy()
    df.columns = ["source", "target", "weight", "region"]
    df = df.dropna()
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["weight"] = df["weight"].astype(float)
    df["region"] = df["region"].astype(str)
    return df


# ============================================================================
# 3. SELECT SAME REGION
# ============================================================================

def select_region(df, region):
    region_df = df[df["region"] == region].copy()
    if len(region_df) == 0:
        raise ValueError(f"No connectivity edges found for region: {region}")
    return region_df


# ============================================================================
# 4. REMOVE ISOLATED NEURONS
# ============================================================================

def remove_isolated_neurons(df):
    connected = set(df["source"]) | set(df["target"])
    result = df[df["source"].isin(connected) & df["target"].isin(connected)].copy()
    return result


# ============================================================================
# 4b. OPTIONAL: SUBSAMPLE TO A SMALLER NEURON COUNT (for feasibility testing)
# ============================================================================

def subsample_by_neuron_count(df, max_neurons, seed=0):
    """
    If the region has more than max_neurons unique neurons, randomly keep
    only max_neurons of them (and only the edges where BOTH endpoints
    survive). This exists purely so you can test the GNCCP pipeline at a
    size where it will actually finish, before ever attempting the full
    ~4,000-10,000 node real graphs.

    IMPORTANT for cross-script comparison: this function is byte-for-byte
    identical to the one in gnccp_wcs_2d.py. Given the same input CSV,
    the same region, the same max_neurons, and the same seed, both
    scripts will keep the EXACT same set of neurons (sorted neuron list +
    RandomState(seed).choice is deterministic). That's what lets you run
    both pipelines on "the same data" and compare their matches.
    """
    neurons = sorted(set(df["source"]) | set(df["target"]))
    if max_neurons is None or len(neurons) <= max_neurons:
        return df

    rng = np.random.RandomState(seed)
    keep = set(rng.choice(neurons, size=max_neurons, replace=False))
    result = df[df["source"].isin(keep) & df["target"].isin(keep)].copy()
    print(f"Subsampled from {len(neurons)} to {max_neurons} neurons "
          f"({len(result)} edges remain, some may now be isolated and get pruned next).")
    return result


# ============================================================================
# 5. NEURON ID <-> MATRIX INDEX
# ============================================================================

def create_neuron_index(df):
    neurons = sorted(set(df["source"]) | set(df["target"]))
    neuron_to_index = {neuron: i for i, neuron in enumerate(neurons)}
    index_to_neuron = {i: neuron for neuron, i in neuron_to_index.items()}
    return neuron_to_index, index_to_neuron


# ============================================================================
# 6. BUILD BINNED WEIGHTED COMBINATORIAL GRAPH
# ============================================================================

def build_combinatorial_graph(df, directed=False):
    df = remove_isolated_neurons(df)
    neuron_to_index, index_to_neuron = create_neuron_index(df)
    n = len(neuron_to_index)

    print(f"Number of non-isolated neurons: {n}")
    print(f"Number of connectivity edges: {len(df)}")

    A = np.zeros((n, n), dtype=float)
    for _, row in df.iterrows():
        i = neuron_to_index[row["source"]]
        j = neuron_to_index[row["target"]]
        bin_id = weight_to_bin(row["weight"])
        A[i, j] = bin_id
        if not directed:
            A[j, i] = bin_id

    return A, neuron_to_index, index_to_neuron


# ============================================================================
# 7. PRINT BIN DISTRIBUTION
# ============================================================================

def print_bin_distribution(df):
    bins = bin_all_weights(df["weight"].values)
    counts = np.bincount(bins, minlength=8)
    print("\nConnectivity weight distribution")
    print("--------------------------------")
    for bin_id in range(8):
        lower = BIN_EDGES[bin_id]
        upper = BIN_EDGES[bin_id + 1]
        interval = f">= {lower}" if np.isinf(upper) else f"{lower} - {upper}"
        print(f"Bin {bin_id}: {interval:<15} {counts[bin_id]} edges")


# ============================================================================
# 8. GRAPH STATISTICS
# ============================================================================

def graph_statistics(A):
    n = A.shape[0]
    number_edges = np.count_nonzero(A) // 2
    possible_edges = (n * (n - 1)) // 2
    density = number_edges / possible_edges if possible_edges > 0 else 0

    print("\nGraph statistics")
    print("----------------")
    print(f"Number of neurons : {n}")
    print(f"Number of edges   : {number_edges}")
    print(f"Density           : {density:.6f}")
    print(f"Matrix size       : {A.shape}")

    return {"neurons": n, "edges": number_edges, "density": density}


# ============================================================================
# 9. CONNECTIVITY-BASED NODE SIGNATURE
# ============================================================================

def connectivity_signature(A):
    n = A.shape[0]
    signatures = np.zeros((n, 8), dtype=float)
    for i in range(n):
        row = A[i]
        for bin_id in range(8):
            signatures[i, bin_id] = np.sum(row == bin_id)
    totals = signatures.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    signatures /= totals
    return signatures


# ============================================================================
# 10. NODE COST  (vectorized broadcasting -- unchanged, already O(M*N*8))
# ============================================================================

def build_node_cost(AG, AH):
    SA = connectivity_signature(AG)
    SB = connectivity_signature(AH)
    C = ((SA[:, None, :] - SB[None, :, :]) ** 2).sum(axis=2)
    max_value = C.max()
    if max_value > 0:
        C /= max_value
    return C


# ============================================================================
# 11-14. H1 / F OBJECTIVE AND GRADIENTS  (term1 rank-one optimized;
#         F_value/F_grad scale-corrected)
# ============================================================================

def h1_term1_and_grad(AG, X):
    """
    Rank-one-optimized term1 = trace((AG*AG) @ U) and its gradient, where
    U = X @ ones(N,N) @ X.T is never formed explicitly.

    U == outer(r, r), r = X.sum(axis=1) (row sums). term1 = r^T (AG*AG) r
    -- O(M^2) instead of the O(M*N^2 + M^2*N) naive matrix chain -- and
    its gradient collapses to one matrix-vector product broadcast across
    columns, O(M^2 + M*N) instead of O(M*N^2).
    """
    N = X.shape[1]
    r = X.sum(axis=1)                     # (M,)  O(M*N)
    AGsq = AG * AG
    v = AGsq @ r                          # (M,)  O(M^2)
    term1 = float(r @ v)                  # scalar
    term1_grad = 2.0 * np.outer(v, np.ones(N))   # (M,N) O(M*N)
    return term1, term1_grad


def H1_value(X, AG, AH):
    term1, _ = h1_term1_and_grad(AG, X)
    term2 = -2.0 * np.trace(AG @ X @ AH.T @ X.T)
    XAHXt = X @ AH @ X.T
    term3 = np.trace(XAHXt @ X @ AH.T @ X.T)
    return float(term1 + term2 + term3)


def H1_grad(X, AG, AH):
    _, term1 = h1_term1_and_grad(AG, X)
    term2 = -4.0 * AG @ X @ AH
    term3 = 4.0 * X @ AH @ X.T @ X @ AH
    return term1 + term2 + term3


def compute_h1_scale(AG, AH):
    """
    H1_value's raw magnitude scales with sum-of-squares of the graphs'
    entries (can be tens of thousands for real binned graphs), while the
    node-cost term C is normalized to a max of 1 per entry. This computes
    a scale so H1's contribution lands in the same rough order of
    magnitude as the node-cost term's, so alpha actually trades off the
    two the way it's meant to.
    """
    scale = float(np.sum(AG * AG) + np.sum(AH * AH))
    return scale if scale > 0 else 1.0


def F_value(X, AG, AH, alpha=0.7, C=None, h1_scale=1.0):
    value = alpha * (H1_value(X, AG, AH) / h1_scale)
    if C is not None:
        value += (1.0 - alpha) * np.trace(C.T @ X)
    return float(value)


def F_grad(X, AG, AH, alpha=0.7, C=None, h1_scale=1.0):
    G = alpha * (H1_grad(X, AG, AH) / h1_scale)
    if C is not None:
        G += (1.0 - alpha) * C
    return G


# ============================================================================
# 15-16. GNCCP OBJECTIVE / GRADIENT
# ============================================================================

def J_zeta_value(X, zeta, AG, AH, alpha=0.7, C=None, h1_scale=1.0):
    Fv = F_value(X, AG, AH, alpha, C, h1_scale)
    reg = np.trace(X.T @ X)
    if zeta >= 0:
        return float((1 - zeta) * Fv + zeta * reg)
    return float((1 + zeta) * Fv + zeta * reg)


def J_zeta_grad(X, zeta, AG, AH, alpha=0.7, C=None, h1_scale=1.0):
    G = F_grad(X, AG, AH, alpha, C, h1_scale)
    if zeta >= 0:
        return (1 - zeta) * G + 2 * zeta * X
    return (1 + zeta) * G + 2 * zeta * X


# ============================================================================
# 17. EXACT L-MATCH ASSIGNMENT
# ============================================================================

def linear_step(cost, L):
    M, N = cost.shape
    if L > min(M, N):
        raise ValueError("L cannot be greater than the number of neurons in either graph.")
    if L == 0:
        return np.zeros((M, N))

    K = M + N - L
    P = (np.max(np.abs(cost)) + 1) * (K + 1)

    augmented = np.full((K, K), P, dtype=float)
    augmented[:M, :N] = cost
    augmented[:M, N:] = 0
    augmented[M:, :N] = 0

    row_ind, col_ind = linear_sum_assignment(augmented)

    Y = np.zeros((M, N), dtype=float)
    for r, c in zip(row_ind, col_ind):
        if r < M and c < N:
            Y[r, c] = 1

    if int(round(Y.sum())) != L:
        raise RuntimeError("Assignment did not produce exactly L matches.")

    return Y


# ============================================================================
# 18. GNCCP + FRANK-WOLFE
# ============================================================================

def gnccp_wcs_match(AG, AH, L, alpha=0.7, C=None, dzeta=0.05, inner_iters=30, tol=1e-6,
                     verbose=False, auto_scale=True):
    M = AG.shape[0]
    N = AH.shape[0]

    h1_scale = compute_h1_scale(AG, AH) if auto_scale else 1.0

    X = np.full((M, N), L / (M * N), dtype=float)
    zeta = 1.0
    log = []

    while zeta >= -1.0 - 1e-10:
        for _ in range(inner_iters):
            grad = J_zeta_grad(X, zeta, AG, AH, alpha, C, h1_scale)
            Y = linear_step(grad, L)
            direction = Y - X

            def objective_lambda(lam, X=X, direction=direction, zeta=zeta):
                X_candidate = X + lam * direction
                return J_zeta_value(X_candidate, zeta, AG, AH, alpha, C, h1_scale)

            result = minimize_scalar(objective_lambda, bounds=(0, 1), method="bounded")
            lam = float(result.x)

            X_new = X + lam * direction
            change = np.linalg.norm(X_new - X)
            X = X_new

            if change < tol:
                break

        current_value = J_zeta_value(X, zeta, AG, AH, alpha, C, h1_scale)
        log.append({"zeta": zeta, "objective": current_value, "max_X": float(X.max()), "X_norm_sq": float(np.sum(X * X))})

        if verbose:
            print(f"zeta={zeta: .2f} | J={current_value:.6f} | max(X)={X.max():.4f}")

        zeta -= dzeta

    X_final = linear_step(-X, L)
    return X_final, X, log


# ============================================================================
# 18b. SELF-TEST: RELABELING INVARIANCE
# ============================================================================

def verify_relabeling_invariance(seed=0, M=6, N=8, verbose=True):
    rng = np.random.RandomState(seed)
    AG = rng.randint(0, 8, size=(M, M)).astype(float)
    AG = (AG + AG.T) / 2
    np.fill_diagonal(AG, 0)
    AH = rng.randint(0, 8, size=(N, N)).astype(float)
    AH = (AH + AH.T) / 2
    np.fill_diagonal(AH, 0)
    X = rng.rand(M, N)

    P = np.eye(M)[rng.permutation(M)]
    Q = np.eye(N)[rng.permutation(N)]
    AGp = P @ AG @ P.T
    AHp = Q @ AH @ Q.T
    Xp = P @ X @ Q.T

    v1 = H1_value(X, AG, AH)
    v2 = H1_value(Xp, AGp, AHp)
    ok = np.isclose(v1, v2)

    if verbose:
        print("\nRelabeling-invariance self-test")
        print("--------------------------------")
        print(f"H1(X ; AG, AH)        = {v1:.10f}")
        print(f"H1(PXQ^T ; AG', AH')  = {v2:.10f}")
        print("PASS" if ok else "FAIL")

    return ok


# ============================================================================
# 19. REAL-WORLD EXPERIMENT
# ============================================================================

def run_real_experiment(fly_A_file, fly_B_file, region, alpha=0.7, dzeta=0.05,
                         inner_iters=30, max_neurons=None, seed=0):
    print("\n" + "=" * 80)
    print("REAL-WORLD CONNECTOME EXPERIMENT (pure GNCCP, 2D structural matching only)")
    print("=" * 80)

    ok = verify_relabeling_invariance(verbose=False)
    if not ok:
        raise RuntimeError("H1 failed the relabeling-invariance self-test -- stopping before running on real data.")
    print("Relabeling-invariance self-test: PASS")

    print("\nLoading Fly A...")
    A_df = load_connectivity_csv(fly_A_file)
    print(f"Fly A total edges: {len(A_df)}")

    print("\nLoading Fly B...")
    B_df = load_connectivity_csv(fly_B_file)
    print(f"Fly B total edges: {len(B_df)}")

    print(f"\nSelecting SAME region: {region}")
    A_region = select_region(A_df, region)
    B_region = select_region(B_df, region)

    if max_neurons is not None:
        print(f"\nCapping each fly at {max_neurons} neurons for feasibility "
              f"(seed={seed}, same as gnccp_wcs_2d.py so both scripts see the same neurons)...")
        A_region = subsample_by_neuron_count(A_region, max_neurons, seed=seed)
        B_region = subsample_by_neuron_count(B_region, max_neurons, seed=seed + 1)

    A_region = remove_isolated_neurons(A_region)
    B_region = remove_isolated_neurons(B_region)

    print(f"\nFly A edges after removing isolated neurons: {len(A_region)}")
    print(f"Fly B edges after removing isolated neurons: {len(B_region)}")

    print("\nFly A weight bins")
    print_bin_distribution(A_region)
    print("\nFly B weight bins")
    print_bin_distribution(B_region)

    print("\nConstructing Fly A graph...")
    AG, A_to_idx, A_from_idx = build_combinatorial_graph(A_region)
    print("\nConstructing Fly B graph...")
    AH, B_to_idx, B_from_idx = build_combinatorial_graph(B_region)

    print("\nFly A")
    graph_statistics(AG)
    print("\nFly B")
    graph_statistics(AH)

    L = min(AG.shape[0], AH.shape[0])
    print(f"\nRequested matches L = {L}")
    print(f"(dzeta={dzeta} -> ~{int(2/dzeta)} zeta steps, up to {inner_iters} Hungarian "
          f"solves each = up to ~{int(2/dzeta) * inner_iters} total Hungarian solves)")

    h1_scale = compute_h1_scale(AG, AH)
    print(f"\nH1 scale correction factor: {h1_scale:.2f} "
          f"(alpha={alpha} now actually trades off structure vs. node-cost, "
          f"rather than H1 silently dominating regardless of alpha)")

    print("\nBuilding connectivity-based node compatibility...")
    C = build_node_cost(AG, AH)

    print("\nRunning GNCCP (2D structural matching, rank-one term1, scale-corrected alpha)...")
    X_final, X_soft, log = gnccp_wcs_match(
        AG, AH, L=L, alpha=alpha, C=C, dzeta=dzeta, inner_iters=inner_iters, tol=1e-6,
        verbose=True, auto_scale=True,
    )

    matches = []
    for i in range(X_final.shape[0]):
        j = np.argmax(X_final[i])
        if X_final[i, j] > 0.5:
            matches.append({
                "fly_A_neuron": A_from_idx[i],
                "fly_B_neuron": B_from_idx[j],
                "score": X_soft[i, j],
            })

    matches_df = pd.DataFrame(matches)
    print("\nFirst 20 predicted matches")
    print(matches_df.head(20))

    matches_df.to_csv("predicted_neuron_matches_pure_2d.csv", index=False)
    print("\nSaved: predicted_neuron_matches_pure_2d.csv")

    return AG, AH, X_final, matches_df


# ============================================================================
# 20. MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure GNCCP connectome matching on real data (2D structural only)")
    parser.add_argument("--fly_A_file", type=str, default="flyA_hemibrain_MB.csv",
                         help="Real Fly A CSV (source,target,weight,region)")
    parser.add_argument("--fly_B_file", type=str, default="flyB_fafb_MB.csv",
                         help="Real Fly B CSV (source,target,weight,region)")
    parser.add_argument("--region", type=str, default="mushroom_body")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--dzeta", type=float, default=0.05)
    parser.add_argument("--inner_iters", type=int, default=30)
    parser.add_argument("--max_neurons", type=int, default=300,
                         help="Cap on neurons per fly for feasibility. Pass a very large "
                              "number to run on the full real graph -- only after confirming "
                              "timing at this smaller size is acceptable.")
    parser.add_argument("--seed", type=int, default=0,
                         help="Subsampling RNG seed. Use the SAME value here and in "
                              "gnccp_wcs_2d.py's --seed to guarantee both scripts operate "
                              "on the identical subsampled neuron set.")
    args = parser.parse_args()

    run_real_experiment(
    fly_A_file=args.fly_A_file,
    fly_B_file=args.fly_B_file,
    region=args.region,
    alpha=args.alpha,
    dzeta=args.dzeta,
    inner_iters=args.inner_iters,
    max_neurons=args.max_neurons,
    seed=args.seed,
)
    print(f"\nTOTAL RUNTIME: {time.time() - _start_time:.2f} seconds")