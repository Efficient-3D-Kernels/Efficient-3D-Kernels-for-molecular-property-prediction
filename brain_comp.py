"""
Both datasets live on the SAME neuPrint server: neuprint-cns.janelia.org
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment #import hungrian algorithm from scipy

_start_time = time.time()
np.set_printoptions(precision=3, suppress=True, linewidth=180)

# 0. FETCH REAL DATA FROM NEUPRINT (both flies, same server)

SERVER = "neuprint-cns.janelia.org"

DATASET_A = "hemibrain:v1.2.2"
ROI_NAMES_A = ["MB(R)"]

DATASET_B = "flywire-fafb:v783b"
ROI_NAMES_B = ["MB_CA_R", "MB_PED_R", "MB_ML_R", "MB_VL_R"]


def _get_token(token=None): # get the auth token from the neuprint site
    token = token or os.environ.get("8e17a61d4f9f7436638a76a0e4885056b574c976f5e8b7bd0e01abcd984ad503")
    if not token:
        raise RuntimeError(
            "No neuPrint token supplied. Set the NEUPRINT_TOKEN environment "
            "variable or pass token=... explicitly."
        )
    return token


def fetch_and_save(dataset, roi_names, out_csv, server=SERVER, token=None):
    from neuprint import Client, fetch_adjacencies

    client = Client(server, dataset=dataset, token=_get_token(token))
    _, conn_df = fetch_adjacencies(rois=roi_names, include_nonprimary=True, client=client)

    conn_df = conn_df[conn_df["roi"].isin(roi_names)].copy() # take only the needed info and ignore the rest
    conn_df = (
        conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"]  #compares and make sure there is only 1 unique edge for a neuron based on the weights 101->205 has 4+2=6 and 101->102 have 4 then take former
        .sum()                                                                     # sum the weight for suplicate records
        .rename(columns={"bodyId_pre": "source", "bodyId_post": "target"})
    )
    conn_df["region"] = "mushroom_body"
    conn_df = conn_df[["source", "target", "weight", "region"]]
    conn_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}: {len(conn_df)} unique edges")
    return conn_df


def fetch_soma_coordinates(dataset, roi_names, server=SERVER, token=None): # it goes to the neuprint and get the soma cordinates of the neurons that we saved
    from neuprint import Client, fetch_neurons, NeuronCriteria as NC

    client = Client(server, dataset=dataset, token=_get_token(token))
    criteria = NC(rois=roi_names, client=client)
    neuron_df, _ = fetch_neurons(criteria, client=client)

    if "somaLocation" in neuron_df.columns:
        coords = neuron_df["somaLocation"].dropna().apply(pd.Series)
        coords.columns = ["x", "y", "z"][: coords.shape[1]]
        coords["bodyId"] = neuron_df.loc[coords.index, "bodyId"].values
    elif {"somaLocation_x", "somaLocation_y", "somaLocation_z"}.issubset(neuron_df.columns):
        coords = neuron_df[["bodyId", "somaLocation_x", "somaLocation_y", "somaLocation_z"]].dropna()
        coords.columns = ["bodyId", "x", "y", "z"]
    else:
        print(f"WARNING: no soma coordinate columns found for {dataset}; Stage 2 will be skipped for this fly.")
        return pd.DataFrame(columns=["bodyId", "x", "y", "z"])

    coords["bodyId"] = coords["bodyId"].astype(str)
    return coords[["bodyId", "x", "y", "z"]]


def fetch_real_datasets(fetch_coords=False): # infor with the corrdinates are saved in .coor files and dataset is also saved
    
    fetch_and_save(DATASET_A, ROI_NAMES_A, "flyA_hemibrain_MB.csv")
    fetch_and_save(DATASET_B, ROI_NAMES_B, "flyB_fafb_MB.csv")
    if fetch_coords:
        fetch_soma_coordinates(DATASET_A, ROI_NAMES_A).to_csv("flyA_hemibrain_MB_coords.csv", index=False)
        fetch_soma_coordinates(DATASET_B, ROI_NAMES_B).to_csv("flyB_fafb_MB_coords.csv", index=False)


# 1. CONFIGURATION  bins are used to represent adjacency matrices and graph representations
N_BINS = 8
BIN_EDGES = np.array([0, 10, 60, 110, 160, 210, 260, 280, np.inf], dtype=float)
BIN_RANGES = ["0-10", "10-60", "60-110", "110-160", "160-210", "210-260", "260-280", ">=280"]


def weight_to_bin(weight):
    weight = float(weight)
    if weight < 0:
        raise ValueError("Connectivity weight cannot be negative.") #weights are put into bins
    for b in range(N_BINS):
        if weight < BIN_EDGES[b + 1]:                               #A-B will have a connection weight/strength this is put into corresponding bins to make it easier to make adjacency matrices
            return b
    return N_BINS - 1


def weights_to_bins(weights):
    return np.array([weight_to_bin(w) for w in weights], dtype=int)

# 2. LOAD / FILTER  

def load_connectivity_csv(filename, source_col="source", target_col="target",
                           weight_col="weight", region_col="region"):
    df = pd.read_csv(filename)
    required = [source_col, target_col, weight_col, region_col] #loads the connectivity csv file, read it, check the required columns, keep only required columns, remove missing values
    missing = [c for c in required if c not in df.columns]      #convert ID to string, weight to numbers, return clean data
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")
    df = df[required].copy()
    df.columns = ["source", "target", "weight", "region"]
    df = df.dropna()
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["weight"] = df["weight"].astype(float)
    df["region"] = df["region"].astype(str)
    return df


def select_region(df, region):
    result = df[df["region"] == str(region)].copy() # only takes the mushroom body region
    if len(result) == 0:
        raise ValueError(f"No edges found for region '{region}'.")
    return result


def remove_isolated_neurons(df):        #remove isolated neurons that is those with no connection
    connected = set(df["source"]) | set(df["target"])
    return df[df["source"].isin(connected) & df["target"].isin(connected)].copy()


def subsample_by_neuron_count(df, max_neurons, seed=0):
    neurons = sorted(set(df["source"]) | set(df["target"]))
    if max_neurons is None or len(neurons) <= max_neurons:  #run a sub sample, there
        return df
    rng = np.random.RandomState(seed)
    keep = set(rng.choice(neurons, size=max_neurons, replace=False))
    result = df[df["source"].isin(keep) & df["target"].isin(keep)].copy()
    print(f"Subsampled from {len(neurons)} to {max_neurons} neurons "
          f"({len(result)} edges remain, some may now be isolated and get pruned next).")
    return result


def create_neuron_index(df):
    neurons = sorted(set(df["source"]) | set(df["target"])) #index is made from neuron ID since it is IDs are large so it is easier to make matrix
    neuron_to_index = {n: i for i, n in enumerate(neurons)}
    index_to_neuron = {i: n for n, i in neuron_to_index.items()}
    return neuron_to_index, index_to_neuron


def build_combinatorial_graph(df):
    """AG/AH in the paper's notation. Stores bin+1 (values 1-8); 0 means
    'no edge'."""
    df = remove_isolated_neurons(df) #remove isolated neurons
    neuron_to_index, index_to_neuron = create_neuron_index(df)
    n = len(neuron_to_index)
    A = np.zeros((n, n), dtype=np.float64)
    for _, row in df.iterrows():
        i = neuron_to_index[row["source"]]
        j = neuron_to_index[row["target"]]
        b = weight_to_bin(row["weight"]) + 1 #adjacency matrix
        A[i, j] = b
        A[j, i] = b
    return A, neuron_to_index, index_to_neuron


def graph_statistics(A, name="Graph"):
    n = A.shape[0]
    edges = np.count_nonzero(np.triu(A, k=1))
    possible = (n * (n - 1)) // 2
    density = edges / possible if possible > 0 else 0  #how big and how connected your graph is.
    print(f"\n{name} statistics")
    print("-" * 50)
    print(f"Neurons : {n}")
    print(f"Edges   : {edges}")
    print(f"Density : {density:.6f}")
    return {"neurons": n, "edges": edges, "density": density}


def print_bin_distribution(df, name="Graph"):
    bins = weights_to_bins(df["weight"].values) #converts the cleaned connectivity data into an adjacency matrix
    counts = np.bincount(bins, minlength=N_BINS)#Each neuron is assigned an index, and each connectivity weight is converted into one of eight bins.
    print(f"\n{name} edge-weight distribution")#The bin number is stored in the matrix, while 0 represents no edge
    print("-" * 60)
    for i in range(N_BINS):
        print(f"Bin {i}: {BIN_RANGES[i]:>10} -> {counts[i]} edges")


# 3. NODE (ATOM/NEURON) COST MATRIX C #low cost=more similar
def connectivity_signature(A):
    n = A.shape[0]
    S = np.zeros((n, N_BINS), dtype=np.float64) #describe each neuron by how many conection it has in each bin
    for b in range(N_BINS):
        S[:, b] = np.sum(A == (b + 1), axis=1) #normalize so both A and B have same to compare,
    totals = S.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    return S / totals


def build_node_cost(AG, AH):
    SA = connectivity_signature(AG) #connectivity signature(how many connections are there in a bin) for A and B, the compare every neuron in both graph to each other
    SB = connectivity_signature(AH)
    norm_A = np.sum(SA ** 2, axis=1) 
    norm_B = np.sum(SB ** 2, axis=1)
    C = norm_A[:, None] + norm_B[None, :] - 2.0 * (SA @ SB.T)   # O(M*N) memory, not O(M*N*8) Cij=∣∣SA(i)−SB(j)∣∣^2 cost is calculated
    np.maximum(C, 0, out=C)
    return C
#connectivity_signature() represents each neuron by the proportion of its connections falling into each weight bin. 
# Then build_node_cost() compares every Fly A neuron with every Fly B neuron using the squared Euclidean distance between their signatures. 
# Thus, matrix C tells GNCCP which neuron pairs have similar local connectivity patterns, where a lower cost means greater similarity.


# 4.FROBENIUS NORMALIZATION (SCALE INVARIANCE)
def frobenius_normalize(A):
    norm = np.linalg.norm(A)
    return A / norm if norm > 0 else A # normalizes the Ag and Ah and make it into a common scale for comparison


def minmax_normalize_cost(C):
    cmin, cmax = C.min(), C.max()
    if cmax - cmin < 1e-15:
        return np.zeros_like(C)
    return (C - cmin) / (cmax - cmin)
#Algorithm 1, line 2: Ĉ ← (C - Cmin) / (Cmax - Cmin).
#Min-max normalization scales all neuron matching costs between 0 and 1. 
# A value near 0 means the pair has low matching cost, while a value near 1 means high matching cost.”


# 5.RANK-ONE FACTORIZATION OF U
def rank_one_U(X): #possible X matrices
    r = X.sum(axis=1)
    return r, np.outer(r, r)


# 6. THEOREM 1 -- STRUCTURAL OBJECTIVE AND ITS GRADIENT
def structural_value_and_grad(X, AG, AH):
    N = X.shape[1]
    r, U = rank_one_U(X)                       # Lemma 1, O(M^2) instead of O(M*N^2+M^2*N)
    XAHXt = X @ AH @ X.T
    R = U * AG - XAHXt                          # residual, Theorem 1
    f_val = float(np.sum(R * R))

    RAG = R * AG                                 # (M,M)
    ones_col = RAG.sum(axis=1)                   # (M,)
    term_a = 4.0 * np.outer(ones_col, np.ones(N))
    term_b = -4.0 * R @ X @ AH
    grad = term_a + term_b
    return f_val, grad


def wcs_objective_and_grad(X, AG, AH, C, alpha):
    """Definition 2: F(X) = alpha*f(X) + (1-alpha)*trace(C^T X)."""
    f_val, f_grad = structural_value_and_grad(X, AG, AH)
    F_val = alpha * f_val + (1.0 - alpha) * float(np.trace(C.T @ X))
    F_grad = alpha * f_grad + (1.0 - alpha) * C
    return F_val, F_grad


# ============================================================
# 7. DEFINITION 3 / PROPOSITION 2 -- GNCCP OBJECTIVE Jζ
# ============================================================

def J_zeta_and_grad(X, zeta, AG, AH, C, alpha):
    F_val, F_grad = wcs_objective_and_grad(X, AG, AH, C, alpha)
    reg_val = float(np.sum(X * X))
    reg_grad = 2.0 * X
    if zeta >= 0:
        return (1 - zeta) * F_val + zeta * reg_val, (1 - zeta) * F_grad + zeta * reg_grad
    return (1 + zeta) * F_val + zeta * reg_val, (1 + zeta) * F_grad + zeta * reg_grad


# ============================================================
# 8. DEFINITION 4 -- LINEAR MINIMIZATION ORACLE
# ============================================================

def linear_minimization_oracle(cost, L):
    M, N = cost.shape
    if L > min(M, N):
        raise ValueError(f"L={L} is larger than min(M,N)={min(M, N)}")
    if L == 0:
        return np.zeros_like(cost)

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
            Y[r, c] = 1.0

    if int(round(Y.sum())) != L:
        raise RuntimeError("Assignment did not produce exactly L matches.")
    return Y


# 9. PROPOSITION 5 -- SINKHORN-KNOPP PROJECTION ONTO D_L

def sinkhorn_project(X, L, iters=20, eps=1e-12):
    X = np.clip(X, 0, None)
    total = X.sum()
    if total > eps:
        X = X * (L / total)
    for _ in range(iters):
        row_sums = X.sum(axis=1, keepdims=True)
        over_rows = row_sums > 1.0
        if np.any(over_rows):
            X = np.where(over_rows, X / np.maximum(row_sums, eps), X)
        col_sums = X.sum(axis=0, keepdims=True)
        over_cols = col_sums > 1.0
        if np.any(over_cols):
            X = np.where(over_cols, X / np.maximum(col_sums, eps), X)
    return X


# 10. ARMIJO LINE SEARCH
def armijo_line_search(X, Y, zeta, AG, AH, C, alpha, c1=1e-4, backtrack=0.5, max_steps=20):
    direction = Y - X
    J_X, grad_X = J_zeta_and_grad(X, zeta, AG, AH, C, alpha)
    directional_deriv = float(np.sum(grad_X * direction))

    lam = 1.0
    for _ in range(max_steps):
        X_candidate = X + lam * direction
        J_candidate, _ = J_zeta_and_grad(X_candidate, zeta, AG, AH, C, alpha)
        if J_candidate <= J_X + c1 * lam * directional_deriv:
            return lam
        lam *= backtrack
    return lam


# 11. ALGORITHM 1, STAGE 1 -- 2D STRUCTURAL WCS-GNCCP MATCHING
def wcs_gnccp_match(AG, AH, L, alpha=0.7, gamma=0.5,
                     zeta_start=1.0, zeta_end=-1.0, dzeta=0.05,
                     inner_iters_fw=30, duality_gap_tol=1e-4,
                     sinkhorn_iters=20, verbose=False):
    """
    Algorithm 1, Stage 1 only (2D structural matching). Returns:
      X_final : discretized partial permutation (M,N)
      k_score : similarity kernel value in (0,1], Proposition 3
      history : per-zeta-step diagnostics
    """
    M, N = AG.shape[0], AH.shape[0]

    AGh = frobenius_normalize(AG)        # Prop 4
    AHh = frobenius_normalize(AH)

    C_raw = build_node_cost(AG, AH)
    C = minmax_normalize_cost(C_raw)     # Algorithm 1, line 2

    X = np.full((M, N), L / (M * N), dtype=float)   # Prop 1(1)
    zeta = zeta_start
    history = []

    while zeta >= zeta_end - 1e-9:
        for _ in range(inner_iters_fw):
            J_val, grad = J_zeta_and_grad(X, zeta, AGh, AHh, C, alpha)
            Y = linear_minimization_oracle(grad, L)              # Def. 4
            gap = float(np.sum(grad * (X - Y)))                  # Theorem 2
            if gap < duality_gap_tol:
                break
            lam = armijo_line_search(X, Y, zeta, AGh, AHh, C, alpha)
            X = sinkhorn_project(X + lam * (Y - X), L, iters=sinkhorn_iters)  # Prop 5

        history.append({"zeta": zeta, "J": J_val, "gap": gap, "max_X": float(X.max())})
        if verbose:
            print(f"zeta={zeta:6.2f} | J={J_val:12.4f} | gap={gap:10.4f} | max(X)={X.max():.4f}")
        zeta -= dzeta

    X_final = linear_minimization_oracle(-X, L)                  # Algorithm 1, line 16

    F_star, _ = wcs_objective_and_grad(X_final, AGh, AHh, C, alpha)
    k_score = float(np.exp(-gamma * F_star / max(L, 1)))         # Prop 3, line 19

    return X_final, k_score, history


# 12. PROPOSITION 6 -- RELABELING INVARIANCE SELF-TEST
def verify_relabeling_invariance(seed=0, M=6, N=8, alpha=0.7, verbose=True):
    rng = np.random.RandomState(seed)
    AG = rng.randint(0, 8, size=(M, M)).astype(float)
    AG = (AG + AG.T) / 2
    np.fill_diagonal(AG, 0)
    AH = rng.randint(0, 8, size=(N, N)).astype(float)
    AH = (AH + AH.T) / 2
    np.fill_diagonal(AH, 0)
    C = rng.rand(M, N)
    X = rng.rand(M, N)

    P = np.eye(M)[rng.permutation(M)]
    Q = np.eye(N)[rng.permutation(N)]
    AGp, AHp, Cp, Xp = P @ AG @ P.T, Q @ AH @ Q.T, P @ C @ Q.T, P @ X @ Q.T

    F1, _ = wcs_objective_and_grad(X, AG, AH, C, alpha)
    F2, _ = wcs_objective_and_grad(Xp, AGp, AHp, Cp, alpha)
    ok = np.isclose(F1, F2)

    if verbose:
        print("\nRelabeling-invariance self-test (Proposition 6)")
        print("------------------------------------------------")
        print(f"F(X; AG, AH, C)          = {F1:.10f}")
        print(f"F(PXQ^T; AG', AH', C')   = {F2:.10f}")
        print("PASS" if ok else "FAIL")
    return ok


# ============================================================
# 13. STAGE 2 (OPTIONAL, OFF BY DEFAULT) -- KABSCH ALIGNMENT
#     (Theorem 4 / Corollary 3). Not invoked unless --run_stage2 is
#     passed AND both coordinate CSVs are supplied -- see section 15.
# ============================================================

def kabsch_rmsd(P_pts, Q_pts):
    """Theorem 4: optimal rotation R* minimizing RMSD between matched
    point sets, via SVD of the 3x3 cross-covariance matrix. O(L) given
    the neuron map from Stage 1 (Corollary 3)."""
    Pc = P_pts - P_pts.mean(axis=0)
    Qc = Q_pts - Q_pts.mean(axis=0)
    H = Pc.T @ Qc
    W, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ W.T))
    D = np.diag([1, 1, d])
    R_star = Vt.T @ D @ W.T
    P_aligned = (R_star @ Pc.T).T
    rmsd = float(np.sqrt(np.mean(np.sum((P_aligned - Qc) ** 2, axis=1))))
    return rmsd, R_star


def stage2_kabsch(match_pairs, coords_A, coords_B, A_from_idx, B_from_idx):
    coordA_map = coords_A.set_index("bodyId")[["x", "y", "z"]]
    coordB_map = coords_B.set_index("bodyId")[["x", "y", "z"]]

    P_pts, Q_pts = [], []
    for i, j in match_pairs:
        bidA, bidB = A_from_idx[i], B_from_idx[j]
        if bidA in coordA_map.index and bidB in coordB_map.index:
            P_pts.append(coordA_map.loc[bidA].values.astype(float))
            Q_pts.append(coordB_map.loc[bidB].values.astype(float))

    if len(P_pts) < 3:
        print(f"Stage 2 skipped: only {len(P_pts)} matched neurons have known "
              f"soma coordinates on both sides (need >= 3 for Kabsch).")
        return None

    rmsd, R_star = kabsch_rmsd(np.array(P_pts), np.array(Q_pts))
    return {"rmsd": rmsd, "n_points": len(P_pts), "rotation": R_star}


def hybrid_kernel(k2D, k3D, beta=0.5):
    """Definition 6."""
    if k3D is None:
        return k2D
    return (1 - beta) * k2D + beta * k3D


# ============================================================
# 14. REAL-WORLD EXPERIMENT -- STOPS AT 2D BY DEFAULT
# ============================================================

def run_real_experiment(fly_A_file, fly_B_file, region, alpha=0.7, gamma=0.5,
                         dzeta=0.05, inner_iters_fw=30, max_neurons=None, seed=0,
                         run_stage2=False, coords_A_file=None, coords_B_file=None,
                         gamma_3d=0.1, beta=0.5):
    """
    By default (run_stage2=False) this stops after Stage 1 -- the 2D
    structural WCS-GNCCP match -- exactly like gnccp_pure_2d.py, so the
    two scripts' outputs (predicted_neuron_matches_*.csv) are directly
    comparable given the same fly_A_file/fly_B_file/region/max_neurons/seed.
    Pass run_stage2=True with both coords_*_file paths to additionally
    run the 3D Kabsch refinement.
    """
    print("=" * 80)
    print("REAL-WORLD -- WCS-GNCCP MATCHING (paper Algorithm 1, Stage 1: 2D only by default)")
    print("=" * 80)

    ok = verify_relabeling_invariance(verbose=False)
    if not ok:
        raise RuntimeError("Relabeling-invariance self-test failed -- stopping before running on real data.")
    print("Relabeling-invariance self-test (Prop. 6): PASS")

    A_df = load_connectivity_csv(fly_A_file)
    B_df = load_connectivity_csv(fly_B_file)

    A_region = select_region(A_df, region)
    B_region = select_region(B_df, region)

    if max_neurons is not None:
        print(f"\nCapping each fly at {max_neurons} neurons for feasibility "
              f"(seed={seed}, same as gnccp_pure_2d.py so both scripts see the same neurons)...")
        A_region = subsample_by_neuron_count(A_region, max_neurons, seed=seed)
        B_region = subsample_by_neuron_count(B_region, max_neurons, seed=seed + 1)

    A_region = remove_isolated_neurons(A_region)
    B_region = remove_isolated_neurons(B_region)
    print(f"\nFly A edges: {len(A_region)}  |  Fly B edges: {len(B_region)}")

    print_bin_distribution(A_region, "Fly A")
    print_bin_distribution(B_region, "Fly B")

    AG, A_to_idx, A_from_idx = build_combinatorial_graph(A_region)
    AH, B_to_idx, B_from_idx = build_combinatorial_graph(B_region)
    graph_statistics(AG, "Fly A")
    graph_statistics(AH, "Fly B")

    M, N = AG.shape[0], AH.shape[0]
    L = min(M, N)
    print(f"\nL = {L}  (Tzeta ~ {int(2/dzeta)} steps x T_FW <= {inner_iters_fw} inner iters "
          f"x Hungarian O(min(M,N)^3) per Theorem 3 -- this is the paper's own stated cost)")

    print("\nRunning WCS-GNCCP Stage 1 (2D structural matching)...")
    X_final, k2D, history = wcs_gnccp_match(
        AG, AH, L=L, alpha=alpha, gamma=gamma, dzeta=dzeta,
        inner_iters_fw=inner_iters_fw, verbose=True,
    )
    print(f"\nTOTAL RUNTIME SO FAR: {time.time() - _start_time:.2f} seconds")

    match_pairs = [(i, int(np.argmax(X_final[i]))) for i in range(M) if X_final[i].max() > 0.5]
    print(f"\n2D similarity kernel k2D = {k2D:.6f}  ({len(match_pairs)} matches)")

    k3D = None
    stage2_result = None
    if run_stage2:
        if coords_A_file and coords_B_file:
            coords_A = pd.read_csv(coords_A_file)
            coords_B = pd.read_csv(coords_B_file)
            coords_A["bodyId"] = coords_A["bodyId"].astype(str)
            coords_B["bodyId"] = coords_B["bodyId"].astype(str)
            stage2_result = stage2_kabsch(match_pairs, coords_A, coords_B, A_from_idx, B_from_idx)
            if stage2_result is not None:
                # Definition 6: k3D = exp(-gamma_3D * RMSD) -- a SEPARATE
                # bandwidth from the 2D kernel's gamma, per the paper.
                k3D = float(np.exp(-gamma_3d * stage2_result["rmsd"]))
                print(f"3D Kabsch RMSD = {stage2_result['rmsd']:.3f} over {stage2_result['n_points']} points "
                      f"-> k3D = {k3D:.6f}")
        else:
            print("\n--run_stage2 was set but coords_A_file/coords_B_file were not both "
                  "supplied -- skipping Stage 2.")
    else:
        print("\nStage 2 (3D Kabsch) not requested -- stopping at 2D structural matching, "
              "matching gnccp_pure_2d.py's scope for direct comparison.")

    k_hybrid = hybrid_kernel(k2D, k3D, beta=beta)
    if k3D is not None:
        print(f"\nHybrid kernel k_hybrid = {k_hybrid:.6f}  (Definition 6, beta={beta})")

    matches = []
    for i, j in match_pairs:
        matches.append({"fly_A_neuron": A_from_idx[i], "fly_B_neuron": B_from_idx[j]})
    matches_df = pd.DataFrame(matches)
    matches_df.to_csv("predicted_neuron_matches_wcs_2d.csv", index=False)
    print(f"\nSaved predicted_neuron_matches_wcs_2d.csv ({len(matches_df)} matches)")
    print(matches_df.head(20))

    # Export interactive visualization data for UI
    export_visualization_data(
        AG=AG, AH=AH, X_final=X_final,
        A_from_idx=A_from_idx, B_from_idx=B_from_idx,
        match_pairs=match_pairs, k2D=k2D, history=history,
        alpha=alpha, gamma=gamma, max_neurons=max_neurons, seed=seed,
    )

    return {
        "AG": AG, "AH": AH, "X_final": X_final,
        "k2D": k2D, "k3D": k3D, "k_hybrid": k_hybrid,
        "matches": matches_df, "history": history,
    }


def export_visualization_data(AG, AH, X_final, A_from_idx, B_from_idx,
                              match_pairs, k2D, history=None,
                              alpha=0.7, gamma=0.5, max_neurons=None, seed=0,
                              out_json="viz_data.json", out_js="viz_data.js"):
    """Export complete graph matrices, X correspondence, signatures, and matches for the UI."""
    M, N = AG.shape[0], AH.shape[0]
    L = min(M, N)
    SA = connectivity_signature(AG)
    SB = connectivity_signature(AH)
    C_raw = build_node_cost(AG, AH)
    C_norm = minmax_normalize_cost(C_raw)

    def edges_from_A(A):
        iu = np.triu_indices(A.shape[0], k=1)
        vals = A[iu]
        mask = vals > 0
        src = iu[0][mask].tolist()
        tgt = iu[1][mask].tolist()
        bins = (vals[mask] - 1).astype(int).tolist()
        return [{"s": int(s), "t": int(t), "b": int(b)} for s, t, b in zip(src, tgt, bins)]

    a_to_b = {int(i): int(j) for i, j in match_pairs}
    b_to_a = {int(j): int(i) for i, j in match_pairs}

    hist_clean = []
    if history:
        for h in history:
            hist_clean.append({
                "zeta": float(h.get("zeta", 0)),
                "J": float(h.get("J", 0)),
                "gap": float(h.get("gap", 0)),
                "max_X": float(h.get("max_X", 0))
            })

    viz = {
        "M": int(M), "N": int(N), "L": int(L),
        "A_ids": [str(A_from_idx[i]) for i in range(M)],
        "B_ids": [str(B_from_idx[j]) for j in range(N)],
        "AG_edges": edges_from_A(AG),
        "AH_edges": edges_from_A(AH),
        "AG_matrix": AG.tolist(),
        "AH_matrix": AH.tolist(),
        "AG_deg": np.count_nonzero(AG, axis=1).astype(int).tolist(),
        "AH_deg": np.count_nonzero(AH, axis=1).astype(int).tolist(),
        "SA": SA.tolist(),
        "SB": SB.tolist(),
        "C": C_norm.tolist(),
        "X": X_final.tolist(),
        "matches": [{"i": int(i), "j": int(j)} for i, j in match_pairs],
        "aToB": a_to_b,
        "bToA": b_to_a,
        "k2D": float(k2D),
        "bin_ranges": BIN_RANGES,
        "alpha": float(alpha), "gamma": float(gamma),
        "max_neurons": int(max_neurons) if max_neurons is not None else M,
        "seed": int(seed),
        "datasetA": "hemibrain:v1.2.2 (MB(R))",
        "datasetB": "flywire-fafb:v783b (MB_CA_R, MB_PED_R, MB_ML_R, MB_VL_R)",
        "history": hist_clean,
    }

    with open(out_json, "w") as f:
        json.dump(viz, f)

    with open(out_js, "w") as f:
        f.write("window.CONNECTOME_DATA = " + json.dumps(viz) + ";\n")

    print(f"Exported visualization data to {out_json} and {out_js} (M={M}, N={N}, {len(match_pairs)} matches)")
    return viz


# ============================================================
# 15. MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WCS-GNCCP connectome matching on real data (Stage 1: 2D by default)")
    parser.add_argument("--fly_A_file", type=str, default="flyA_hemibrain_MB.csv",
                         help="Real Fly A CSV (source,target,weight,region). "
                              "Use the SAME file as gnccp_pure_2d.py's --fly_A_file.")
    parser.add_argument("--fly_B_file", type=str, default="flyB_fafb_MB.csv",
                         help="Real Fly B CSV (source,target,weight,region). "
                              "Use the SAME file as gnccp_pure_2d.py's --fly_B_file.")
    parser.add_argument("--region", type=str, default="mushroom_body")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.5,
                         help="2D similarity-kernel bandwidth (Proposition 3).")
    parser.add_argument("--dzeta", type=float, default=0.05)
    parser.add_argument("--inner_iters_fw", type=int, default=30,
                         help="Max Frank-Wolfe iterations per zeta step. Defaults to "
                              "30 to match gnccp_pure_2d.py's --inner_iters for a fair "
                              "compute-budget comparison.")
    parser.add_argument("--max_neurons", type=int, default=300,
                         help="Cap on neurons per fly for feasibility. Use the SAME "
                              "value as gnccp_pure_2d.py's --max_neurons.")
    parser.add_argument("--seed", type=int, default=0,
                         help="Subsampling RNG seed. Use the SAME value as "
                              "gnccp_pure_2d.py's --seed to guarantee both scripts "
                              "operate on the identical subsampled neuron set.")
    parser.add_argument("--fetch", action="store_true",
                         help="Fetch fresh CSVs from neuPrint first (requires "
                              "NEUPRINT_TOKEN env var and the neuprint-python package). "
                              "Off by default -- normally you'll already have the CSVs.")
    parser.add_argument("--run_stage2", action="store_true",
                         help="Also run the optional 3D Kabsch refinement stage. Off by "
                              "default so this script's output stays comparable to "
                              "gnccp_pure_2d.py, which is 2D-only.")
    parser.add_argument("--coords_A_file", type=str, default=None,
                         help="Fly A soma-coordinate CSV, only used if --run_stage2 is set.")
    parser.add_argument("--coords_B_file", type=str, default=None,
                         help="Fly B soma-coordinate CSV, only used if --run_stage2 is set.")
    parser.add_argument("--gamma_3d", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()

    if args.fetch:
        fetch_real_datasets(fetch_coords=args.run_stage2)

    run_real_experiment(
        fly_A_file=args.fly_A_file,
        fly_B_file=args.fly_B_file,
        region=args.region,
        alpha=args.alpha,
        gamma=args.gamma,
        dzeta=args.dzeta,
        inner_iters_fw=args.inner_iters_fw,
        max_neurons=args.max_neurons,
        seed=args.seed,
        run_stage2=args.run_stage2,
        coords_A_file=args.coords_A_file,
        coords_B_file=args.coords_B_file,
        gamma_3d=args.gamma_3d,
        beta=args.beta,
    )