"""
GNCCP-based Weighted Common Subgraph Matching -- applied to a synthetic
SOCIAL NETWORK matching (de-anonymization) problem.

THE APPLICATION
----------------
This is the classic "social network de-anonymization" problem (Narayanan
& Shmatikov's line of work): you have a REFERENCE network G (e.g. a known
platform's friend graph, with real usernames) and a SAMPLE network H
(e.g. an anonymized export, or a different platform's graph) that:
  - shares most of the same real people with G (structural correspondence)
  - is MISSING some of them (a few users' accounts are private/deleted ->
    L < M: again the WCS/subgraph case, not plain equal-sized matching)
  - has extra unrelated accounts with no counterpart in G (outliers)
  - has noisy/different edges (friendships observed differ slightly
    between platforms/snapshots: some edges dropped, a few spurious ones
    added, interaction strengths jittered)
  - is stored under a COMPLETELY DIFFERENT, ARBITRARY node numbering
    (anonymized user IDs, database re-indexing, etc.)

WCS matching answers: "which anonymous account in H is which real user
from G?" -- despite the missing users, the extra accounts, the edge
noise, and having no idea how H's IDs relate to G's IDs.

NO GEOMETRY HERE (unlike the earlier point-cloud / MOF examples): edges
are friendship/interaction WEIGHTS, not distances. The paper's objective
doesn't care whether large numbers mean "close" or "strongly connected"
-- it only cares that corresponding edges have corresponding weights.

THE RELEVANT INVARIANCE
------------------------
There's no "rotation" for a social graph. The natural analogue is NODE
RELABELING: if you take the exact same network and renumber every user
with new arbitrary IDs (a pure bookkeeping change), the real-world answer
("which anonymous ID is which real person") must not change. We prove
this the same way as the geometric rotation test: permute H's adjacency
matrix rows/columns and show the algorithm finds the same real people
under the new numbering.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar

np.set_printoptions(precision=3, suppress=True, linewidth=150)


# ============================================================================
# 1. SYNTHETIC SOCIAL NETWORK GENERATOR (scale-free / preferential attachment
#    -- real social networks have a few high-degree "hub" users and many
#    low-degree ones, which is exactly what this produces)
# ============================================================================
def generate_social_network(n_nodes=25, m_edges=2, weight_range=(0.2, 1.0), seed=0):
    """
    Barabasi-Albert-style preferential-attachment graph: new users are more
    likely to befriend already-popular users, producing realistic hub
    structure. Edge weights represent interaction strength (e.g. messages/
    week, normalized).

    Returns: A (n_nodes x n_nodes weighted adjacency), labels (n_nodes,
    interest-category strings, used later as a chemistry-style unary
    feature analogous to metal/linker type).
    """
    rng = np.random.RandomState(seed)
    A = np.zeros((n_nodes, n_nodes))

    init = m_edges + 1
    for i in range(init):
        for j in range(i + 1, init):
            A[i, j] = A[j, i] = 1.0

    degree = A.sum(axis=1)
    for new in range(init, n_nodes):
        probs = degree[:new] / degree[:new].sum()
        targets = rng.choice(new, size=min(m_edges, new), replace=False, p=probs)
        for t in targets:
            A[new, t] = A[t, new] = 1.0
        degree[new] = len(targets)
        degree[targets] += 1

    # convert binary edges to weighted "interaction strength"
    W = A * rng.uniform(weight_range[0], weight_range[1], size=A.shape)
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)

    categories = ['Sports', 'Music', 'Tech', 'Art']
    labels = rng.choice(categories, size=n_nodes)
    return W, labels


# ============================================================================
# 2. BUILD A "SAMPLE" NETWORK H FROM REFERENCE G
# ============================================================================
def build_sample_from_reference(AG, G_labels, n_missing=4, n_outliers=8,
                                 edge_drop_prob=0.08, edge_add_prob=0.03,
                                 weight_noise_std=0.05, seed=1):
    """
    Builds H:
      - drops n_missing random G users (accounts private/deleted -> L < M)
      - keeps the rest, but with noisy edges: some real friendships not
        observed (edge_drop_prob), a few spurious ones added (edge_add_prob),
        interaction weights jittered (weight_noise_std)
      - adds n_outliers extra accounts with their own random connections
      - places everything under a NEW random index ordering (H's IDs bear
        no relationship to G's IDs -- exactly like a real anonymized export)
    """
    rng = np.random.RandomState(seed)
    M = AG.shape[0]

    keep_idx = np.sort(rng.choice(M, size=M - n_missing, replace=False))
    L = len(keep_idx)
    order = rng.permutation(L)   # random new position for each kept user

    N = L + n_outliers
    AH = np.zeros((N, N))
    H_labels = np.array([''] * N, dtype='<U6')
    ground_truth = {}

    # place the shared block with noisy edges
    sub = AG[np.ix_(keep_idx, keep_idx)].copy()
    drop_mask = rng.random(sub.shape) < edge_drop_prob
    add_mask = (rng.random(sub.shape) < edge_add_prob) & (sub == 0)
    sub[drop_mask] = 0.0
    sub[add_mask] = rng.uniform(0.2, 1.0, size=sub.shape)[add_mask]
    sub = (sub + sub.T) / 2.0
    sub += rng.normal(0, weight_noise_std, size=sub.shape)
    sub = np.clip(sub, 0.0, None)
    np.fill_diagonal(sub, 0.0)

    for a in range(L):
        for b in range(L):
            AH[order[a], order[b]] = sub[a, b]
    for k, g_idx in enumerate(keep_idx):
        H_labels[order[k]] = G_labels[g_idx]
        ground_truth[int(g_idx)] = int(order[k])

    # outlier accounts: random connections to everyone (shared block + other outliers)
    for o in range(n_outliers):
        idx = L + o
        n_friends = rng.randint(1, 4)
        friends = rng.choice(N, size=min(n_friends, N - 1), replace=False)
        friends = friends[friends != idx]
        for f in friends:
            w = rng.uniform(0.2, 1.0)
            AH[idx, f] = AH[f, idx] = w
        categories = ['Sports', 'Music', 'Tech', 'Art']
        H_labels[idx] = rng.choice(categories)

    return AH, ground_truth, H_labels, L


def relabel_network(AH, ground_truth, seed=2):
    """
    Applies a purely cosmetic relabeling: new arbitrary IDs for every node,
    representing that H's actual database indices carry no meaning at all.
    AH2[a,b] = AH[perm[a], perm[b]] -- same network, different numbering.
    """
    rng = np.random.RandomState(seed)
    N = AH.shape[0]
    perm = rng.permutation(N)          # new position a <- old content perm[a]
    inv_perm = np.argsort(perm)        # old index h_old -> new position

    AH2 = AH[np.ix_(perm, perm)]
    gt2 = {g: int(inv_perm[h_old]) for g, h_old in ground_truth.items()}
    return AH2, gt2, perm, inv_perm


def build_label_cost(G_labels, H_labels, mismatch_penalty=4.0):
    """C[i,j] = 0 if same interest category, else penalty. Same idea as the
    metal/linker cost matrix in the MOF example -- discourages matching a
    'Tech' user to a 'Sports' user even if their connection patterns look
    superficially similar."""
    M, N = len(G_labels), len(H_labels)
    C = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if G_labels[i] != H_labels[j]:
                C[i, j] = mismatch_penalty
    return C


# ============================================================================
# 3. H1(X) OBJECTIVE + GRADIENT  (paper Eq. 9, Eq. 11) -- identical to the
#    geometric/MOF versions: this part of the algorithm has no idea whether
#    A_G/A_H came from distances or from friendship strengths.
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
# 4. LINEAR STEP (Sec. II-D) + GNCCP MAIN LOOP (Algorithm 1)
# ============================================================================
def linear_step(grad, L):
    M, N = grad.shape
    row_ind, col_ind = linear_sum_assignment(grad)
    costs = grad[row_ind, col_ind]
    keep = np.argsort(costs)[:L]
    Y = np.zeros((M, N))
    Y[row_ind[keep], col_ind[keep]] = 1.0
    return Y


def gnccp_wcs_match(AG, AH, L, alpha=1.0, C=None, dzeta=0.05, inner_iters=15, tol=1e-6):
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
# 5. DISCRETE LOCAL-SEARCH REFINEMENT (post-processing hill-climb)
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


def match_and_refine(AG, AH, L, alpha=1.0, C=None, **gnccp_kwargs):
    M, N = AG.shape[0], AH.shape[0]
    X = gnccp_wcs_match(AG, AH, L, alpha=alpha, C=C, **gnccp_kwargs)
    assign = {i: int(np.argmax(row)) for i, row in enumerate(X) if row.sum() > 0.5}
    return local_search_refine(assign, AG, AH, M, N, alpha=alpha, C=C)


# ============================================================================
# 6. EVALUATION
# ============================================================================
def evaluate(assign, ground_truth, G_labels, H_labels):
    correct = sum(1 for g, h in ground_truth.items() if assign.get(g) == h)
    total = len(ground_truth)
    type_ok = sum(1 for g, h in assign.items() if G_labels[g] == H_labels[h])
    type_total = len(assign)
    return correct, total, type_ok, type_total


def print_breakdown(assign, ground_truth, G_labels, H_labels, limit=20):
    print(f"{'G user':<8} | {'category':<8} | {'True H':<8} | {'Pred H':<8} | Status")
    print("-" * 55)
    for i, (g, true_h) in enumerate(sorted(ground_truth.items())):
        if i >= limit:
            print(f"... ({len(ground_truth)-limit} more not shown)")
            break
        pred_h = assign.get(g, None)
        status = "OK" if pred_h == true_h else "MISS"
        print(f"{g:<8} | {G_labels[g]:<8} | {true_h:<8} | {str(pred_h):<8} | {status}")


# ============================================================================
# 7. DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("Building synthetic reference social network G")
    print("=" * 78)
    AG, G_labels = generate_social_network(n_nodes=22, m_edges=2, seed=0)
    M = AG.shape[0]
    print(f"Reference network: {M} users, {int((AG > 0).sum() / 2)} friendships")

    print("\n" + "=" * 78)
    print("A) RUN 1: sample export -- 4 users missing, 8 unrelated accounts added,")
    print("   edges partly dropped/added, weights jittered")
    print("=" * 78)
    AH1, gt1, Hl1, L1 = build_sample_from_reference(
        AG, G_labels, n_missing=4, n_outliers=8,
        edge_drop_prob=0.08, edge_add_prob=0.03, weight_noise_std=0.05, seed=7)
    N1 = AH1.shape[0]
    C1 = build_label_cost(G_labels, Hl1, mismatch_penalty=4.0)
    print(f"Sample network: {N1} accounts total ({L1} real correspondences + {N1-L1} decoys)")

    assign1 = match_and_refine(AG, AH1, L1, alpha=0.8, C=C1, dzeta=0.05, inner_iters=20)
    c, t, tok, ttot = evaluate(assign1, gt1, G_labels, Hl1)
    print_breakdown(assign1, gt1, G_labels, Hl1)
    print(f"\nDe-anonymization accuracy: {c}/{t} = {100*c/t:.1f}%")
    print(f"Interest-category consistency: {tok}/{ttot} = {100*tok/ttot:.1f}%")

    print("\n" + "=" * 78)
    print("B) RUN 2: THE SAME sample network, but every account given a brand-new")
    print("   arbitrary ID number (pure relabeling -- e.g. a different anonymization")
    print("   pass, or just re-exporting the same data with a different database)")
    print("=" * 78)
    AH2, gt2, perm, inv_perm = relabel_network(AH1, gt1, seed=99)
    Hl2 = Hl1[perm]     # labels move with the relabeling too
    C2 = build_label_cost(G_labels, Hl2, mismatch_penalty=4.0)
    print("AH2 is NOT numerically equal to AH1 (different index order) -- that's expected.")
    print("What must hold is: AH2 = P . AH1 . P^T for a permutation P (same network, new IDs).")
    diff_check = np.allclose(AH1, AH2[np.ix_(inv_perm, inv_perm)])
    print(f"Check AH2 un-relabeled back equals AH1: {diff_check}")

    assign2 = match_and_refine(AG, AH2, L1, alpha=0.8, C=C2, dzeta=0.05, inner_iters=20)
    c2, t2, tok2, ttot2 = evaluate(assign2, gt2, G_labels, Hl2)
    print_breakdown(assign2, gt2, G_labels, Hl2)
    print(f"\nDe-anonymization accuracy: {c2}/{t2} = {100*c2/t2:.1f}%")
    print(f"Interest-category consistency: {tok2}/{ttot2} = {100*tok2/ttot2:.1f}%")

    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    # Translate run-2 predictions back through the relabeling to compare in
    # "real identity" terms with run 1's predictions.
    assign2_mapped_back = {g: int(perm[h]) for g, h in assign2.items()}
    same_real_answer = assign1 == assign2_mapped_back
    print(f"Same real-world identification after undoing the relabeling: {same_real_answer}")
    print("Arbitrary user-ID numbering is bookkeeping, not information -- the")
    print("algorithm only ever looks at WHO IS CONNECTED TO WHOM (and how strongly),")
    print("never at the ID numbers themselves, so renumbering everyone changes")
    print("nothing about which real people it identifies.")