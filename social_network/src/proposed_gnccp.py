import numpy as np
from scipy.optimize import linear_sum_assignment


# ============================================================
# PROPOSED WCS-GNCCP
#
# Proposed pipeline:
# 1. Normalized adjacency matrices
# 2. Normalized node-cost matrix
# 3. WCS objective
# 4. Rank-one structural gradient
# 5. GNCCP continuation
# 6. Frank-Wolfe linear oracle
# 7. Frank-Wolfe gap
# 8. Armijo line search
# 9. Sinkhorn-style projection
# 10. Hungarian discretization
# 11. WCS kernel
# ============================================================


def normalize_adjacency(A):
    """Frobenius normalization of adjacency matrix."""

    A = np.asarray(A, dtype=float)

    norm = np.linalg.norm(A, ord="fro")

    if norm == 0:
        return A.copy()

    return A / norm


def normalize_cost(C):
    """Min-max normalization of node-cost matrix."""

    C = np.asarray(C, dtype=float)

    c_min = C.min()
    c_max = C.max()

    if c_max == c_min:
        return np.zeros_like(C)

    return (C - c_min) / (c_max - c_min)


def compute_U(X):
    """
    Rank-one formulation:

        U = X 1 X^T

    Since:

        X 1 = r

    this is equivalent to:

        U = r r^T
    """

    row_mass = X @ np.ones(X.shape[1])

    return np.outer(
        row_mass,
        row_mass
    )


def compute_wcs_objective(
    X,
    AG,
    AH,
    C,
    alpha=0.5
):
    """Compute normalized WCS objective."""

    U = compute_U(X)

    mapped_AH = X @ AH @ X.T

    residual = (
        U * AG
        - mapped_AH
    )

    structural_term = np.sum(
        residual ** 2
    )

    node_cost = np.trace(
        C.T @ X
    )

    return (
        alpha * structural_term
        + (1.0 - alpha) * node_cost
    )


def compute_structural_gradient(
    X,
    AG,
    AH
):
    """
    Rank-one structural gradient used by
    the proposed implementation.

        R = U o AG - X AH X^T

        grad =
        4(R o AG)X1 - 4RXAH
    """

    M, N = X.shape

    ones = np.ones(
        (N, N),
        dtype=float
    )

    U = compute_U(X)

    mapped_AH = X @ AH @ X.T

    R = (
        U * AG
        - mapped_AH
    )

    gradient = (
        4.0 * (R * AG) @ X @ ones
        - 4.0 * R @ X @ AH
    )

    return gradient


def compute_wcs_gradient(
    X,
    AG,
    AH,
    C,
    alpha=0.5
):
    """Gradient of proposed WCS objective."""

    structural_gradient = (
        compute_structural_gradient(
            X,
            AG,
            AH
        )
    )

    return (
        alpha * structural_gradient
        + (1.0 - alpha) * C
    )


def compute_gnccp_objective(
    X,
    AG,
    AH,
    C,
    alpha,
    zeta
):
    """
    GNCCP continuation objective.

    For zeta >= 0:
        J = (1-zeta)F + zeta||X||²

    For zeta < 0:
        J = (1+zeta)F + zeta||X||²
    """

    F = compute_wcs_objective(
        X,
        AG,
        AH,
        C,
        alpha
    )

    norm_squared = np.sum(X ** 2)

    if zeta >= 0:

        return (
            (1.0 - zeta) * F
            + zeta * norm_squared
        )

    return (
        (1.0 + zeta) * F
        + zeta * norm_squared
    )


def compute_gnccp_gradient(
    X,
    AG,
    AH,
    C,
    alpha,
    zeta
):
    """Gradient of GNCCP continuation objective."""

    grad_F = compute_wcs_gradient(
        X,
        AG,
        AH,
        C,
        alpha
    )

    if zeta >= 0:

        return (
            (1.0 - zeta) * grad_F
            + 2.0 * zeta * X
        )

    return (
        (1.0 + zeta) * grad_F
        + 2.0 * zeta * X
    )


def initialize_X(M, N, L):
    """Uniform relaxed initialization."""

    X = np.ones(
        (M, N),
        dtype=float
    )

    return X * L / (M * N)


def linear_oracle(
    gradient,
    L
):
    """
    Hungarian linear minimization oracle.

    Select L lowest-cost one-to-one assignments.
    """

    rows, cols = linear_sum_assignment(
        gradient
    )

    candidates = [
        (
            r,
            c,
            gradient[r, c]
        )
        for r, c in zip(rows, cols)
    ]

    candidates.sort(
        key=lambda x: x[2]
    )

    candidates = candidates[:L]

    Y = np.zeros_like(
        gradient
    )

    for r, c, _ in candidates:
        Y[r, c] = 1.0

    return Y


def sinkhorn_project(
    X,
    L,
    iterations=50
):
    """
    Sinkhorn-style projection toward the
    relaxed partial-permutation domain.

    Constraints:

        X >= 0
        row sums <= 1
        column sums <= 1
        total mass = L
    """

    X = np.maximum(
        X,
        0.0
    )

    if X.sum() == 0:
        return initialize_X(
            X.shape[0],
            X.shape[1],
            L
        )

    # First enforce total mass
    X *= L / X.sum()

    for _ in range(iterations):

        # Row normalization
        row_sums = X.sum(axis=1)

        for i in range(X.shape[0]):

            if row_sums[i] > 1.0:

                X[i, :] /= row_sums[i]

        # Column normalization
        col_sums = X.sum(axis=0)

        for j in range(X.shape[1]):

            if col_sums[j] > 1.0:

                X[:, j] /= col_sums[j]

        # Restore total mass
        total = X.sum()

        if total > 0:

            X *= L / total

    return X


def frank_wolfe_gap(
    X,
    Y,
    gradient
):
    """
    Frank-Wolfe gap:

        gap = <gradient, X-Y>

    A small gap indicates convergence.
    """

    gap = np.sum(
        gradient * (X - Y)
    )

    return max(
        0.0,
        float(gap)
    )


def armijo_line_search(
    X,
    Y,
    gradient,
    AG,
    AH,
    C,
    alpha,
    zeta
):
    """
    Armijo backtracking line search.
    """

    direction = Y - X

    current_value = compute_gnccp_objective(
        X,
        AG,
        AH,
        C,
        alpha,
        zeta
    )

    directional_derivative = np.sum(
        gradient * direction
    )

    step = 1.0

    beta = 0.5
    sigma = 1e-4

    while step >= 1e-8:

        candidate = X + step * direction

        candidate_value = compute_gnccp_objective(
            candidate,
            AG,
            AH,
            C,
            alpha,
            zeta
        )

        if candidate_value <= (
            current_value
            + sigma
            * step
            * directional_derivative
        ):

            return step

        step *= beta

    return 0.0


def optimize_at_zeta(
    X,
    AG,
    AH,
    C,
    alpha,
    zeta,
    L,
    max_iterations,
    gap_tolerance
):
    """Optimize the proposed objective at one zeta."""

    for iteration in range(max_iterations):

        gradient = compute_gnccp_gradient(
            X,
            AG,
            AH,
            C,
            alpha,
            zeta
        )

        Y = linear_oracle(
            gradient,
            L
        )

        gap = frank_wolfe_gap(
            X,
            Y,
            gradient
        )

        if gap < gap_tolerance:
            break

        step = armijo_line_search(
            X,
            Y,
            gradient,
            AG,
            AH,
            C,
            alpha,
            zeta
        )

        if step <= 0:
            break

        X = X + step * (Y - X)

        # Proposed Sinkhorn-style projection
        X = sinkhorn_project(
            X,
            L
        )

    return X, gap, iteration + 1


def discretize_matching(
    X,
    L
):
    """
    Convert relaxed solution to a discrete
    partial permutation matrix.
    """

    rows, cols = linear_sum_assignment(
        -X
    )

    assignments = [
        (
            r,
            c,
            X[r, c]
        )
        for r, c in zip(rows, cols)
    ]

    assignments.sort(
        key=lambda x: x[2],
        reverse=True
    )

    assignments = assignments[:L]

    X_discrete = np.zeros_like(X)

    for r, c, _ in assignments:
        X_discrete[r, c] = 1.0

    return X_discrete


def compute_kernel(
    objective,
    L,
    gamma=1.0
):
    """
    Proposed WCS graph similarity kernel:

        k(G,H) = exp(-gamma F*/L)
    """

    if L <= 0:
        return 0.0

    return np.exp(
        -gamma * objective / L
    )


def run_proposed_gnccp(
    AG,
    AH,
    C,
    matching_size,
    alpha=0.5,
    zeta_step=0.1,
    fw_iterations=50,
    gap_tolerance=1e-6,
    gamma=1.0
):
    """
    Complete proposed WCS-GNCCP algorithm.

    Normalization is performed inside this function
    so that the proposed method explicitly owns the
    normalization stage.
    """

    # --------------------------------------------------------
    # PROPOSED NORMALIZATION
    # --------------------------------------------------------

    AG = normalize_adjacency(
        AG
    )

    AH = normalize_adjacency(
        AH
    )

    C = normalize_cost(
        C
    )

    M, N = C.shape

    L = matching_size

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    X = initialize_X(
        M,
        N,
        L
    )

    zeta = 1.0

    total_iterations = 0

    print("\nRunning PROPOSED WCS-GNCCP")
    print("=" * 55)

    # --------------------------------------------------------
    # GNCCP CONTINUATION
    # --------------------------------------------------------

    while zeta >= -1.0 - 1e-9:

        X, gap, iterations = optimize_at_zeta(
            X,
            AG,
            AH,
            C,
            alpha,
            zeta,
            L,
            fw_iterations,
            gap_tolerance
        )

        total_iterations += iterations

        objective = compute_gnccp_objective(
            X,
            AG,
            AH,
            C,
            alpha,
            zeta
        )

        print(
            f"zeta = {zeta:5.2f} | "
            f"J_zeta = {objective:.6f} | "
            f"FW gap = {gap:.8f}"
        )

        zeta -= zeta_step

    # --------------------------------------------------------
    # DISCRETIZATION
    # --------------------------------------------------------

    X_discrete = discretize_matching(
        X,
        L
    )

    # --------------------------------------------------------
    # FINAL WCS OBJECTIVE
    # --------------------------------------------------------

    final_objective = compute_wcs_objective(
        X_discrete,
        AG,
        AH,
        C,
        alpha
    )

    # --------------------------------------------------------
    # KERNEL
    # --------------------------------------------------------

    kernel = compute_kernel(
        final_objective,
        L,
        gamma
    )

    return (
        X_discrete,
        final_objective,
        total_iterations,
        kernel
    )