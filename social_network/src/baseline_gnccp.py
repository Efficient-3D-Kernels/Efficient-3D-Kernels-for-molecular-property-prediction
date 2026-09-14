import numpy as np
from scipy.optimize import linear_sum_assignment


# ============================================================
# ORIGINAL WCS-GNCCP BASELINE
# Based on the H1 relaxation and GNCCP framework
# from Yang, Qiao & Liu, IEEE TNNLS 2018.
# ============================================================


def compute_U(X):
    """
    U(X) = X 1 X^T
    """

    N = X.shape[1]

    return X @ np.ones((N, N)) @ X.T


def compute_H1(X, AG, AH):
    """
    Original paper's H1 structural relaxation:

        H1(X) =
        || U o AG - X AH X^T ||_F^2

    where:
        U = X 1 X^T
    """

    U = compute_U(X)

    residual = (
        U * AG
        - X @ AH @ X.T
    )

    return np.sum(residual ** 2)


def compute_F(X, AG, AH, C, alpha):
    """
    Original WCS objective:

        F(X) =
        alpha H1(X)
        + (1-alpha) tr(C^T X)
    """

    H1 = compute_H1(
        X,
        AG,
        AH
    )

    node_cost = np.trace(
        C.T @ X
    )

    return (
        alpha * H1
        + (1.0 - alpha) * node_cost
    )


def compute_H1_gradient(X, AG, AH):
    """
    Gradient of H1.

    R = U o AG - X AH X^T

    grad H1 =
        4 (R o AG) X 1
        - 4 R X AH
    """

    M, N = X.shape

    ones = np.ones((N, N))

    U = compute_U(X)

    mapped_AH = (
        X @ AH @ X.T
    )

    R = (
        U * AG
        - mapped_AH
    )

    gradient = (
        4.0 * (R * AG) @ X @ ones
        - 4.0 * R @ X @ AH
    )

    return gradient


def compute_F_gradient(
    X,
    AG,
    AH,
    C,
    alpha
):
    """
    Gradient of original WCS objective.
    """

    return (
        alpha
        * compute_H1_gradient(
            X,
            AG,
            AH
        )
        + (1.0 - alpha) * C
    )


def compute_J_zeta(
    X,
    AG,
    AH,
    C,
    alpha,
    zeta
):
    """
    GNCCP continuation objective.

    zeta >= 0:
        J = (1-zeta)F + zeta ||X||^2

    zeta < 0:
        J = (1+zeta)F + zeta ||X||^2
    """

    F = compute_F(
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


def compute_J_gradient(
    X,
    AG,
    AH,
    C,
    alpha,
    zeta
):
    """
    Gradient of J_zeta.
    """

    grad_F = compute_F_gradient(
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
    """
    Initialization used for the relaxed
    cardinality-L matching domain.
    """

    return np.ones(
        (M, N),
        dtype=float
    ) * L / (M * N)


def linear_assignment_oracle(
    gradient,
    L
):
    """
    Linear minimization oracle.

    Solve:

        min <gradient, Y>

    over one-to-one assignments, then retain
    the L lowest-cost assignments.
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
        key=lambda item: item[2]
    )

    candidates = candidates[:L]

    Y = np.zeros_like(
        gradient
    )

    for r, c, _ in candidates:
        Y[r, c] = 1.0

    return Y


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

    current = compute_J_zeta(
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

    while step > 1e-8:

        candidate = X + step * direction

        candidate_value = compute_J_zeta(
            candidate,
            AG,
            AH,
            C,
            alpha,
            zeta
        )

        if candidate_value <= (
            current
            + sigma
            * step
            * directional_derivative
        ):
            return step

        step *= beta

    return 0.0


def frank_wolfe(
    X,
    AG,
    AH,
    C,
    alpha,
    zeta,
    L,
    max_iterations=50,
    tolerance=1e-6
):
    """
    Frank-Wolfe optimization for fixed zeta.
    """

    previous_value = compute_J_zeta(
        X,
        AG,
        AH,
        C,
        alpha,
        zeta
    )

    for _ in range(max_iterations):

        gradient = compute_J_gradient(
            X,
            AG,
            AH,
            C,
            alpha,
            zeta
        )

        Y = linear_assignment_oracle(
            gradient,
            L
        )

        direction = Y - X

        # Frank-Wolfe gap
        gap = -np.sum(
            gradient * direction
        )

        if gap < tolerance:
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

        X = X + step * direction

        current_value = compute_J_zeta(
            X,
            AG,
            AH,
            C,
            alpha,
            zeta
        )

        if abs(
            current_value
            - previous_value
        ) < tolerance:
            break

        previous_value = current_value

    return X


def discretize_matching(X, L):
    """
    Convert relaxed X to a discrete partial
    permutation matrix.
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
        key=lambda item: item[2],
        reverse=True
    )

    assignments = assignments[:L]

    X_discrete = np.zeros_like(X)

    for r, c, _ in assignments:
        X_discrete[r, c] = 1.0

    return X_discrete


def run_baseline_gnccp(
    AG,
    AH,
    C,
    matching_size,
    alpha=0.5,
    zeta_step=0.1,
    fw_iterations=50,
    tolerance=1e-6
):
    """
    Complete baseline GNCCP procedure.

    Continuation:
        zeta = 1 -> -1

    At each zeta:
        1. calculate gradient
        2. Hungarian linear oracle
        3. Frank-Wolfe direction
        4. Armijo line search
        5. update X
    """

    M, N = C.shape
    L = matching_size

    X = initialize_X(
        M,
        N,
        L
    )

    zeta = 1.0

    total_iterations = 0

    print("\nRunning ORIGINAL WCS-GNCCP Baseline")
    print("=" * 55)

    while zeta >= -1.0 - 1e-9:

        X = frank_wolfe(
            X,
            AG,
            AH,
            C,
            alpha,
            zeta,
            L,
            max_iterations=fw_iterations,
            tolerance=tolerance
        )

        total_iterations += fw_iterations

        value = compute_J_zeta(
            X,
            AG,
            AH,
            C,
            alpha,
            zeta
        )

        print(
            f"zeta = {zeta:5.2f} | "
            f"J_zeta = {value:.6f}"
        )

        zeta -= zeta_step

    X_discrete = discretize_matching(
        X,
        L
    )

    final_objective = compute_F(
        X_discrete,
        AG,
        AH,
        C,
        alpha
    )

    return (
        X_discrete,
        final_objective,
        total_iterations
    )