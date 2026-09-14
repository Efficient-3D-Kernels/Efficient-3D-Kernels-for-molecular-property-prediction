import os
import sys
import time
import numpy as np
import pandas as pd
import networkx as nx

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data_loader import load_college_msg
from src.graph_builder import build_real_social_graphs
from src.graph_features import build_feature_cost_matrix
from src.enhanced_features import build_enhanced_cost_matrix

from src.proposed_gnccp import run_proposed_gnccp
from src.proposed_gnccp_enhanced import run_enhanced_gnccp


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

USER_SIZES = [
    40,
    100,
    200,
    400
]

ALPHA_VALUES = [
    0.25,
    0.50,
    0.75
]

PERTURBATIONS = [
    0,
    10,
    20,
    30
]

RANDOM_SEED = 42

ZETA_STEP = 0.10

FW_ITERATIONS = 50

GAP_TOLERANCE = 1e-6

GAMMA = 1.0


# ============================================================
# MATCHING SIZE
# ============================================================

def get_matching_size(num_users):
    """
    Match approximately 75% of the selected nodes.

    This keeps the partial matching proportion consistent
    across different graph sizes.
    """

    return max(
        1,
        int(round(0.75 * num_users))
    )


# ============================================================
# GRAPH PERTURBATION
# ============================================================

def perturb_graph(G, perturbation_percent, seed=42):

    rng = np.random.default_rng(seed)

    H = G.copy()

    edges = list(H.edges())

    if perturbation_percent == 0:
        return H

    if len(edges) == 0:
        return H

    # --------------------------------------------------------
    # Select edges
    # --------------------------------------------------------

    num_perturb = max(
        1,
        int(
            len(edges)
            * perturbation_percent
            / 100.0
        )
    )

    num_perturb = min(
        num_perturb,
        len(edges)
    )

    selected_indices = rng.choice(
        len(edges),
        size=num_perturb,
        replace=False
    )

    selected_edges = [
        edges[i]
        for i in selected_indices
    ]

    # --------------------------------------------------------
    # Reduce edge weights
    # --------------------------------------------------------

    for u, v in selected_edges:

        old_weight = H[u][v].get(
            "weight",
            1.0
        )

        reduction = rng.uniform(
            0.10,
            0.50
        )

        H[u][v]["weight"] = max(
            old_weight * (1.0 - reduction),
            1e-8
        )

    # --------------------------------------------------------
    # Remove 25% of selected edges
    # --------------------------------------------------------

    num_remove = int(
        len(selected_edges) * 0.25
    )

    if num_remove > 0:

        remove_indices = rng.choice(
            len(selected_edges),
            size=num_remove,
            replace=False
        )

        for idx in remove_indices:

            u, v = selected_edges[idx]

            if H.has_edge(u, v):
                H.remove_edge(u, v)

    return H


# ============================================================
# MATCHING ACCURACY
# ============================================================

def calculate_matching_accuracy(
    X,
    node_order,
    ground_truth_order
):

    if X is None:
        return np.nan

    X = np.asarray(X)

    if X.ndim != 2:
        return np.nan

    selected_pairs = np.argwhere(
        X > 0.5
    )

    if len(selected_pairs) == 0:
        return np.nan

    correct = 0

    for i, j in selected_pairs:

        if i >= len(node_order):
            continue

        if j >= len(ground_truth_order):
            continue

        source_node = node_order[i]

        matched_node = ground_truth_order[j]

        if source_node == matched_node:
            correct += 1

    return (
        100.0
        * correct
        / len(selected_pairs)
    )


# ============================================================
# RUN V1
# ============================================================

def run_v1(
    AG,
    AH,
    C,
    matching_size,
    alpha
):

    start = time.perf_counter()

    result = run_proposed_gnccp(
        AG,
        AH,
        C,
        matching_size=matching_size,
        alpha=alpha,
        zeta_step=ZETA_STEP,
        fw_iterations=FW_ITERATIONS,
        gap_tolerance=GAP_TOLERANCE,
        gamma=GAMMA
    )

    runtime = time.perf_counter() - start

    if not isinstance(result, tuple):
        raise TypeError(
            "V1 did not return a tuple."
        )

    if len(result) != 4:
        raise ValueError(
            "Unexpected V1 return length."
        )

    X = result[0]

    objective = result[1]

    iterations = result[2]

    kernel = result[3]

    return (
        X,
        objective,
        iterations,
        kernel,
        runtime
    )


# ============================================================
# RUN V2
# ============================================================
def run_v2(
    G,
    H,
    AG,
    AH,
    C,
    matching_size,
    alpha
):
    """
    Run Enhanced GNCCP V2.

    run_enhanced_gnccp() returns a dictionary.
    The actual GNCCP result is stored under:

        result["gnccp_result"]

    The nested GNCCP result is:

        (
            X_discrete,
            final_objective,
            total_iterations,
            kernel
        )
    """

    start = time.perf_counter()

    result = run_enhanced_gnccp(
        AG,
        AH,
        G,
        H,
        matching_size=matching_size,
        alpha=alpha,
        zeta_step=ZETA_STEP,
        fw_iterations=FW_ITERATIONS,
        gap_tolerance=GAP_TOLERANCE,
        gamma=GAMMA
    )

    runtime = time.perf_counter() - start

    # --------------------------------------------------------
    # Validate outer result
    # --------------------------------------------------------

    if not isinstance(result, dict):

        raise TypeError(
            "V2 outer result is not a dictionary."
        )

    if "gnccp_result" not in result:

        raise KeyError(
            "V2 result does not contain "
            "'gnccp_result'."
        )

    # --------------------------------------------------------
    # Extract actual GNCCP result
    # --------------------------------------------------------

    gnccp_result = result["gnccp_result"]

    if not isinstance(
        gnccp_result,
        tuple
    ):

        raise TypeError(
            "V2 gnccp_result is not a tuple."
        )

    if len(gnccp_result) != 4:

        raise ValueError(
            f"Unexpected V2 GNCCP result length: "
            f"{len(gnccp_result)}"
        )

    # --------------------------------------------------------
    # Unpack
    # --------------------------------------------------------

    X = gnccp_result[0]

    objective = gnccp_result[1]

    iterations = gnccp_result[2]

    kernel = gnccp_result[3]

    # --------------------------------------------------------
    # Return standardized benchmark result
    # --------------------------------------------------------

    return (
        X,
        objective,
        iterations,
        kernel,
        runtime
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GNCCP SCALABILITY + ALPHA SENSITIVITY EXPERIMENT")
    print("=" * 80)

    print(
        "\nUser sizes:",
        USER_SIZES
    )

    print(
        "Alpha values:",
        ALPHA_VALUES
    )

    print(
        "Perturbations:",
        PERTURBATIONS
    )

    print(
        "Random seed:",
        RANDOM_SEED
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading CollegeMsg dataset...")

    interactions = load_college_msg()

    print(
        f"Interactions: "
        f"{len(interactions):,}"
    )

    # ========================================================
    # TEMPORAL SPLIT
    # ========================================================

    interactions = interactions.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    midpoint = len(interactions) // 2

    early_data = interactions.iloc[
        :midpoint
    ].copy()

    later_data = interactions.iloc[
        midpoint:
    ].copy()

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    # ========================================================
    # USER SIZE LOOP
    # ========================================================

    for num_users in USER_SIZES:

        print("\n")
        print("=" * 80)
        print(
            f"GRAPH SIZE: {num_users} USERS"
        )
        print("=" * 80)

        matching_size = get_matching_size(
            num_users
        )

        print(
            f"Matching size: "
            f"{matching_size}"
        )

        # ----------------------------------------------------
        # Build graphs
        # ----------------------------------------------------

        G, H_temporal, common_nodes = (
            build_real_social_graphs(
                early_data,
                later_data,
                num_users=num_users
            )
        )

        node_order = sorted(
            common_nodes
        )

        G = G.subgraph(
            node_order
        ).copy()

        H_temporal = H_temporal.subgraph(
            node_order
        ).copy()

        print(
            f"Actual G nodes: "
            f"{G.number_of_nodes()}"
        )

        print(
            f"Actual G edges: "
            f"{G.number_of_edges()}"
        )

        # ----------------------------------------------------
        # Base adjacency
        # ----------------------------------------------------

        AG = nx.to_numpy_array(
            G,
            nodelist=node_order,
            weight="weight",
            dtype=float
        )

        # ====================================================
        # PERTURBATION LOOP
        # ====================================================

        for perturbation in PERTURBATIONS:

            print("\n" + "-" * 80)

            print(
                f"Perturbation: "
                f"{perturbation}%"
            )

            # ------------------------------------------------
            # SAME perturbed graph for V1 and V2
            # ------------------------------------------------

            H = perturb_graph(
                G,
                perturbation,
                seed=RANDOM_SEED
            )

            H = H.subgraph(
                node_order
            ).copy()

            AH = nx.to_numpy_array(
                H,
                nodelist=node_order,
                weight="weight",
                dtype=float
            )

            # ------------------------------------------------
            # V1 cost
            # ------------------------------------------------

            C_v1, _, _ = (
                build_feature_cost_matrix(
                    G,
                    H
                )
            )

            # ------------------------------------------------
            # V2 cost
            # ------------------------------------------------

            C_v2 = (
                build_enhanced_cost_matrix(
                    G,
                    H
                )
            )

            # =================================================
            # ALPHA LOOP
            # =================================================

            for alpha in ALPHA_VALUES:

                print("\n" + "*" * 70)

                print(
                    f"Alpha = {alpha:.2f}"
                )

                # =================================================
                # V1
                # =================================================

                print(
                    "\nRunning V1 Proposed GNCCP..."
                )

                (
                    X1,
                    obj1,
                    iter1,
                    kernel1,
                    runtime1
                ) = run_v1(
                    AG,
                    AH,
                    C_v1,
                    matching_size,
                    alpha
                )

                acc1 = (
                    calculate_matching_accuracy(
                        X1,
                        node_order,
                        node_order
                    )
                )

                print(
                    f"\nV1 Accuracy: "
                    f"{acc1:.2f}%"
                )

                print(
                    f"V1 Objective: "
                    f"{obj1:.10f}"
                )

                print(
                    f"V1 Iterations: "
                    f"{iter1}"
                )

                print(
                    f"V1 Runtime: "
                    f"{runtime1:.4f}s"
                )

                print(
                    f"V1 Kernel: "
                    f"{kernel1:.6f}"
                )

                # =================================================
                # V2
                # =================================================

                print(
                    "\nRunning V2 Enhanced GNCCP..."
                )

                (
                    X2,
                    obj2,
                    iter2,
                    kernel2,
                    runtime2
                ) = run_v2(
                    G,
                    H,
                    AG,
                    AH,
                    C_v2,
                    matching_size,
                    alpha
                )

                acc2 = (
                    calculate_matching_accuracy(
                        X2,
                        node_order,
                        node_order
                    )
                )

                print(
                    f"\nV2 Accuracy: "
                    f"{acc2:.2f}%"
                )

                print(
                    f"V2 Objective: "
                    f"{obj2:.10f}"
                )

                print(
                    f"V2 Iterations: "
                    f"{iter2}"
                )

                print(
                    f"V2 Runtime: "
                    f"{runtime2:.4f}s"
                )

                print(
                    f"V2 Kernel: "
                    f"{kernel2:.6f}"
                )

                # =================================================
                # SAVE V1
                # =================================================

                results.append({
                    "Users": num_users,
                    "Matching Size": matching_size,
                    "Perturbation (%)": perturbation,
                    "Alpha": alpha,
                    "Method": "Proposed GNCCP",
                    "Accuracy (%)": acc1,
                    "Objective": obj1,
                    "Runtime (s)": runtime1,
                    "Iterations": iter1,
                    "Kernel": kernel1
                })

                # =================================================
                # SAVE V2
                # =================================================

                results.append({
                    "Users": num_users,
                    "Matching Size": matching_size,
                    "Perturbation (%)": perturbation,
                    "Alpha": alpha,
                    "Method": "Enhanced GNCCP",
                    "Accuracy (%)": acc2,
                    "Objective": obj2,
                    "Runtime (s)": runtime2,
                    "Iterations": iter2,
                    "Kernel": kernel2
                })

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print("\n\n")
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    results_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        "tables"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        results_dir,
        "gnccp_scalability_alpha_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nResults saved to:")
    print(output_file)

    print("\nBenchmark completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()