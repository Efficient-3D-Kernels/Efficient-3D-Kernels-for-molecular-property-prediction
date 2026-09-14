import os
import sys
import time
import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data_loader import load_college_msg
from src.graph_builder import build_real_social_graphs
from src.proposed_gnccp_enhanced import run_enhanced_gnccp


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

# Only the two requested graph sizes
USER_SIZES = [
    100,
    400
]

# Only the requested alpha
ALPHA_VALUES = [
    0.70
]

# Keep the perturbation experiment
PERTURBATIONS = [
    0,
    5,
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

    100 users -> 75 matches
    400 users -> 300 matches
    """

    return max(
        1,
        int(round(0.75 * num_users))
    )


# ============================================================
# GRAPH PERTURBATION
# ============================================================

def perturb_graph(
    G,
    perturbation_percent,
    seed=42
):
    """
    Create a perturbed copy of G.

    The node identities remain unchanged so that the
    identity mapping can be used as ground truth.

    Perturbation consists of:
        1. Reducing weights of selected edges.
        2. Removing 25% of those selected edges.
    """

    rng = np.random.default_rng(seed)

    H = G.copy()

    edges = list(
        H.edges()
    )

    if perturbation_percent == 0:
        return H

    if len(edges) == 0:
        return H

    # --------------------------------------------------------
    # Number of edges to perturb
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

    # --------------------------------------------------------
    # Randomly select edges
    # --------------------------------------------------------

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
    # Reduce selected edge weights
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

                H.remove_edge(
                    u,
                    v
                )

    return H


# ============================================================
# MATCHING ACCURACY
# ============================================================

def calculate_matching_accuracy(
    X,
    node_order,
    ground_truth_order
):
    """
    Calculate matching accuracy using node identity
    as the ground truth.

    Since perturbation changes only graph structure/weights
    and not node IDs, the correct mapping is:

        node i in G -> same node in H
    """

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
# RUN ENHANCED GNCCP
# ============================================================

def run_enhanced(
    G,
    H,
    AG,
    AH,
    matching_size,
    alpha
):
    """
    Run ONLY Enhanced GNCCP.

    The enhanced wrapper internally:
        - extracts the 8 enhanced features
        - builds the enhanced feature cost matrix
        - passes that cost matrix to the existing
          proposed GNCCP optimizer
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

    runtime = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # The enhanced wrapper returns an outer dictionary.
    # The actual GNCCP result is inside "gnccp_result".
    # --------------------------------------------------------

    if not isinstance(
        result,
        dict
    ):

        raise TypeError(
            "Enhanced GNCCP did not return a dictionary."
        )

    if "gnccp_result" not in result:

        raise KeyError(
            "Enhanced GNCCP result does not contain "
            "'gnccp_result'."
        )

    gnccp_result = result[
        "gnccp_result"
    ]

    if not isinstance(
        gnccp_result,
        tuple
    ):

        raise TypeError(
            "Enhanced GNCCP 'gnccp_result' "
            "is not a tuple."
        )

    if len(gnccp_result) != 4:

        raise ValueError(
            "Unexpected Enhanced GNCCP "
            "result length."
        )

    # --------------------------------------------------------
    # Existing GNCCP return:
    #
    # X
    # objective
    # iterations
    # kernel
    # --------------------------------------------------------

    X = gnccp_result[0]

    objective = gnccp_result[1]

    iterations = gnccp_result[2]

    kernel = gnccp_result[3]

    return (
        X,
        objective,
        iterations,
        kernel,
        runtime
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 80)
    print(
        "ENHANCED GNCCP — SCALABILITY + PERTURBATION EXPERIMENT"
    )
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

    print("=" * 80)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        "\nLoading CollegeMsg dataset..."
    )

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
    ).reset_index(
        drop=True
    )

    midpoint = len(interactions) // 2

    early_data = interactions.iloc[
        :midpoint
    ].copy()

    later_data = interactions.iloc[
        midpoint:
    ].copy()

    print(
        f"Early interactions: "
        f"{len(early_data):,}"
    )

    print(
        f"Later interactions: "
        f"{len(later_data):,}"
    )

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
            f"Matching size L: "
            f"{matching_size}"
        )

        # ----------------------------------------------------
        # Build common-user graphs
        # ----------------------------------------------------

        G, H_temporal, common_nodes = (
            build_real_social_graphs(
                early_data,
                later_data,
                num_users=num_users
            )
        )

        # ----------------------------------------------------
        # Fixed node ordering
        # ----------------------------------------------------

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

        print(
            f"Actual H nodes: "
            f"{H_temporal.number_of_nodes()}"
        )

        print(
            f"Actual H edges: "
            f"{H_temporal.number_of_edges()}"
        )

        # ----------------------------------------------------
        # Base adjacency matrix
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

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"Perturbation: "
                f"{perturbation}%"
            )

            # ------------------------------------------------
            # Perturb graph
            #
            # IMPORTANT:
            # Node identities are preserved.
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

            print(
                f"Perturbed H edges: "
                f"{H.number_of_edges()}"
            )

            # =================================================
            # ALPHA LOOP
            # =================================================

            for alpha in ALPHA_VALUES:

                print(
                    "\n"
                    + "*" * 70
                )

                print(
                    f"Alpha = {alpha:.2f}"
                )

                print(
                    "*" * 70
                )

                # =================================================
                # ENHANCED GNCCP
                # =================================================

                print(
                    "\nRunning Enhanced GNCCP..."
                )

                (
                    X,
                    objective,
                    iterations,
                    kernel,
                    runtime
                ) = run_enhanced(
                    G,
                    H,
                    AG,
                    AH,
                    matching_size,
                    alpha
                )

                # -------------------------------------------------
                # Accuracy
                # -------------------------------------------------

                accuracy = (
                    calculate_matching_accuracy(
                        X,
                        node_order,
                        node_order
                    )
                )

                actual_matches = int(
                    np.sum(X > 0.5)
                )

                # -------------------------------------------------
                # Print results
                # -------------------------------------------------

                print()

                print(
                    f"Enhanced Accuracy: "
                    f"{accuracy:.2f}%"
                )

                print(
                    f"Enhanced Objective: "
                    f"{objective:.10f}"
                )

                print(
                    f"Enhanced Iterations: "
                    f"{iterations}"
                )

                print(
                    f"Enhanced Runtime: "
                    f"{runtime:.4f}s"
                )

                print(
                    f"Enhanced Kernel: "
                    f"{kernel:.6f}"
                )

                print(
                    f"Actual Matches: "
                    f"{actual_matches}"
                )

                # -------------------------------------------------
                # Save result
                # -------------------------------------------------

                results.append({

                    "Users":
                        num_users,

                    "Matching Size":
                        matching_size,

                    "Perturbation (%)":
                        perturbation,

                    "Alpha":
                        alpha,

                    "Method":
                        "Enhanced GNCCP",

                    "Accuracy (%)":
                        accuracy,

                    "Objective":
                        objective,

                    "Runtime (s)":
                        runtime,

                    "Iterations":
                        iterations,

                    "Kernel":
                        kernel,

                    "Actual Matches":
                        actual_matches
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
    # SAVE RESULTS
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
        "enhanced_gnccp_100_400_alpha07_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        output_file
    )

    print(
        "\nExperiment completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()