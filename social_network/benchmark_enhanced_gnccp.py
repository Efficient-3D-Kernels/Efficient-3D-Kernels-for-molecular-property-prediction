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

MATCHING_SIZE = 30

ALPHA = 0.50

ZETA_STEP = 0.10

FW_ITERATIONS = 50

GAP_TOLERANCE = 1e-6

GAMMA = 1.0

NUM_USERS = 40

RANDOM_SEED = 42

PERTURBATIONS = [0, 5, 10, 20, 30]


# ============================================================
# GRAPH PERTURBATION
# ============================================================

def perturb_graph(G, perturbation_percent, seed=42):
    """
    Create a controlled perturbed copy of G.

    The perturbation:
    1. randomly decreases selected edge weights
    2. removes a portion of selected edges

    Node identities are preserved.

    This allows matching accuracy to be evaluated against
    the original graph because the ground-truth correspondence
    is the identity mapping.
    """

    rng = np.random.default_rng(seed)

    H = G.copy()

    edges = list(H.edges())

    if len(edges) == 0:
        return H

    # --------------------------------------------------------
    # Number of edges affected
    # --------------------------------------------------------

    num_perturb = max(
        1,
        int(len(edges) * perturbation_percent / 100.0)
    ) if perturbation_percent > 0 else 0

    if num_perturb == 0:
        return H

    selected_indices = rng.choice(
        len(edges),
        size=min(num_perturb, len(edges)),
        replace=False
    )

    selected_edges = [
        edges[i] for i in selected_indices
    ]

    # --------------------------------------------------------
    # Reduce weights
    # --------------------------------------------------------

    for u, v in selected_edges:

        old_weight = H[u][v].get("weight", 1.0)

        reduction = rng.uniform(
            0.10,
            0.50
        )

        new_weight = old_weight * (1.0 - reduction)

        H[u][v]["weight"] = max(
            new_weight,
            1e-8
        )

    # --------------------------------------------------------
    # Remove approximately 25% of selected edges
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
    """
    Evaluate partial matching accuracy.

    Since matching_size = 30 and the graph has 40 nodes,
    only the 30 selected assignments are evaluated.

    The ground truth is identity correspondence because
    the perturbed graph preserves node identities.
    """

    if X is None:
        return np.nan

    X = np.asarray(X)

    if X.ndim != 2:
        return np.nan

    # --------------------------------------------------------
    # Find selected assignments
    # --------------------------------------------------------

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

        # Identity ground truth
        if source_node == matched_node:
            correct += 1

    return (
        100.0 * correct / len(selected_pairs)
    )


# ============================================================
# RUN ONE METHOD
# ============================================================

def run_method(
    method_name,
    AG,
    AH,
    C,
    matching_size
):

    start_time = time.perf_counter()

    if method_name == "Proposed GNCCP":

        result = run_proposed_gnccp(
            AG,
            AH,
            C,
            matching_size=matching_size,
            alpha=ALPHA,
            zeta_step=ZETA_STEP,
            fw_iterations=FW_ITERATIONS,
            gap_tolerance=GAP_TOLERANCE,
            gamma=GAMMA
        )

    elif method_name == "Enhanced GNCCP":

        result = run_enhanced_gnccp(
            AG,
            AH,
            matching_size=matching_size,
            alpha=ALPHA,
            zeta_step=ZETA_STEP,
            fw_iterations=FW_ITERATIONS,
            gap_tolerance=GAP_TOLERANCE,
            gamma=GAMMA
        )

    else:

        raise ValueError(
            f"Unknown method: {method_name}"
        )

    runtime = time.perf_counter() - start_time

    # ========================================================
    # IMPORTANT:
    # run_proposed_gnccp() RETURNS A TUPLE:
    #
    # (
    #     X_discrete,
    #     final_objective,
    #     total_iterations,
    #     kernel
    # )
    # ========================================================

    if not isinstance(result, tuple):

        raise TypeError(
            f"{method_name} returned "
            f"{type(result)}, expected tuple."
        )

    if len(result) != 4:

        raise ValueError(
            f"{method_name} returned tuple of length "
            f"{len(result)}, expected 4."
        )

    X = result[0]

    objective = result[1]

    iterations = result[2]

    kernel = result[3]

    return {
        "X": X,
        "objective": objective,
        "iterations": iterations,
        "kernel": kernel,
        "runtime": runtime
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 78)
    print("PROPOSED GNCCP V1 vs ENHANCED GNCCP V2")
    print("=" * 78)

    print(f"\nAlpha: {ALPHA}")
    print(f"Matching size: {MATCHING_SIZE}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Number of users: {NUM_USERS}")

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading CollegeMsg dataset...")

    interactions = load_college_msg()

    print("\nDataset loaded successfully.")

    print(
        f"Number of interactions: "
        f"{len(interactions):,}"
    )

    print(
    "Number of users: "
    "1,899"
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
    # BUILD REAL SOCIAL GRAPHS
    # ========================================================

    print("\nBuilding real social graphs...")

    G, H_temporal, common_nodes = (
        build_real_social_graphs(
            early_data,
            later_data,
            num_users=NUM_USERS
        )
    )

    node_order = sorted(common_nodes)

    G = G.subgraph(node_order).copy()

    H_temporal = H_temporal.subgraph(
        node_order
    ).copy()

    print(
        f"\nBase graph nodes: "
        f"{G.number_of_nodes()}"
    )

    print(
        f"Base graph edges: "
        f"{G.number_of_edges()}"
    )

    # ========================================================
    # BASE ADJACENCY MATRIX
    # ========================================================

    AG = nx.to_numpy_array(
        G,
        nodelist=node_order,
        weight="weight",
        dtype=float
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    # ========================================================
    # PERTURBATION EXPERIMENT
    # ========================================================

    for perturbation in PERTURBATIONS:

        print("\n" + "-" * 78)

        print(
            f"Perturbation: "
            f"{perturbation}%"
        )

        # ----------------------------------------------------
        # Create identical perturbed graph for both methods
        # ----------------------------------------------------

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

        # ====================================================
        # V1 FEATURE COST
        # ====================================================

        C_v1, _, _ = build_feature_cost_matrix(
            G,
            H
        )

        # ====================================================
        # V2 ENHANCED FEATURE COST
        # ====================================================

        C_v2 = build_enhanced_cost_matrix(
            G,
            H
        )

        # ====================================================
        # RUN V1
        # ====================================================

        print("\nRunning V1 Proposed GNCCP...")

        v1 = run_method(
            "Proposed GNCCP",
            AG,
            AH,
            C_v1,
            MATCHING_SIZE
        )

        v1_accuracy = calculate_matching_accuracy(
            v1["X"],
            node_order,
            node_order
        )

        # ====================================================
        # RUN V2
        # ====================================================

        print("\nRunning V2 Enhanced GNCCP...")

        v2 = run_method(
            "Enhanced GNCCP",
            AG,
            AH,
            C_v2,
            MATCHING_SIZE
        )

        v2_accuracy = calculate_matching_accuracy(
            v2["X"],
            node_order,
            node_order
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            f"\nV1 accuracy: "
            f"{v1_accuracy:.2f}%"
        )

        print(
            f"V2 accuracy: "
            f"{v2_accuracy:.2f}%"
        )

        print(
            f"V1 objective: "
            f"{v1['objective']:.10f}"
        )

        print(
            f"V2 objective: "
            f"{v2['objective']:.10f}"
        )

        print(
            f"V1 runtime: "
            f"{v1['runtime']:.4f}s"
        )

        print(
            f"V2 runtime: "
            f"{v2['runtime']:.4f}s"
        )

        print(
            f"V1 iterations: "
            f"{v1['iterations']}"
        )

        print(
            f"V2 iterations: "
            f"{v2['iterations']}"
        )

        print(
            f"V1 kernel: "
            f"{v1['kernel']:.6f}"
        )

        print(
            f"V2 kernel: "
            f"{v2['kernel']:.6f}"
        )

        # ====================================================
        # STORE V1
        # ====================================================

        results.append({
            "Method": "Proposed GNCCP",
            "Perturbation": perturbation,
            "Accuracy (%)": v1_accuracy,
            "Objective": v1["objective"],
            "Runtime (s)": v1["runtime"],
            "Iterations": v1["iterations"],
            "Kernel": v1["kernel"]
        })

        # ====================================================
        # STORE V2
        # ====================================================

        results.append({
            "Method": "Enhanced GNCCP",
            "Perturbation": perturbation,
            "Accuracy (%)": v2_accuracy,
            "Objective": v2["objective"],
            "Runtime (s)": v2["runtime"],
            "Iterations": v2["iterations"],
            "Kernel": v2["kernel"]
        })

    # ========================================================
    # FINAL TABLE
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 78)
    print("FINAL RESULTS")
    print("=" * 78)

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
        "enhanced_gnccp_collegemsg_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nSaved to:")
    print(output_file)

    print("\nBenchmark completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()