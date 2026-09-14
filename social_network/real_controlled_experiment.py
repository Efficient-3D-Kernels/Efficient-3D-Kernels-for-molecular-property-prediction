import time
import numpy as np
import pandas as pd

from src.data_loader import (
    load_college_msg,
    split_temporal_data
)

from src.graph_builder import (
    build_real_social_graphs
)

from src.graph_features import (
    build_feature_cost_matrix
)

from src.baseline_gnccp import (
    run_baseline_gnccp
)

from src.proposed_gnccp import (
    run_proposed_gnccp
)


# ============================================================
# REAL SOCIAL NETWORK CONTROLLED EXPERIMENT
#
# We use a REAL CollegeMsg social graph.
#
# Graph G = real early-period graph
#
# Graph H = controlled perturbation of G
#
# The node identities remain known, giving us a reliable
# ground truth for evaluating graph matching.
# ============================================================


def graph_to_adjacency_matrix(G, nodes):
    """Convert weighted NetworkX graph to adjacency matrix."""

    n = len(nodes)

    A = np.zeros(
        (n, n),
        dtype=float
    )

    index = {
        node: i
        for i, node in enumerate(nodes)
    }

    for u, v, data in G.edges(data=True):

        i = index[u]
        j = index[v]

        weight = data.get(
            "weight",
            1.0
        )

        A[i, j] = weight
        A[j, i] = weight

    return A


def normalize_adjacency(A):
    """Frobenius normalization."""

    norm = np.linalg.norm(
        A,
        ord="fro"
    )

    if norm == 0:
        return A.copy()

    return A / norm


def calculate_accuracy(
    X,
    G_nodes,
    H_nodes
):
    """
    Ground truth:

        user in G -> same user in H

    Calculate percentage of correctly recovered
    correspondences.
    """

    correct = 0
    total = int(X.sum())

    for i in range(X.shape[0]):

        for j in range(X.shape[1]):

            if X[i, j] == 1:

                if G_nodes[i] == H_nodes[j]:

                    correct += 1

    if total == 0:
        return 0.0

    return (
        100.0
        * correct
        / total
    )


def perturb_graph(
    G,
    perturbation_level,
    seed=42
):
    """
    Create a controlled perturbed version of a REAL graph.

    The nodes remain the same.

    Edge weights are modified according to the
    perturbation level.

    This keeps the experiment real-data based while
    controlling the difficulty of graph matching.
    """

    rng = np.random.default_rng(seed)

    H = G.copy()

    # --------------------------------------------------------
    # Weight perturbation
    # --------------------------------------------------------

    for u, v in H.edges():

        original_weight = H[u][v]["weight"]

        noise = rng.normal(
            0.0,
            perturbation_level
            * max(original_weight, 1.0)
        )

        H[u][v]["weight"] = max(
            0.01,
            original_weight + noise
        )

    # --------------------------------------------------------
    # Edge deletion
    # --------------------------------------------------------

    if perturbation_level > 0:

        edges = list(H.edges())

        deletion_probability = min(
            perturbation_level * 0.25,
            0.20
        )

        for u, v in edges:

            if rng.random() < deletion_probability:

                H.remove_edge(
                    u,
                    v
                )

    return H


def run_single_experiment(
    G,
    H,
    C,
    nodes,
    perturbation,
    matching_size=30
):
    """
    Run baseline and proposed methods on one
    real controlled graph pair.
    """

    AG = graph_to_adjacency_matrix(
        G,
        nodes
    )

    AH = graph_to_adjacency_matrix(
        H,
        nodes
    )

    AG = normalize_adjacency(
        AG
    )

    AH = normalize_adjacency(
        AH
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    start = time.perf_counter()

    X_baseline, obj_baseline, iterations_baseline = (
        run_baseline_gnccp(
            AG,
            AH,
            C,
            matching_size,
            alpha=0.5,
            zeta_step=0.1,
            fw_iterations=30
        )
    )

    baseline_time = (
        time.perf_counter()
        - start
    )

    baseline_accuracy = calculate_accuracy(
        X_baseline,
        nodes,
        nodes
    )

    # --------------------------------------------------------
    # PROPOSED
    # --------------------------------------------------------

    start = time.perf_counter()

    (
        X_proposed,
        obj_proposed,
        iterations_proposed,
        proposed_kernel
    ) = run_proposed_gnccp(
        AG,
        AH,
        C,
        matching_size,
        alpha=0.5,
        zeta_step=0.1,
        fw_iterations=30
    )

    proposed_time = (
        time.perf_counter()
        - start
    )

    proposed_accuracy = calculate_accuracy(
        X_proposed,
        nodes,
        nodes
    )

    return {
        "Perturbation": perturbation,

        "Baseline Accuracy (%)":
            baseline_accuracy,

        "Proposed Accuracy (%)":
            proposed_accuracy,

        "Baseline Objective":
            obj_baseline,

        "Proposed Objective":
            obj_proposed,

        "Baseline Runtime":
            baseline_time,

        "Proposed Runtime":
            proposed_time,

        "Baseline Iterations":
            iterations_baseline,

        "Proposed Iterations":
            iterations_proposed,

        "Proposed Kernel":
            proposed_kernel
    }


def main():

    print("\n")
    print("=" * 75)
    print("REAL COLLEGEMSG CONTROLLED GNCCP EXPERIMENT")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. LOAD REAL DATA
    # --------------------------------------------------------

    data = load_college_msg()

    early_data, later_data = (
        split_temporal_data(data)
    )

    # --------------------------------------------------------
    # 2. BUILD REAL SOCIAL GRAPH
    # --------------------------------------------------------

    G, _, selected_users = (
        build_real_social_graphs(
            early_data,
            later_data,
            num_users=40
        )
    )

    nodes = sorted(
        selected_users
    )

    # --------------------------------------------------------
    # 3. BUILD REAL SOCIAL FEATURES
    # --------------------------------------------------------

    # For this controlled experiment the feature cost
    # is calculated from G and a copy of G initially.
    C, _, _ = build_feature_cost_matrix(
        G,
        G.copy()
    )

    # --------------------------------------------------------
    # 4. EXPERIMENT LEVELS
    # --------------------------------------------------------

    perturbation_levels = [
        0.00,
        0.05,
        0.10,
        0.20,
        0.30
    ]

    results = []

    # --------------------------------------------------------
    # 5. RUN EXPERIMENTS
    # --------------------------------------------------------

    for level in perturbation_levels:

        print("\n")
        print("=" * 75)
        print(
            f"PERTURBATION LEVEL = {level:.2f}"
        )
        print("=" * 75)

        H = perturb_graph(
            G,
            perturbation_level=level,
            seed=100 + int(level * 1000)
        )

        result = run_single_experiment(
            G,
            H,
            C,
            nodes,
            level,
            matching_size=30
        )

        results.append(
            result
        )

        print("\nResult:")
        print(
            f"Baseline Accuracy : "
            f"{result['Baseline Accuracy (%)']:.2f}%"
        )

        print(
            f"Proposed Accuracy : "
            f"{result['Proposed Accuracy (%)']:.2f}%"
        )

        print(
            f"Baseline Objective: "
            f"{result['Baseline Objective']:.6f}"
        )

        print(
            f"Proposed Objective: "
            f"{result['Proposed Objective']:.6f}"
        )

    # --------------------------------------------------------
    # 6. RESULTS TABLE
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 100)
    print("FINAL REAL-DATA RESULTS")
    print("=" * 100)

    print(
        df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 7. SAVE RESULTS
    # --------------------------------------------------------

    output_path = (
        "results/tables/"
        "real_controlled_results.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nResults saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()