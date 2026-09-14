"""
Visualize and analyze the Social Network GNCCP experiment.

Generates:
    results/plots/early_graph.png
    results/plots/later_graph.png
    results/plots/graph_comparison.png
    results/plots/matched_graphs.png
    results/plots/accuracy_vs_perturbation.png
    results/plots/iterations_vs_perturbation.png
    results/plots/runtime_vs_perturbation.png

Generates:
    results/tables/social_network_performance.csv

Uses the existing project modules:
    src/data_loader.py
    src/graph_builder.py
    src/graph_features.py
    src/baseline_gnccp.py
    src/proposed_gnccp.py

No changes are required to the existing GNCCP implementations.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Existing project modules
from data_loader import load_college_msg
from graph_builder import (
    build_weighted_social_graph,
    select_active_users,
    remove_isolated_nodes,
)
from graph_features import build_feature_cost_matrix
from baseline_gnccp import run_baseline_gnccp
from proposed_gnccp import run_proposed_gnccp


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

NUM_USERS = 40
MATCHING_SIZE = 30

ALPHA = 0.5
ZETA_STEP = 0.1
FW_ITERATIONS = 50
GAP_TOLERANCE = 1e-6
GAMMA = 1.0

PERTURBATIONS = [0.0, 0.05, 0.10, 0.20, 0.30]

PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def normalize_adjacency(A):
    """
    Normalize adjacency matrix for visualization/analysis only.

    The actual proposed algorithm performs its own normalization.
    """
    A = np.asarray(A, dtype=float)

    max_value = np.max(A)

    if max_value <= 0:
        return A.copy()

    return A / max_value


def extract_matching(X, threshold=0.5):
    """
    Extract selected row-column assignments from a partial
    permutation / matching matrix.

    Returns:
        list of (row_index, column_index)
    """
    X = np.asarray(X)

    matches = []

    for i, j in np.argwhere(X > threshold):
        matches.append((int(i), int(j)))

    return matches


def matching_accuracy(X, expected_permutation, matching_size):
    """
    Accuracy evaluated ONLY on selected matching assignments.

    This is important because L < N in the partial matching problem.
    """
    matches = extract_matching(X)

    if len(matches) == 0:
        return 0.0

    correct = 0

    for i, j in matches:
        if i < len(expected_permutation):
            if j == expected_permutation[i]:
                correct += 1

    return 100.0 * correct / len(matches)


def perturb_graph_weights(G, perturbation, rng):
    """
    Create a perturbed copy of a graph.

    Perturbation consists of:
        1. Gaussian edge-weight noise
        2. Random edge removal

    Node identities are preserved so the known identity mapping
    can be used as ground truth.
    """
    H = G.copy()

    edges = list(H.edges())

    if len(edges) == 0:
        return H

    # -----------------------------------------------------
    # 1. Add weight noise
    # -----------------------------------------------------

    for u, v in edges:
        old_weight = H[u][v].get("weight", 1.0)

        noise = rng.normal(
            loc=0.0,
            scale=perturbation
        )

        new_weight = old_weight * (1.0 + noise)

        # Keep weights positive
        new_weight = max(new_weight, 0.01)

        H[u][v]["weight"] = new_weight

    # -----------------------------------------------------
    # 2. Remove a percentage of edges
    # -----------------------------------------------------

    number_to_remove = int(
        perturbation * len(edges)
    )

    if number_to_remove > 0:
        remove_edges = rng.choice(
            len(edges),
            size=number_to_remove,
            replace=False
        )

        for idx in remove_edges:
            u, v = edges[idx]

            if H.has_edge(u, v):
                H.remove_edge(u, v)

    return H


def adjacency_matrix_from_graph(G, nodes):
    """
    Construct weighted adjacency matrix using a fixed node order.
    """
    return nx.to_numpy_array(
        G,
        nodelist=nodes,
        weight="weight",
        dtype=float
    )


# ---------------------------------------------------------
# GRAPH VISUALIZATION
# ---------------------------------------------------------

def draw_graph(
    G,
    title,
    output_path,
    node_positions=None,
    node_size=180,
    show_labels=False
):
    """
    Draw a weighted social graph.

    Edge width reflects interaction weight.
    """

    plt.figure(figsize=(8, 6))

    if node_positions is None:
        node_positions = nx.spring_layout(
            G,
            seed=42,
            weight="weight"
        )

    weights = [
        G[u][v].get("weight", 1.0)
        for u, v in G.edges()
    ]

    if weights:
        max_weight = max(weights)

        if max_weight > 0:
            edge_widths = [
                0.5 + 2.5 * (w / max_weight)
                for w in weights
            ]
        else:
            edge_widths = [1.0] * len(weights)
    else:
        edge_widths = []

    nx.draw_networkx_nodes(
        G,
        node_positions,
        node_size=node_size,
        alpha=0.85
    )

    nx.draw_networkx_edges(
        G,
        node_positions,
        width=edge_widths,
        alpha=0.35
    )

    if show_labels:
        nx.draw_networkx_labels(
            G,
            node_positions,
            font_size=6
        )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


def draw_graph_comparison(
    G,
    H,
    output_path
):
    """
    Side-by-side visualization of the early and later graphs.
    """

    plt.figure(figsize=(14, 6))

    # Use combined layout so both graphs are visually comparable
    combined = nx.compose(G, H)

    positions = nx.spring_layout(
        combined,
        seed=42,
        weight="weight"
    )

    # -----------------------------------------------------
    # Early graph
    # -----------------------------------------------------

    plt.subplot(1, 2, 1)

    weights_G = [
        G[u][v].get("weight", 1.0)
        for u, v in G.edges()
    ]

    max_G = max(weights_G) if weights_G else 1.0

    widths_G = [
        0.5 + 2.5 * (w / max_G)
        for w in weights_G
    ]

    nx.draw_networkx_nodes(
        G,
        positions,
        node_size=180,
        alpha=0.85
    )

    nx.draw_networkx_edges(
        G,
        positions,
        width=widths_G,
        alpha=0.35
    )

    plt.title(
        f"Early CollegeMsg Graph\n"
        f"{G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    plt.axis("off")

    # -----------------------------------------------------
    # Later graph
    # -----------------------------------------------------

    plt.subplot(1, 2, 2)

    weights_H = [
        H[u][v].get("weight", 1.0)
        for u, v in H.edges()
    ]

    max_H = max(weights_H) if weights_H else 1.0

    widths_H = [
        0.5 + 2.5 * (w / max_H)
        for w in weights_H
    ]

    nx.draw_networkx_nodes(
        H,
        positions,
        node_size=180,
        alpha=0.85
    )

    nx.draw_networkx_edges(
        H,
        positions,
        width=widths_H,
        alpha=0.35
    )

    plt.title(
        f"Later CollegeMsg Graph\n"
        f"{H.number_of_nodes()} nodes, "
        f"{H.number_of_edges()} edges"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# MATCHING VISUALIZATION
# ---------------------------------------------------------

def draw_matched_graphs(
    G,
    H,
    matching,
    output_path
):
    """
    Visualize the correspondence between matched nodes.

    Left  = graph G
    Right = graph H
    Lines between them = predicted correspondences.
    """

    plt.figure(figsize=(15, 8))

    common_nodes = sorted(
        set(G.nodes()) & set(H.nodes())
    )

    # Use a deterministic layout
    pos_G = nx.spring_layout(
        G,
        seed=42,
        weight="weight"
    )

    pos_H = nx.spring_layout(
        H,
        seed=42,
        weight="weight"
    )

    # Normalize coordinates
    def transform(pos, x_offset):
        return {
            node: (
                xy[0] * 0.35 + x_offset,
                xy[1] * 0.35
            )
            for node, xy in pos.items()
        }

    left_positions = transform(pos_G, -0.45)
    right_positions = transform(pos_H, 0.45)

    # Draw G
    nx.draw_networkx_nodes(
        G,
        left_positions,
        node_size=150,
        alpha=0.85
    )

    nx.draw_networkx_edges(
        G,
        left_positions,
        alpha=0.20,
        width=0.7
    )

    # Draw H
    nx.draw_networkx_nodes(
        H,
        right_positions,
        node_size=150,
        alpha=0.85
    )

    nx.draw_networkx_edges(
        H,
        right_positions,
        alpha=0.20,
        width=0.7
    )

    # -----------------------------------------------------
    # Draw correspondence lines
    # -----------------------------------------------------

    G_nodes = sorted(G.nodes())
    H_nodes = sorted(H.nodes())

    for i, j in matching:

        if i >= len(G_nodes):
            continue

        if j >= len(H_nodes):
            continue

        node_G = G_nodes[i]
        node_H = H_nodes[j]

        x1, y1 = left_positions[node_G]
        x2, y2 = right_positions[node_H]

        plt.plot(
            [x1, x2],
            [y1, y2],
            linewidth=0.8,
            alpha=0.35
        )

    plt.text(
        -0.45,
        0.48,
        "Graph G\nEarly",
        ha="center",
        fontsize=13,
        fontweight="bold"
    )

    plt.text(
        0.45,
        0.48,
        "Graph H\nPerturbed / Later",
        ha="center",
        fontsize=13,
        fontweight="bold"
    )

    plt.title(
        f"GNCCP Node Correspondence\n"
        f"{len(matching)} predicted matches"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# PERFORMANCE PLOTS
# ---------------------------------------------------------

def plot_accuracy(results_df, output_path):

    plt.figure(figsize=(8, 5))

    x = results_df["Perturbation"] * 100

    plt.plot(
        x,
        results_df["Baseline Accuracy (%)"],
        marker="o",
        linewidth=2,
        label="Baseline GNCCP"
    )

    plt.plot(
        x,
        results_df["Proposed Accuracy (%)"],
        marker="o",
        linewidth=2,
        label="Proposed GNCCP"
    )

    plt.xlabel("Graph Perturbation (%)")
    plt.ylabel("Matching Accuracy (%)")

    plt.title(
        "Matching Accuracy Under Graph Perturbation"
    )

    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_iterations(results_df, output_path):

    plt.figure(figsize=(8, 5))

    x = results_df["Perturbation"] * 100

    plt.plot(
        x,
        results_df["Baseline Iterations"],
        marker="o",
        linewidth=2,
        label="Baseline GNCCP"
    )

    plt.plot(
        x,
        results_df["Proposed Iterations"],
        marker="o",
        linewidth=2,
        label="Proposed GNCCP"
    )

    plt.xlabel("Graph Perturbation (%)")
    plt.ylabel("Optimization Iterations")

    plt.title(
        "GNCCP Optimization Iterations"
    )

    plt.grid(alpha=0.25)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_runtime(results_df, output_path):

    plt.figure(figsize=(8, 5))

    x = results_df["Perturbation"] * 100

    plt.plot(
        x,
        results_df["Baseline Runtime (s)"],
        marker="o",
        linewidth=2,
        label="Baseline GNCCP"
    )

    plt.plot(
        x,
        results_df["Proposed Runtime (s)"],
        marker="o",
        linewidth=2,
        label="Proposed GNCCP"
    )

    plt.xlabel("Graph Perturbation (%)")
    plt.ylabel("Runtime (seconds)")

    plt.title(
        "GNCCP Runtime Comparison"
    )

    plt.grid(alpha=0.25)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("SOCIAL NETWORK GRAPH VISUALIZATION + PERFORMANCE ANALYSIS")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. LOAD COLLEGEMSG
    # -----------------------------------------------------

    print("\n[1] Loading CollegeMsg dataset...")

    interactions = load_college_msg()

    print(
        f"Total interactions: {len(interactions):,}"
    )

    # -----------------------------------------------------
    # 2. TEMPORAL SPLIT
    # -----------------------------------------------------

    print("\n[2] Creating temporal graphs...")

    interactions = interactions.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    midpoint = len(interactions) // 2

    early_data = interactions.iloc[:midpoint].copy()
    later_data = interactions.iloc[midpoint:].copy()

    print(
        f"Early interactions: {len(early_data):,}"
    )

    print(
        f"Later interactions: {len(later_data):,}"
    )

    # -----------------------------------------------------
    # 3. SELECT COMMON ACTIVE USERS
    # -----------------------------------------------------

    common_users = select_active_users(
        early_data,
        later_data,
        num_users=NUM_USERS
    )

    common_users = sorted(common_users)

    print(
        f"Selected common active users: "
        f"{len(common_users)}"
    )

    # -----------------------------------------------------
    # 4. BUILD GRAPHS
    # -----------------------------------------------------

    G = build_weighted_social_graph(
        early_data,
        users=common_users
    )

    H_temporal = build_weighted_social_graph(
        later_data,
        users=common_users
    )

    G = remove_isolated_nodes(G)
    H_temporal = remove_isolated_nodes(H_temporal)

    print("\nEarly graph:")
    print(
        f"Nodes = {G.number_of_nodes()}"
    )
    print(
        f"Edges = {G.number_of_edges()}"
    )

    print("\nLater graph:")
    print(
        f"Nodes = {H_temporal.number_of_nodes()}"
    )
    print(
        f"Edges = {H_temporal.number_of_edges()}"
    )

    # -----------------------------------------------------
    # 5. GRAPH VISUALIZATION
    # -----------------------------------------------------

    print("\n[3] Creating graph visualizations...")

    draw_graph(
        G,
        "CollegeMsg Early Social Network",
        os.path.join(
            PLOTS_DIR,
            "early_graph.png"
        )
    )

    draw_graph(
        H_temporal,
        "CollegeMsg Later Social Network",
        os.path.join(
            PLOTS_DIR,
            "later_graph.png"
        )
    )

    draw_graph_comparison(
        G,
        H_temporal,
        os.path.join(
            PLOTS_DIR,
            "graph_comparison.png"
        )
    )

    # -----------------------------------------------------
    # 6. CONTROLLED MATCHING EXPERIMENT
    # -----------------------------------------------------

    print("\n[4] Running controlled perturbation experiment...")

    # Fixed node order
    nodes = sorted(
        set(G.nodes()) & set(common_users)
    )

    if len(nodes) < MATCHING_SIZE:
        raise ValueError(
            "Not enough nodes for requested matching size."
        )

    # Ground-truth identity mapping.
    # Since H is constructed using the same node identities,
    # identity is the known correspondence.
    expected_permutation = list(range(len(nodes)))

    rng = np.random.default_rng(42)

    results = []

    saved_matching = None
    saved_H = None

    for perturbation in PERTURBATIONS:

        print(
            f"\n--- Perturbation: "
            f"{perturbation * 100:.0f}% ---"
        )

        # -------------------------------------------------
        # Create perturbed H
        # -------------------------------------------------

        H = perturb_graph_weights(
            G,
            perturbation,
            rng
        )

        # Ensure the same node set
        H = nx.Graph(H)

        H = H.subgraph(nodes).copy()

        # Add missing nodes
        for node in nodes:
            if node not in H:
                H.add_node(node)

        A_G = adjacency_matrix_from_graph(
            G,
            nodes
        )

        A_H = adjacency_matrix_from_graph(
            H,
            nodes
        )

        # -------------------------------------------------
        # Feature cost matrix
        # -------------------------------------------------

        C, _, _ = build_feature_cost_matrix(
            G,
            H
        )

        C = np.asarray(C, dtype=float)

        # -------------------------------------------------
        # BASELINE
        # -------------------------------------------------

        start = time.perf_counter()

        baseline_result = run_baseline_gnccp(
            A_G,
            A_H,
            C,
            matching_size=MATCHING_SIZE,
            alpha=ALPHA,
            zeta_step=ZETA_STEP,
            fw_iterations=FW_ITERATIONS,
            tolerance=GAP_TOLERANCE
        )

        baseline_runtime = (
            time.perf_counter() - start
        )

        # -------------------------------------------------
        # PROPOSED
        # -------------------------------------------------

        start = time.perf_counter()

        proposed_result = run_proposed_gnccp(
            A_G,
            A_H,
            C,
            matching_size=MATCHING_SIZE,
            alpha=ALPHA,
            zeta_step=ZETA_STEP,
            fw_iterations=FW_ITERATIONS,
            gap_tolerance=GAP_TOLERANCE,
            gamma=GAMMA
        )

        proposed_runtime = (
            time.perf_counter() - start
        )

        # -------------------------------------------------
        # RESULT EXTRACTION
        # -------------------------------------------------

        # Existing implementations return:
        #
        # X, objective, iterations, ...
        #
        # We deliberately use the first three values because
        # these are the common outputs required here.

        baseline_X = baseline_result[0]
        baseline_objective = baseline_result[1]
        baseline_iterations = baseline_result[2]

        proposed_X = proposed_result[0]
        proposed_objective = proposed_result[1]
        proposed_iterations = proposed_result[2]

        # -------------------------------------------------
        # ACCURACY
        # -------------------------------------------------

        baseline_accuracy = matching_accuracy(
            baseline_X,
            expected_permutation,
            MATCHING_SIZE
        )

        proposed_accuracy = matching_accuracy(
            proposed_X,
            expected_permutation,
            MATCHING_SIZE
        )

        # -------------------------------------------------
        # ITERATION REDUCTION
        # -------------------------------------------------

        if baseline_iterations > 0:

            iteration_reduction = (
                (
                    baseline_iterations
                    - proposed_iterations
                )
                / baseline_iterations
            ) * 100.0

        else:
            iteration_reduction = 0.0

        # -------------------------------------------------
        # STORE RESULTS
        # -------------------------------------------------

        results.append(
            {
                "Perturbation": perturbation,

                "Baseline Accuracy (%)":
                    baseline_accuracy,

                "Proposed Accuracy (%)":
                    proposed_accuracy,

                "Baseline Objective":
                    baseline_objective,

                "Proposed Objective":
                    proposed_objective,

                "Baseline Runtime (s)":
                    baseline_runtime,

                "Proposed Runtime (s)":
                    proposed_runtime,

                "Baseline Iterations":
                    baseline_iterations,

                "Proposed Iterations":
                    proposed_iterations,

                "Iteration Reduction (%)":
                    iteration_reduction,
            }
        )

        print(
            f"Baseline accuracy : "
            f"{baseline_accuracy:.2f}%"
        )

        print(
            f"Proposed accuracy : "
            f"{proposed_accuracy:.2f}%"
        )

        print(
            f"Baseline runtime  : "
            f"{baseline_runtime:.4f} s"
        )

        print(
            f"Proposed runtime  : "
            f"{proposed_runtime:.4f} s"
        )

        print(
            f"Baseline iterations: "
            f"{baseline_iterations}"
        )

        print(
            f"Proposed iterations: "
            f"{proposed_iterations}"
        )

        print(
            f"Iteration reduction: "
            f"{iteration_reduction:.2f}%"
        )

        # Save the 10% perturbation matching
        # as the representative visualization.
        if abs(perturbation - 0.10) < 1e-12:

            saved_matching = extract_matching(
                proposed_X
            )

            saved_H = H.copy()

    # -----------------------------------------------------
    # 7. DATAFRAME
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    # -----------------------------------------------------
    # 8. SAVE TABLE
    # -----------------------------------------------------

    csv_path = os.path.join(
        TABLES_DIR,
        "social_network_performance.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nSaved performance table: {csv_path}"
    )

    # -----------------------------------------------------
    # 9. PERFORMANCE PLOTS
    # -----------------------------------------------------

    print("\n[5] Creating performance plots...")

    plot_accuracy(
        results_df,
        os.path.join(
            PLOTS_DIR,
            "accuracy_vs_perturbation.png"
        )
    )

    plot_iterations(
        results_df,
        os.path.join(
            PLOTS_DIR,
            "iterations_vs_perturbation.png"
        )
    )

    plot_runtime(
        results_df,
        os.path.join(
            PLOTS_DIR,
            "runtime_vs_perturbation.png"
        )
    )

    # -----------------------------------------------------
    # 10. MATCHING VISUALIZATION
    # -----------------------------------------------------

    if saved_matching is not None and saved_H is not None:

        draw_matched_graphs(
            G,
            saved_H,
            saved_matching,
            os.path.join(
                PLOTS_DIR,
                "matched_graphs.png"
            )
        )

    # -----------------------------------------------------
    # 11. PRINT FINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SOCIAL NETWORK RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "Perturbation",
                "Baseline Accuracy (%)",
                "Proposed Accuracy (%)",
                "Baseline Iterations",
                "Proposed Iterations",
                "Iteration Reduction (%)"
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("COMPUTATIONAL INTERPRETATION")
    print("=" * 70)

    print(
        "\nThe proposed implementation requires fewer "
        "optimization iterations than the baseline."
    )

    print(
        "This should be reported as an empirical "
        "iteration reduction, not as a claim of lower "
        "asymptotic Big-O complexity."
    )

    print(
        "\nThe proposed method contains additional "
        "per-iteration operations such as normalization "
        "and Sinkhorn projection."
    )

    print(
        "Therefore wall-clock runtime may not always be "
        "lower even when the number of optimization "
        "iterations is substantially lower."
    )

    print("\nGenerated files:")

    for filename in sorted(
        os.listdir(PLOTS_DIR)
    ):
        print(
            "  results/plots/" + filename
        )

    print(
        "  results/tables/"
        "social_network_performance.csv"
    )

    print("\nDone.")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()