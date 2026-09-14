"""
Final Baseline vs Proposed WCS-GNCCP Comparison
================================================

Real-world social-network experiment using the Stanford
CollegeMsg dataset.

The experiment compares:

1. Original WCS-GNCCP baseline
2. Proposed WCS-GNCCP pipeline

Metrics:
- Matching accuracy
- WCS objective
- Runtime
- Number of continuation iterations
- Kernel similarity

Two experiments are performed:

A. Controlled perturbation of a real social graph
B. Temporal early-vs-later CollegeMsg graphs
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Make src importable
# ------------------------------------------------------------

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.data_loader import load_college_msg
from src.graph_builder import build_real_social_graphs
from src.graph_features import build_feature_cost_matrix

from src.baseline_gnccp import run_baseline_gnccp
from src.proposed_gnccp import run_proposed_gnccp


# ============================================================
# Configuration
# ============================================================

MATCHING_SIZE = 30

ALPHA = 0.5

ZETA_STEP = 0.1

FW_ITERATIONS = 50

GAP_TOLERANCE = 1e-6

GAMMA = 1.0

PERTURBATION_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
]


# ============================================================
# Utility functions
# ============================================================

def adjacency_from_graph(G, nodes):
    """
    Convert weighted NetworkX graph to adjacency matrix
    using a fixed node ordering.
    """

    import networkx as nx

    return nx.to_numpy_array(
        G,
        nodelist=nodes,
        weight="weight",
        dtype=float
    )

def identity_accuracy(X):
    """
    Accuracy for a partial matching whose ground-truth
    correspondence is identity.

    Only actually selected matches are evaluated.

    If X contains L selected assignments, accuracy is:

        correct selected matches / total selected matches

    This avoids counting unmatched rows as predictions.
    """

    X = np.asarray(
        X,
        dtype=float
    )

    # Find actual selected assignments.
    selected = np.argwhere(
        X > 0.5
    )

    if len(selected) == 0:
        return 0.0

    correct = 0

    for row, col in selected:

        if row == col:
            correct += 1

    return (
        100.0
        * correct
        / len(selected)
    )


def run_baseline(
    AG,
    AH,
    C,
    L
):
    """
    Run original baseline and measure runtime.
    """

    start = time.perf_counter()

    result = run_baseline_gnccp(
        AG,
        AH,
        C,
        matching_size=L,
        alpha=ALPHA,
        zeta_step=ZETA_STEP,
        fw_iterations=FW_ITERATIONS,
        tolerance=GAP_TOLERANCE
    )

    runtime = (
        time.perf_counter()
        - start
    )

    X = result[0]
    objective = result[1]
    iterations = result[2]

    accuracy = identity_accuracy(X)

    return {
        "accuracy": accuracy,
        "objective": objective,
        "runtime": runtime,
        "iterations": iterations,
    }


def run_proposed(
    AG,
    AH,
    C,
    L
):
    """
    Run proposed pipeline and measure runtime.
    """

    start = time.perf_counter()

    result = run_proposed_gnccp(
        AG,
        AH,
        C,
        matching_size=L,
        alpha=ALPHA,
        zeta_step=ZETA_STEP,
        fw_iterations=FW_ITERATIONS,
        gap_tolerance=GAP_TOLERANCE,
        gamma=GAMMA
    )

    runtime = (
        time.perf_counter()
        - start
    )

    X = result[0]
    objective = result[1]
    iterations = result[2]
    kernel = result[3]

    accuracy = identity_accuracy(X)

    return {
        "accuracy": accuracy,
        "objective": objective,
        "runtime": runtime,
        "iterations": iterations,
        "kernel": kernel,
    }


# ============================================================
# Controlled perturbation experiment
# ============================================================

def controlled_experiment(
    G,
    nodes
):
    """
    Perturb the real social graph while preserving
    node identities.

    This creates a controlled real-data robustness
    experiment.

    IMPORTANT:
    The perturbation is applied to the real graph;
    this is not synthetic random graph generation.
    """

    print("\n")
    print("=" * 75)
    print("EXPERIMENT A: CONTROLLED REAL-GRAPH PERTURBATION")
    print("=" * 75)

    AG = adjacency_from_graph(
        G,
        nodes
    )

    results = []

    rng = np.random.default_rng(42)

    for level in PERTURBATION_LEVELS:

        print("\n")
        print("-" * 75)
        print(
            f"Perturbation level: "
            f"{level:.2f}"
        )
        print("-" * 75)

        # ----------------------------------------------------
        # Copy original adjacency
        # ----------------------------------------------------

        AH = AG.copy()

        # ----------------------------------------------------
        # Perturb edge weights
        # ----------------------------------------------------

        if level > 0:

            noise = rng.normal(
                loc=0.0,
                scale=level,
                size=AH.shape
            )

            # Preserve symmetry
            noise = (
                noise + noise.T
            ) / 2.0

            AH = AH + noise

            AH = np.maximum(
                AH,
                0.0
            )

            # Preserve zero locations
            original_edges = (
                AG > 0
            )

            AH *= original_edges

            # Randomly remove a fraction of edges
            upper = np.triu(
                AG > 0,
                k=1
            )

            edge_positions = np.argwhere(
                upper
            )

            number_to_remove = int(
                level * len(edge_positions)
            )

            if number_to_remove > 0:

                chosen = rng.choice(
                    len(edge_positions),
                    size=number_to_remove,
                    replace=False
                )

                for idx in chosen:

                    i, j = (
                        edge_positions[idx]
                    )

                    AH[i, j] = 0.0
                    AH[j, i] = 0.0

        # ----------------------------------------------------
        # Build perturbed graph
        # ----------------------------------------------------

        import networkx as nx

        H = nx.from_numpy_array(
            AH
        )

        # ----------------------------------------------------
        # Recompute node costs using BOTH graphs
        # ----------------------------------------------------

        C,_,_ = build_feature_cost_matrix(
            G,
            H
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline = run_baseline(
            AG,
            AH,
            C,
            MATCHING_SIZE
        )

        # ----------------------------------------------------
        # Proposed
        # ----------------------------------------------------

        proposed = run_proposed(
            AG,
            AH,
            C,
            MATCHING_SIZE
        )

        print(
            f"Baseline accuracy : "
            f"{baseline['accuracy']:.2f}%"
        )

        print(
            f"Proposed accuracy : "
            f"{proposed['accuracy']:.2f}%"
        )

        print(
            f"Baseline objective: "
            f"{baseline['objective']:.6f}"
        )

        print(
            f"Proposed objective: "
            f"{proposed['objective']:.6f}"
        )

        print(
            f"Baseline runtime   : "
            f"{baseline['runtime']:.4f} s"
        )

        print(
            f"Proposed runtime   : "
            f"{proposed['runtime']:.4f} s"
        )

        print(
            f"Baseline iterations: "
            f"{baseline['iterations']}"
        )

        print(
            f"Proposed iterations: "
            f"{proposed['iterations']}"
        )

        print(
            f"Proposed kernel    : "
            f"{proposed['kernel']:.8f}"
        )

        results.append({
            "Experiment": "Controlled Real Graph",
            "Perturbation": level,

            "Baseline Accuracy (%)":
                baseline["accuracy"],

            "Proposed Accuracy (%)":
                proposed["accuracy"],

            "Baseline Objective":
                baseline["objective"],

            "Proposed Objective":
                proposed["objective"],

            "Baseline Runtime (s)":
                baseline["runtime"],

            "Proposed Runtime (s)":
                proposed["runtime"],

            "Baseline Iterations":
                baseline["iterations"],

            "Proposed Iterations":
                proposed["iterations"],

            "Proposed Kernel":
                proposed["kernel"],
        })

    return results


# ============================================================
# Temporal CollegeMsg experiment
# ============================================================

def temporal_experiment(
    early_graph,
    later_graph,
    nodes
):
    """
    Compare early and later temporal snapshots of the
    real CollegeMsg social network.

    IMPORTANT:
    There is no assumed identity ground truth here.

    Therefore matching accuracy is NOT reported.

    We instead compare:
        - objective
        - runtime
        - iterations
        - kernel
    """

    print("\n")
    print("=" * 75)
    print("EXPERIMENT B: TEMPORAL COLLEGEMSG GRAPH")
    print("=" * 75)

    AG = adjacency_from_graph(
        early_graph,
        nodes
    )

    AH = adjacency_from_graph(
        later_graph,
        nodes
    )

    # Cost matrix based on structural/social features
    C,_,_ = build_feature_cost_matrix(
        early_graph,
        later_graph
    )

    print(
        f"\nNodes in each graph: "
        f"{len(nodes)}"
    )

    print(
        f"Matching size L: "
        f"{MATCHING_SIZE}"
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    start = time.perf_counter()

    baseline_result = run_baseline_gnccp(
        AG,
        AH,
        C,
        matching_size=MATCHING_SIZE,
        alpha=ALPHA,
        zeta_step=ZETA_STEP,
        fw_iterations=FW_ITERATIONS,
        tolerance=GAP_TOLERANCE
    )

    baseline_runtime = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Proposed
    # --------------------------------------------------------

    start = time.perf_counter()

    proposed_result = run_proposed_gnccp(
        AG,
        AH,
        C,
        matching_size=MATCHING_SIZE,
        alpha=ALPHA,
        zeta_step=ZETA_STEP,
        fw_iterations=FW_ITERATIONS,
        gap_tolerance=GAP_TOLERANCE,
        gamma=GAMMA
    )

    proposed_runtime = (
        time.perf_counter()
        - start
    )

    baseline_objective = (
        baseline_result[1]
    )

    proposed_objective = (
        proposed_result[1]
    )

    baseline_iterations = (
        baseline_result[2]
    )

    proposed_iterations = (
        proposed_result[2]
    )

    proposed_kernel = (
        proposed_result[3]
    )

    print("\nTemporal comparison:")
    print(
        f"Baseline objective : "
        f"{baseline_objective:.6f}"
    )

    print(
        f"Proposed objective : "
        f"{proposed_objective:.6f}"
    )

    print(
        f"Baseline runtime   : "
        f"{baseline_runtime:.4f} s"
    )

    print(
        f"Proposed runtime   : "
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
        f"Proposed kernel    : "
        f"{proposed_kernel:.8f}"
    )

    return {
        "Experiment": "Temporal CollegeMsg",
        "Perturbation": np.nan,

        "Baseline Accuracy (%)":
            np.nan,

        "Proposed Accuracy (%)":
            np.nan,

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

        "Proposed Kernel":
            proposed_kernel,
    }


# ============================================================
# Plot results
# ============================================================

def create_plots(df):
    """
    Generate panel-ready comparison plots.
    """

    os.makedirs(
        "results/plots",
        exist_ok=True
    )

    controlled = df[
        df["Experiment"]
        == "Controlled Real Graph"
    ]

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        controlled["Perturbation"],
        controlled["Baseline Accuracy (%)"],
        marker="o",
        label="Original WCS-GNCCP"
    )

    plt.plot(
        controlled["Perturbation"],
        controlled["Proposed Accuracy (%)"],
        marker="s",
        label="Proposed WCS-GNCCP"
    )

    plt.xlabel(
        "Perturbation Level"
    )

    plt.ylabel(
        "Matching Accuracy (%)"
    )

    plt.title(
        "Social Graph Matching Accuracy"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "results/plots/"
        "accuracy_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        controlled["Perturbation"],
        controlled["Baseline Objective"],
        marker="o",
        label="Original WCS-GNCCP"
    )

    plt.plot(
        controlled["Perturbation"],
        controlled["Proposed Objective"],
        marker="s",
        label="Proposed WCS-GNCCP"
    )

    plt.xlabel(
        "Perturbation Level"
    )

    plt.ylabel(
        "WCS Objective"
    )

    plt.title(
        "WCS Objective under Graph Perturbation"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "results/plots/"
        "objective_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        controlled["Perturbation"],
        controlled["Baseline Runtime (s)"],
        marker="o",
        label="Original WCS-GNCCP"
    )

    plt.plot(
        controlled["Perturbation"],
        controlled["Proposed Runtime (s)"],
        marker="s",
        label="Proposed WCS-GNCCP"
    )

    plt.xlabel(
        "Perturbation Level"
    )

    plt.ylabel(
        "Runtime (seconds)"
    )

    plt.title(
        "Runtime Comparison"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "results/plots/"
        "runtime_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Iterations
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        controlled["Perturbation"],
        controlled["Baseline Iterations"],
        marker="o",
        label="Original WCS-GNCCP"
    )

    plt.plot(
        controlled["Perturbation"],
        controlled["Proposed Iterations"],
        marker="s",
        label="Proposed WCS-GNCCP"
    )

    plt.xlabel(
        "Perturbation Level"
    )

    plt.ylabel(
        "Iterations"
    )

    plt.title(
        "Convergence Iterations"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "results/plots/"
        "iteration_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print("FINAL SOCIAL-NETWORK WCS-GNCCP COMPARISON")
    print("=" * 75)

    # --------------------------------------------------------
    # Load CollegeMsg
    # --------------------------------------------------------

    print("\nLoading CollegeMsg dataset...")
    print("\nLoading CollegeMsg dataset...")

    data = load_college_msg()

    # Sort interactions chronologically
    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Split the temporal dataset into two halves
    split_index = len(data) // 2

    early_df = data.iloc[
        :split_index
    ].copy()

    later_df = data.iloc[
        split_index:
    ].copy()

    print(
        f"Total interactions: {len(data)}"
    )

    print(
        f"Early interactions: {len(early_df)}"
    )

    print(
        f"Later interactions: {len(later_df)}"
    )

    # Build the two real social graphs
    early_graph, later_graph, selected_users = (
        build_real_social_graphs(
            early_df,
            later_df,
            num_users=40
        )
    )

    nodes = sorted(
        selected_users
    )

    print(
        f"\nSelected users: {len(nodes)}"
    )

    # --------------------------------------------------------
    # Experiment A
    # --------------------------------------------------------

    controlled_results = (
        controlled_experiment(
            early_graph,
            nodes
        )
    )

    # --------------------------------------------------------
    # Experiment B
    # --------------------------------------------------------

    temporal_result = (
        temporal_experiment(
            early_graph,
            later_graph,
            nodes
        )
    )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    all_results = (
        controlled_results
        + [temporal_result]
    )

    df = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # Save table
    # --------------------------------------------------------

    os.makedirs(
        "results/tables",
        exist_ok=True
    )

    output_file = (
        "results/tables/"
        "final_social_comparison.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Print final table
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("FINAL RESULTS TABLE")
    print("=" * 75)

    print(
        df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved table to:\n"
        f"{output_file}"
    )

    # --------------------------------------------------------
    # Create plots
    # --------------------------------------------------------

    create_plots(
        df
    )

    print(
        "\nPlots saved in:\n"
        "results/plots/"
    )

    print("\n")
    print("=" * 75)
    print("EXPERIMENT COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()