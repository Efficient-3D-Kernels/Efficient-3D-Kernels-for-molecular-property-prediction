"""
Repeated Noise Robustness Experiment
=====================================

Purpose:
    Evaluate Baseline GNCCP vs Proposed GNCCP over multiple
    independent perturbation trials using the real CollegeMsg graph.

Experiment:
    - Real Stanford CollegeMsg data
    - 40 selected active users
    - 30-node partial matching
    - Perturbation levels: 0%, 5%, 10%, 20%, 30%
    - 10 independent trials per perturbation level

Outputs:
    results/tables/repeated_noise_results.csv
    results/tables/repeated_noise_summary.csv

Metrics:
    - Matching accuracy
    - Common normalized WCS objective
    - Runtime
    - Iterations
    - Proposed similarity kernel

Important:
    Accuracy is measured against the original graph identity
    because the perturbed graph preserves node identities.
"""

import os
import sys
import time
import random

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# Make src importable
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# Project imports
# ============================================================

from data_loader import load_college_msg
from graph_builder import build_real_social_graphs
from graph_features import build_feature_cost_matrix
from baseline_gnccp import run_baseline_gnccp
from proposed_gnccp import run_proposed_gnccp


# ============================================================
# Experiment configuration
# ============================================================

NUM_USERS = 40
MATCHING_SIZE = 30

PERTURBATION_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
TRIALS_PER_LEVEL = 10

ALPHA = 0.5
GAMMA = 1.0

ZETA_STEP = 0.1
FW_ITERATIONS = 50

GAP_TOLERANCE = 1e-6

BASE_SEED = 2026

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

os.makedirs(TABLES_DIR, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================

def set_seed(seed):
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


def graph_to_adjacency(G, nodes):
    """
    Convert a NetworkX weighted graph into a symmetric
    weighted adjacency matrix.
    """
    A = nx.to_numpy_array(
        G,
        nodelist=nodes,
        weight="weight",
        dtype=float
    )

    return A


def normalize_matrix(A):
    """
    Min-max normalization.

    If all values are identical, return zeros.
    """
    A = np.asarray(A, dtype=float)

    min_val = np.min(A)
    max_val = np.max(A)

    if max_val - min_val < 1e-12:
        return np.zeros_like(A)

    return (A - min_val) / (max_val - min_val)


def common_normalized_objective(
    X,
    A_G,
    A_H,
    C,
    alpha=0.5
):
    """
    Evaluate both methods under the SAME normalized WCS objective.

    F(X) =
        alpha || U o A_G - X A_H X^T ||_F^2
        + (1-alpha) tr(C^T X)

    where

        U = r r^T
        r = X 1

    The adjacency matrices and cost matrix are normalized
    before this function is called.
    """

    X = np.asarray(X, dtype=float)

    r = X @ np.ones(X.shape[1])
    U = np.outer(r, r)

    structural_part = U * A_G - X @ A_H @ X.T

    structural_loss = np.linalg.norm(
        structural_part,
        ord="fro"
    ) ** 2

    feature_loss = np.trace(C.T @ X)

    objective = (
        alpha * structural_loss
        + (1.0 - alpha) * feature_loss
    )

    return float(objective)


def identity_accuracy(X):
    """
    Accuracy of the selected partial matching.

    The expected correspondence is identity:
        node i in G <-> node i in H

    Only assignments actually selected by X are counted.

    This is important because L < n in the partial matching
    problem.
    """

    X = np.asarray(X)

    selected_pairs = np.argwhere(X > 0.5)

    if len(selected_pairs) == 0:
        return 0.0

    correct = 0

    for i, j in selected_pairs:

        if i == j:
            correct += 1

    return 100.0 * correct / len(selected_pairs)


def extract_iterations(result):
    """
    Try to robustly extract iteration count from the result
    returned by the existing GNCCP implementations.

    Different versions of the implementation may return
    slightly different tuple structures.
    """

    if isinstance(result, dict):

        possible_keys = [
            "iterations",
            "iteration",
            "iters",
            "n_iterations",
            "num_iterations"
        ]

        for key in possible_keys:
            if key in result:
                return int(result[key])

        return None

    if isinstance(result, tuple):

        for item in result:

            if isinstance(item, (int, np.integer)):
                value = int(item)

                if value > 0:
                    return value

            if isinstance(item, dict):

                value = extract_iterations(item)

                if value is not None:
                    return value

    return None


def extract_solution(result):
    """
    Extract X from the existing solver output.

    The current project implementations return X as the first
    element of their result tuple.
    """

    if isinstance(result, np.ndarray):
        return result

    if isinstance(result, tuple):

        for item in result:

            if isinstance(item, np.ndarray):

                if item.ndim == 2:
                    return item

    if isinstance(result, dict):

        possible_keys = [
            "X",
            "x",
            "solution",
            "matching"
        ]

        for key in possible_keys:

            if key in result:

                value = result[key]

                if isinstance(value, np.ndarray):
                    return value

    raise ValueError(
        "Could not extract matching matrix X from solver output."
    )


def extract_kernel(result):
    """
    Extract the proposed kernel/similarity value if available.
    """

    if isinstance(result, dict):

        possible_keys = [
            "kernel",
            "similarity",
            "graph_kernel",
            "K"
        ]

        for key in possible_keys:

            if key in result:

                return float(result[key])

    if isinstance(result, tuple):

        for item in result:

            if isinstance(item, (float, np.floating)):

                value = float(item)

                if 0.0 <= value <= 1.0:
                    return value

    return np.nan


# ============================================================
# Perturbation
# ============================================================

def perturb_graph(
    G,
    perturbation,
    seed
):
    """
    Create a perturbed copy of G.

    Perturbation has two components:

    1. Randomly modify edge weights.
    2. Randomly remove a fraction of edges.

    Node identities are preserved.

    This allows us to know the ground-truth correspondence:
        original node i -> perturbed node i
    """

    rng = np.random.default_rng(seed)

    H = G.copy()

    edges = list(H.edges(data=True))

    if perturbation <= 0.0:
        return H

    # --------------------------------------------------------
    # 1. Perturb edge weights
    # --------------------------------------------------------

    for u, v, data in H.edges(data=True):

        original_weight = float(
            data.get("weight", 1.0)
        )

        noise = rng.uniform(
            1.0 - perturbation,
            1.0 + perturbation
        )

        new_weight = original_weight * noise

        H[u][v]["weight"] = max(
            new_weight,
            1e-6
        )

    # --------------------------------------------------------
    # 2. Remove edges
    # --------------------------------------------------------

    number_of_edges = H.number_of_edges()

    number_to_remove = int(
        perturbation * number_of_edges
    )

    if number_to_remove > 0:

        removable_edges = list(H.edges())

        rng.shuffle(removable_edges)

        edges_to_remove = removable_edges[
            :number_to_remove
        ]

        H.remove_edges_from(edges_to_remove)

    return H


# ============================================================
# Build base graph
# ============================================================

def build_base_graph():
    """
    Load CollegeMsg and construct the real social graph.
    """

    print("\n" + "=" * 70)
    print("LOADING REAL COLLEGEMSG DATASET")
    print("=" * 70)

    data = load_college_msg()

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    split_index = len(data) // 2

    early_df = data.iloc[
        :split_index
    ].copy()

    later_df = data.iloc[
        split_index:
    ].copy()

    print(f"Total interactions : {len(data)}")
    print(f"Early interactions : {len(early_df)}")
    print(f"Later interactions : {len(later_df)}")

    G, H_real, selected_users = build_real_social_graphs(
        early_df,
        later_df,
        num_users=NUM_USERS
    )

    print(
        f"Selected users     : {len(selected_users)}"
    )

    nodes = sorted(selected_users)

    # Reorder graph nodes consistently.
    G = G.subgraph(nodes).copy()

    return G, nodes


# ============================================================
# Run one trial
# ============================================================

def run_single_trial(
    G,
    nodes,
    perturbation,
    trial_number
):
    """
    Run baseline and proposed GNCCP for one perturbation trial.
    """

    seed = (
        BASE_SEED
        + int(perturbation * 1000) * 100
        + trial_number
    )

    set_seed(seed)

    # --------------------------------------------------------
    # Generate perturbed graph
    # --------------------------------------------------------

    H = perturb_graph(
        G,
        perturbation,
        seed
    )

    # Ensure identical node ordering.
    H = H.subgraph(nodes).copy()

    # --------------------------------------------------------
    # Construct matrices
    # --------------------------------------------------------

    A_G_raw = graph_to_adjacency(
        G,
        nodes
    )

    A_H_raw = graph_to_adjacency(
        H,
        nodes
    )

    # Common normalized matrices.
    A_G = normalize_matrix(A_G_raw)
    A_H = normalize_matrix(A_H_raw)

    # Feature cost.
    C, _, _ = build_feature_cost_matrix(
        G,
        H
    )

    C = normalize_matrix(C)

    # --------------------------------------------------------
    # Baseline GNCCP
    # --------------------------------------------------------

    baseline_start = time.perf_counter()

    baseline_result = run_baseline_gnccp(
        A_G_raw,
        A_H_raw,
        C,
        matching_size=MATCHING_SIZE,
        alpha=ALPHA,
        zeta_step=ZETA_STEP,
        fw_iterations=FW_ITERATIONS,
        tolerance=GAP_TOLERANCE
    )

    baseline_runtime = (
        time.perf_counter()
        - baseline_start
    )

    baseline_X = extract_solution(
        baseline_result
    )

    baseline_iterations = extract_iterations(
        baseline_result
    )

    # --------------------------------------------------------
    # Proposed GNCCP
    # --------------------------------------------------------

    proposed_start = time.perf_counter()

    proposed_result = run_proposed_gnccp(
        A_G_raw,
        A_H_raw,
        C,
        L=MATCHING_SIZE,
        alpha=ALPHA,
        zeta_step=ZETA_STEP,
        fw_iterations=FW_ITERATIONS,
        gap_tolerance=GAP_TOLERANCE,
        gamma=GAMMA
    )

    proposed_runtime = (
        time.perf_counter()
        - proposed_start
    )

    proposed_X = extract_solution(
        proposed_result
    )

    proposed_iterations = extract_iterations(
        proposed_result
    )

    proposed_kernel = extract_kernel(
        proposed_result
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    baseline_accuracy = identity_accuracy(
        baseline_X
    )

    proposed_accuracy = identity_accuracy(
        proposed_X
    )

    # --------------------------------------------------------
    # COMMON objective
    # --------------------------------------------------------

    baseline_common_objective = (
        common_normalized_objective(
            baseline_X,
            A_G,
            A_H,
            C,
            alpha=ALPHA
        )
    )

    proposed_common_objective = (
        common_normalized_objective(
            proposed_X,
            A_G,
            A_H,
            C,
            alpha=ALPHA
        )
    )

    # --------------------------------------------------------
    # Result row
    # --------------------------------------------------------

    result = {
        "Experiment": "Repeated Controlled Real Graph",
        "Perturbation": perturbation,
        "Trial": trial_number,
        "Seed": seed,

        "Baseline Accuracy (%)":
            baseline_accuracy,

        "Proposed Accuracy (%)":
            proposed_accuracy,

        "Baseline Common Objective":
            baseline_common_objective,

        "Proposed Common Objective":
            proposed_common_objective,

        "Baseline Runtime (s)":
            baseline_runtime,

        "Proposed Runtime (s)":
            proposed_runtime,

        "Baseline Iterations":
            baseline_iterations,

        "Proposed Iterations":
            proposed_iterations,

        "Proposed Kernel":
            proposed_kernel
    }

    return result


# ============================================================
# Main experiment
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("REPEATED NOISE ROBUSTNESS EXPERIMENT")
    print("=" * 70)

    print(f"Users              : {NUM_USERS}")
    print(f"Matching size      : {MATCHING_SIZE}")
    print(f"Trials / level     : {TRIALS_PER_LEVEL}")
    print(
        f"Perturbation levels: "
        f"{[int(x * 100) for x in PERTURBATION_LEVELS]}%"
    )

    # --------------------------------------------------------
    # Build real base graph
    # --------------------------------------------------------

    G, nodes = build_base_graph()

    print("\nBase graph:")
    print(f"Nodes : {G.number_of_nodes()}")
    print(f"Edges : {G.number_of_edges()}")

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    all_results = []

    total_runs = (
        len(PERTURBATION_LEVELS)
        * TRIALS_PER_LEVEL
    )

    current_run = 0

    for perturbation in PERTURBATION_LEVELS:

        print("\n")
        print("-" * 70)
        print(
            f"PERTURBATION = "
            f"{perturbation * 100:.0f}%"
        )
        print("-" * 70)

        for trial in range(
            1,
            TRIALS_PER_LEVEL + 1
        ):

            current_run += 1

            print(
                f"Trial {trial:02d}/"
                f"{TRIALS_PER_LEVEL} "
                f""
                f"(overall "
                f"{current_run}/{total_runs})...",
                end=" ",
                flush=True
            )

            try:

                result = run_single_trial(
                    G,
                    nodes,
                    perturbation,
                    trial
                )

                all_results.append(
                    result
                )

                print(
                    f"Baseline = "
                    f"{result['Baseline Accuracy (%)']:.1f}% | "
                    f"Proposed = "
                    f"{result['Proposed Accuracy (%)']:.1f}%"
                )

            except Exception as e:

                print("FAILED")

                print(
                    f"  Error: {type(e).__name__}: {e}"
                )

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    raw_path = os.path.join(
        TABLES_DIR,
        "repeated_noise_results.csv"
    )

    results_df.to_csv(
        raw_path,
        index=False
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    summary_rows = []

    for perturbation in PERTURBATION_LEVELS:

        subset = results_df[
            results_df["Perturbation"]
            == perturbation
        ]

        if len(subset) == 0:
            continue

        baseline_acc_mean = subset[
            "Baseline Accuracy (%)"
        ].mean()

        baseline_acc_std = subset[
            "Baseline Accuracy (%)"
        ].std(ddof=1)

        proposed_acc_mean = subset[
            "Proposed Accuracy (%)"
        ].mean()

        proposed_acc_std = subset[
            "Proposed Accuracy (%)"
        ].std(ddof=1)

        baseline_obj_mean = subset[
            "Baseline Common Objective"
        ].mean()

        baseline_obj_std = subset[
            "Baseline Common Objective"
        ].std(ddof=1)

        proposed_obj_mean = subset[
            "Proposed Common Objective"
        ].mean()

        proposed_obj_std = subset[
            "Proposed Common Objective"
        ].std(ddof=1)

        baseline_runtime_mean = subset[
            "Baseline Runtime (s)"
        ].mean()

        baseline_runtime_std = subset[
            "Baseline Runtime (s)"
        ].std(ddof=1)

        proposed_runtime_mean = subset[
            "Proposed Runtime (s)"
        ].mean()

        proposed_runtime_std = subset[
            "Proposed Runtime (s)"
        ].std(ddof=1)

        baseline_iterations_mean = subset[
            "Baseline Iterations"
        ].mean()

        baseline_iterations_std = subset[
            "Baseline Iterations"
        ].std(ddof=1)

        proposed_iterations_mean = subset[
            "Proposed Iterations"
        ].mean()

        proposed_iterations_std = subset[
            "Proposed Iterations"
        ].std(ddof=1)

        kernel_mean = subset[
            "Proposed Kernel"
        ].mean()

        kernel_std = subset[
            "Proposed Kernel"
        ].std(ddof=1)

        accuracy_improvement = (
            proposed_acc_mean
            - baseline_acc_mean
        )

        iteration_reduction = np.nan

        if baseline_iterations_mean > 0:

            iteration_reduction = (
                100.0
                * (
                    baseline_iterations_mean
                    - proposed_iterations_mean
                )
                / baseline_iterations_mean
            )

        summary_rows.append({

            "Perturbation (%)":
                perturbation * 100,

            "Trials":
                len(subset),

            "Baseline Accuracy Mean (%)":
                baseline_acc_mean,

            "Baseline Accuracy Std (%)":
                baseline_acc_std,

            "Proposed Accuracy Mean (%)":
                proposed_acc_mean,

            "Proposed Accuracy Std (%)":
                proposed_acc_std,

            "Accuracy Improvement (pp)":
                accuracy_improvement,

            "Baseline Common Objective Mean":
                baseline_obj_mean,

            "Baseline Common Objective Std":
                baseline_obj_std,

            "Proposed Common Objective Mean":
                proposed_obj_mean,

            "Proposed Common Objective Std":
                proposed_obj_std,

            "Baseline Runtime Mean (s)":
                baseline_runtime_mean,

            "Baseline Runtime Std (s)":
                baseline_runtime_std,

            "Proposed Runtime Mean (s)":
                proposed_runtime_mean,

            "Proposed Runtime Std (s)":
                proposed_runtime_std,

            "Baseline Iterations Mean":
                baseline_iterations_mean,

            "Baseline Iterations Std":
                baseline_iterations_std,

            "Proposed Iterations Mean":
                proposed_iterations_mean,

            "Proposed Iterations Std":
                proposed_iterations_std,

            "Iteration Reduction (%)":
                iteration_reduction,

            "Proposed Kernel Mean":
                kernel_mean,

            "Proposed Kernel Std":
                kernel_std
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        TABLES_DIR,
        "repeated_noise_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("FINAL REPEATED-TRIAL SUMMARY")
    print("=" * 90)

    display_columns = [
        "Perturbation (%)",
        "Baseline Accuracy Mean (%)",
        "Baseline Accuracy Std (%)",
        "Proposed Accuracy Mean (%)",
        "Proposed Accuracy Std (%)",
        "Accuracy Improvement (pp)",
        "Baseline Iterations Mean",
        "Proposed Iterations Mean",
        "Iteration Reduction (%)"
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Raw results : {raw_path}"
    )

    print(
        f"Summary     : {summary_path}"
    )

    print("\nExperiment completed.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()