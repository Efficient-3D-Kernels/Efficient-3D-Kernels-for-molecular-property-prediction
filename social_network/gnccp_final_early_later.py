"""
GNCCP FINAL — EARLY vs LATER COLLEGMESSAGE EXPERIMENT

Purpose
-------
Run the Enhanced GNCCP method on TWO REAL, TEMPORALLY DISTINCT
CollegeMsg social-network graphs:

    G = early half of the CollegeMsg interactions
    H = later half of the CollegeMsg interactions

Unlike the perturbation experiment, H is NOT constructed as a
perturbed copy of G.

Important:
- There is no ground-truth correspondence between the two temporal
  snapshots, so this experiment does NOT report matching accuracy.
- It reports objective, runtime, optimization iterations, kernel,
  and the number of returned matches.
- The matching itself is also saved as a CSV so the discovered
  correspondences can be inspected.

This script uses the project's existing Enhanced GNCCP implementation:
    src.proposed_gnccp_enhanced.run_enhanced_gnccp

Do NOT replace the optimizer with a standalone reimplementation here.
"""

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
from src.proposed_gnccp_enhanced import run_enhanced_gnccp


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

USER_SIZES = [100, 400]

ALPHA = 0.70

ZETA_STEP = 0.10
FW_ITERATIONS = 50
GAP_TOLERANCE = 1e-6
GAMMA = 1.0

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ============================================================
# MATCHING SIZE
# ============================================================

def get_matching_size(num_users):
    """
    Use a 75% partial matching for each graph size.

    100 users -> 75 matches
    400 users -> 300 matches
    """
    return max(1, int(round(0.75 * num_users)))


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(interactions):
    """
    Split CollegeMsg chronologically into two equal halves.
    """
    interactions = interactions.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    midpoint = len(interactions) // 2

    early_data = interactions.iloc[:midpoint].copy()
    later_data = interactions.iloc[midpoint:].copy()

    return early_data, later_data


# ============================================================
# RESULT EXTRACTION
# ============================================================

def unpack_enhanced_result(result):
    """
    The project's current Enhanced GNCCP wrapper returns:

        {
            "method": ...,
            "gnccp_result": (
                X_discrete,
                objective,
                iterations,
                kernel
            ),
            ...
        }

    This function extracts the actual GNCCP outputs.
    """

    if not isinstance(result, dict):
        raise TypeError(
            "Enhanced GNCCP did not return the expected dictionary."
        )

    if "gnccp_result" not in result:
        raise KeyError(
            "Enhanced GNCCP result does not contain 'gnccp_result'."
        )

    gnccp_result = result["gnccp_result"]

    if not isinstance(gnccp_result, tuple):
        raise TypeError(
            "Expected 'gnccp_result' to be a tuple."
        )

    if len(gnccp_result) != 4:
        raise ValueError(
            f"Unexpected GNCCP result length: {len(gnccp_result)}"
        )

    X = gnccp_result[0]
    objective = float(gnccp_result[1])
    iterations = int(gnccp_result[2])
    kernel = float(gnccp_result[3])

    return X, objective, iterations, kernel


# ============================================================
# SAVE MATCHING
# ============================================================

def save_matching(
    X,
    node_order,
    output_file
):
    """
    Save the discrete node-to-node matching.

    Since G and H are temporal snapshots with no ground truth,
    these are discovered correspondences, NOT accuracy-labelled
    matches.
    """

    X = np.asarray(X)

    selected_pairs = np.argwhere(X > 0.5)

    rows = []

    for i, j in selected_pairs:
        if i >= len(node_order) or j >= len(node_order):
            continue

        rows.append({
            "G_Node": int(node_order[i]),
            "H_Node": int(node_order[j]),
            "Match_Value": float(X[i, j])
        })

    matching_df = pd.DataFrame(
        rows,
        columns=[
            "G_Node",
            "H_Node",
            "Match_Value"
        ]
    )

    matching_df.to_csv(
        output_file,
        index=False
    )

    return matching_df


# ============================================================
# RUN ONE TEMPORAL EXPERIMENT
# ============================================================

def run_one_experiment(
    G,
    H,
    node_order,
    matching_size,
    num_users
):
    """
    Run Enhanced GNCCP on the real early/later graph pair.
    """

    AG = nx.to_numpy_array(
        G,
        nodelist=node_order,
        weight="weight",
        dtype=float
    )

    AH = nx.to_numpy_array(
        H,
        nodelist=node_order,
        weight="weight",
        dtype=float
    )

    print("\nBuilding/running Enhanced GNCCP...")
    print(f"Users:          {num_users}")
    print(f"Matching size:  {matching_size}")
    print(f"Alpha:          {ALPHA}")

    start_time = time.perf_counter()

    result = run_enhanced_gnccp(
        AG,
        AH,
        G,
        H,
        matching_size=matching_size,
        alpha=ALPHA,
        zeta_step=ZETA_STEP,
        fw_iterations=FW_ITERATIONS,
        gap_tolerance=GAP_TOLERANCE,
        gamma=GAMMA
    )

    runtime = time.perf_counter() - start_time

    X, objective, iterations, kernel = unpack_enhanced_result(
        result
    )

    actual_matches = int(
        np.sum(np.asarray(X) > 0.5)
    )

    return (
        X,
        objective,
        iterations,
        kernel,
        runtime,
        actual_matches
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    print("=" * 80)
    print("ENHANCED GNCCP — EARLY vs LATER COLLEGMESSAGE")
    print("=" * 80)

    print(f"User sizes:      {USER_SIZES}")
    print(f"Alpha:           {ALPHA}")
    print(f"Matching ratio:  75%")
    print(f"Random seed:     {RANDOM_SEED}")
    print(f"Zeta step:       {ZETA_STEP}")
    print(f"FW iterations:   {FW_ITERATIONS}")
    print(f"Gap tolerance:   {GAP_TOLERANCE}")
    print(f"Gamma:           {GAMMA}")

    print("\nIMPORTANT:")
    print(
        "H is the REAL LATER TEMPORAL GRAPH, not a perturbed copy of G."
    )
    print(
        "No matching accuracy is reported because ground truth is unavailable."
    )

    # --------------------------------------------------------
    # Load CollegeMsg
    # --------------------------------------------------------

    print("\nLoading CollegeMsg dataset...")

    interactions = load_college_msg()

    print(
        f"Total interactions: {len(interactions):,}"
    )

    # --------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------

    early_data, later_data = temporal_split(
        interactions
    )

    print(
        f"Early interactions: {len(early_data):,}"
    )

    print(
        f"Later interactions: {len(later_data):,}"
    )

    results = []

    # ========================================================
    # GRAPH SIZE LOOP
    # ========================================================

    for num_users in USER_SIZES:

        print("\n\n")
        print("=" * 80)
        print(f"GRAPH SIZE: {num_users} USERS")
        print("=" * 80)

        matching_size = get_matching_size(
            num_users
        )

        print(
            f"Matching size L: {matching_size}"
        )

        # ----------------------------------------------------
        # Build REAL temporal graphs
        # ----------------------------------------------------

        G, H, common_nodes = build_real_social_graphs(
            early_data,
            later_data,
            num_users=num_users
        )

        node_order = sorted(
            common_nodes
        )

        G = G.subgraph(
            node_order
        ).copy()

        H = H.subgraph(
            node_order
        ).copy()

        print("\nEARLY SOCIAL GRAPH G")
        print("-" * 50)
        print(
            f"Nodes:             {G.number_of_nodes()}"
        )
        print(
            f"Edges:             {G.number_of_edges()}"
        )
        print(
            f"Total edge weight: "
            f"{sum(data.get('weight', 1.0) for _, _, data in G.edges(data=True)):.2f}"
        )
        print(
            f"Average degree:    "
            f"{np.mean([d for _, d in G.degree()]):.2f}"
        )

        print("\nLATER SOCIAL GRAPH H")
        print("-" * 50)
        print(
            f"Nodes:             {H.number_of_nodes()}"
        )
        print(
            f"Edges:             {H.number_of_edges()}"
        )
        print(
            f"Total edge weight: "
            f"{sum(data.get('weight', 1.0) for _, _, data in H.edges(data=True)):.2f}"
        )
        print(
            f"Average degree:    "
            f"{np.mean([d for _, d in H.degree()]):.2f}"
        )

        print(
            f"\nCommon users available: {len(common_nodes)}"
        )

        # ----------------------------------------------------
        # Run Enhanced GNCCP
        # ----------------------------------------------------

        (
            X,
            objective,
            iterations,
            kernel,
            runtime,
            actual_matches
        ) = run_one_experiment(
            G,
            H,
            node_order,
            matching_size,
            num_users
        )

        # ----------------------------------------------------
        # Save discovered matching
        # ----------------------------------------------------

        results_dir = os.path.join(
            PROJECT_ROOT,
            "results",
            "tables"
        )

        matching_dir = os.path.join(
            PROJECT_ROOT,
            "results",
            "matchings"
        )

        os.makedirs(
            results_dir,
            exist_ok=True
        )

        os.makedirs(
            matching_dir,
            exist_ok=True
        )

        matching_file = os.path.join(
            matching_dir,
            f"enhanced_gnccp_early_later_{num_users}_matches.csv"
        )

        matching_df = save_matching(
            X,
            node_order,
            matching_file
        )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print("\nRESULT")
        print("-" * 60)

        print(
            f"Users:              {num_users}"
        )

        print(
            f"Matching size:      {matching_size}"
        )

        print(
            f"Actual matches:     {actual_matches}"
        )

        print(
            f"Alpha:              {ALPHA}"
        )

        print(
            f"Objective:           {objective:.10f}"
        )

        print(
            f"Iterations:          {iterations}"
        )

        print(
            f"Runtime:             {runtime:.4f} s"
        )

        print(
            f"Kernel:              {kernel:.10f}"
        )

        print(
            "Accuracy:            N/A (no ground truth)"
        )

        print(
            f"Matching saved to:   {matching_file}"
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({
            "Users": num_users,
            "Matching Size": matching_size,
            "Alpha": ALPHA,
            "Method": "Enhanced GNCCP",
            "Graph Pair": "Early vs Later CollegeMsg",
            "Accuracy (%)": np.nan,
            "Objective": objective,
            "Runtime (s)": runtime,
            "Iterations": iterations,
            "Kernel": kernel,
            "Actual Matches": actual_matches,
            "Ground Truth": "Unavailable"
        })

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print("\n\n")
    print("=" * 80)
    print("FINAL EARLY vs LATER RESULTS")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save final table
    # --------------------------------------------------------

    output_file = os.path.join(
        PROJECT_ROOT,
        "results",
        "tables",
        "enhanced_gnccp_early_later_100_400_alpha07_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nResults saved to:")
    print(output_file)

    print("\nExperiment completed successfully.")

    return results_df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_experiment()
