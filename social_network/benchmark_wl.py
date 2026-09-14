"""
Benchmark the Weisfeiler-Lehman (WL) kernel
on the existing CollegeMsg social-network experiment.
"""

import sys
import os
import time
import copy

import networkx as nx
import pandas as pd

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_college_msg
from graph_builder import (
    build_real_social_graphs,
    build_weighted_social_graph,
)
from kernels.wl_kernel import run_wl_kernel


# ============================================================
# SETTINGS
# ============================================================

NUM_USERS = 40
WL_ITERATIONS = 3

PERTURBATION_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]

RANDOM_SEED = 42


# ============================================================
# PERTURBATION
# ============================================================

def perturb_graph(G, perturbation, seed=42):
    """
    Create a perturbed copy of G.

    Two types of perturbation are applied:
        1. Edge-weight modification
        2. Edge removal

    Node identities are preserved so that the original graph
    correspondence remains the ground truth.
    """

    H = copy.deepcopy(G)

    rng = __import__("random").Random(seed)

    edges = list(H.edges())

    if len(edges) == 0:
        return H

    number_to_modify = int(len(edges) * perturbation)

    if number_to_modify == 0:
        return H

    selected_edges = rng.sample(
        edges,
        min(number_to_modify, len(edges))
    )

    # --------------------------------------------------------
    # Modify weights
    # --------------------------------------------------------

    for u, v in selected_edges:

        old_weight = float(
            H[u][v].get("weight", 1.0)
        )

        # Reduce the edge weight by a random amount.
        reduction = rng.uniform(0.10, 0.50)

        new_weight = max(
            0.1,
            old_weight * (1.0 - reduction)
        )

        H[u][v]["weight"] = new_weight

    # --------------------------------------------------------
    # Remove a subset of edges
    # --------------------------------------------------------

    number_to_remove = int(
        len(edges) * perturbation * 0.25
    )

    if number_to_remove > 0:

        removable = [
            edge for edge in selected_edges
            if H.has_edge(edge[0], edge[1])
        ]

        remove_edges = rng.sample(
            removable,
            min(number_to_remove, len(removable))
        )

        H.remove_edges_from(remove_edges)

    return H


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("WEISFEILER-LEHMAN KERNEL — COLLEGEMSG BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------------
    # Load CollegeMsg
    # --------------------------------------------------------

    print("\nLoading CollegeMsg dataset...")

    interactions = load_college_msg()

    print(f"Total interactions: {len(interactions):,}")

    # --------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------

    interactions = interactions.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    midpoint = len(interactions) // 2

    early_data = interactions.iloc[:midpoint].copy()
    later_data = interactions.iloc[midpoint:].copy()

    print(f"Early interactions: {len(early_data):,}")
    print(f"Later interactions: {len(later_data):,}")

    # --------------------------------------------------------
    # Build the same 40-user graph used in your GNCCP
    # experiment.
    # --------------------------------------------------------

    print("\nBuilding CollegeMsg graphs...")

    G, H_real, common_nodes = build_real_social_graphs(
        early_data,
        later_data,
        num_users=NUM_USERS
    )

    print(f"Selected users: {len(common_nodes)}")
    print(f"G nodes: {G.number_of_nodes()}")
    print(f"G edges: {G.number_of_edges()}")
    print(f"H nodes: {H_real.number_of_nodes()}")
    print(f"H edges: {H_real.number_of_edges()}")

    # --------------------------------------------------------
    # Ensure both graphs use the same node set.
    # --------------------------------------------------------

    common_nodes = list(common_nodes)

    G = G.subgraph(common_nodes).copy()
    H_real = H_real.subgraph(common_nodes).copy()

    # --------------------------------------------------------
    # Run perturbation experiment
    # --------------------------------------------------------

    results = []

    print("\n" + "-" * 70)
    print("PERTURBATION EXPERIMENT")
    print("-" * 70)

    for perturbation in PERTURBATION_LEVELS:

        print(
            f"\nPerturbation: "
            f"{perturbation * 100:.0f}%"
        )

        H = perturb_graph(
            G,
            perturbation,
            seed=RANDOM_SEED
        )

        # ----------------------------------------------------
        # WL kernel
        # ----------------------------------------------------

        start_time = time.perf_counter()

        result = run_wl_kernel(
            G,
            H,
            h=WL_ITERATIONS
        )

        end_time = time.perf_counter()

        runtime = end_time - start_time

        similarity = result["similarity"]

        print(
            f"WL similarity : {similarity:.6f}"
        )

        print(
            f"Runtime       : {runtime:.6f} sec"
        )

        results.append({
            "Method": "WL",
            "Perturbation": perturbation * 100,
            "Similarity": similarity,
            "Runtime (s)": runtime,
            "WL Iterations": WL_ITERATIONS,
            "Nodes G": G.number_of_nodes(),
            "Edges G": G.number_of_edges(),
            "Nodes H": H.number_of_nodes(),
            "Edges H": H.number_of_edges(),
        })

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    output_dir = os.path.join(
        "results",
        "tables"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        "wl_collegemsg_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("WL BENCHMARK COMPLETE")
    print("=" * 70)

    print("\nResults:")
    print(results_df.to_string(index=False))

    print(
        f"\nSaved to: {output_file}"
    )


if __name__ == "__main__":
    main()