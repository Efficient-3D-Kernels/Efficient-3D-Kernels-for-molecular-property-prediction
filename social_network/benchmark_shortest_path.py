"""
Benchmark the Shortest-Path Graph Kernel
on the existing CollegeMsg social-network experiment.
"""

import sys
import os
import time
import copy
import random

import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from data_loader import load_college_msg

from graph_builder import (
    build_real_social_graphs,
)

from kernels.shortest_path_kernel import (
    run_shortest_path_kernel,
)


# ============================================================
# SETTINGS
# ============================================================

NUM_USERS = 40

PERTURBATION_LEVELS = [
    0.0,
    0.05,
    0.10,
    0.20,
    0.30,
]

RANDOM_SEED = 42


# ============================================================
# GRAPH PERTURBATION
# ============================================================

def perturb_graph(G, perturbation, seed=42):
    """
    Create a perturbed copy of G.

    Perturbation consists of:
        1. Edge-weight modifications
        2. Edge removals

    Node identities are preserved.
    """

    H = copy.deepcopy(G)

    rng = random.Random(seed)

    edges = list(H.edges())

    if len(edges) == 0:
        return H

    # --------------------------------------------------------
    # Number of edges affected
    # --------------------------------------------------------

    number_to_modify = int(
        len(edges) * perturbation
    )

    if number_to_modify == 0:
        return H

    selected_edges = rng.sample(
        edges,
        min(number_to_modify, len(edges))
    )

    # --------------------------------------------------------
    # Modify edge weights
    # --------------------------------------------------------

    for u, v in selected_edges:

        old_weight = float(
            H[u][v].get("weight", 1.0)
        )

        reduction = rng.uniform(
            0.10,
            0.50
        )

        new_weight = max(
            0.1,
            old_weight * (1.0 - reduction)
        )

        H[u][v]["weight"] = new_weight

    # --------------------------------------------------------
    # Remove a subset of selected edges
    # --------------------------------------------------------

    number_to_remove = int(
        len(edges) * perturbation * 0.25
    )

    if number_to_remove > 0:

        removable_edges = [
            edge
            for edge in selected_edges
            if H.has_edge(
                edge[0],
                edge[1]
            )
        ]

        remove_edges = rng.sample(
            removable_edges,
            min(
                number_to_remove,
                len(removable_edges)
            )
        )

        H.remove_edges_from(
            remove_edges
        )

    return H


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "SHORTEST-PATH KERNEL — "
        "COLLEGEMSG BENCHMARK"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\nLoading CollegeMsg dataset...")

    interactions = load_college_msg()

    print(
        f"Total interactions: "
        f"{len(interactions):,}"
    )

    # --------------------------------------------------------
    # TEMPORAL SPLIT
    # --------------------------------------------------------

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

    print(
        f"Early interactions: "
        f"{len(early_data):,}"
    )

    print(
        f"Later interactions: "
        f"{len(later_data):,}"
    )

    # --------------------------------------------------------
    # BUILD SAME GRAPHS USED BY GNCCP
    # --------------------------------------------------------

    print("\nBuilding CollegeMsg graphs...")

    G, H_real, common_nodes = (
        build_real_social_graphs(
            early_data,
            later_data,
            num_users=NUM_USERS
        )
    )

    print(
        f"Selected users: "
        f"{len(common_nodes)}"
    )

    print(
        f"G nodes: "
        f"{G.number_of_nodes()}"
    )

    print(
        f"G edges: "
        f"{G.number_of_edges()}"
    )

    print(
        f"H nodes: "
        f"{H_real.number_of_nodes()}"
    )

    print(
        f"H edges: "
        f"{H_real.number_of_edges()}"
    )

    # --------------------------------------------------------
    # FORCE SAME NODE SET
    # --------------------------------------------------------

    common_nodes = list(common_nodes)

    G = G.subgraph(
        common_nodes
    ).copy()

    H_real = H_real.subgraph(
        common_nodes
    ).copy()

    # --------------------------------------------------------
    # RUN EXPERIMENT
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

        # ----------------------------------------------------
        # Create perturbed graph
        # ----------------------------------------------------

        H = perturb_graph(
            G,
            perturbation,
            seed=RANDOM_SEED
        )

        # ----------------------------------------------------
        # Run Shortest-Path kernel
        # ----------------------------------------------------

        start_time = time.perf_counter()

        result = run_shortest_path_kernel(
            G,
            H
        )

        end_time = time.perf_counter()

        runtime = (
            end_time -
            start_time
        )

        similarity = result[
            "similarity"
        ]

        print(
            f"SP similarity : "
            f"{similarity:.6f}"
        )

        print(
            f"Runtime       : "
            f"{runtime:.6f} sec"
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "Method":
                "Shortest Path",

            "Perturbation":
                perturbation * 100,

            "Similarity":
                similarity,

            "Runtime (s)":
                runtime,

            "Nodes G":
                G.number_of_nodes(),

            "Edges G":
                G.number_of_edges(),

            "Nodes H":
                H.number_of_nodes(),

            "Edges H":
                H.number_of_edges(),

        })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    output_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        "tables"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        "shortest_path_collegemsg_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "SHORTEST-PATH BENCHMARK COMPLETE"
    )
    print("=" * 70)

    print("\nResults:")

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: "
        f"{output_file}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()