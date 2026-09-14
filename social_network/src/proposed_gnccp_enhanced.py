"""
Enhanced GNCCP for Social Network Graph Matching.

V2 enhancement:
    - Uses multi-scale structural node features
    - Uses an enhanced node-to-node cost matrix
    - Passes the enhanced cost matrix to the existing GNCCP optimizer

The underlying GNCCP optimization procedure is unchanged.

Enhanced features:
    1. Degree
    2. Weighted degree
    3. Clustering coefficient
    4. Average neighbor degree
    5. Average neighbor weighted degree
    6. Average 2-hop degree
    7. Average 2-hop weighted degree
    8. PageRank

This file is intentionally a wrapper around the existing
proposed_gnccp implementation so that V1 and V2 can be
compared fairly.
"""

import os
import sys
import time
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(
        0,
        CURRENT_DIR
    )


# ============================================================
# IMPORT EXISTING GNCCP
# ============================================================

from proposed_gnccp import run_proposed_gnccp

from enhanced_features import (
    build_enhanced_cost_matrix
)


# ============================================================
# ENHANCED GNCCP
# ============================================================

def run_enhanced_gnccp(
    AG,
    AH,
    G,
    H,
    matching_size,
    alpha=0.5,
    zeta_step=0.1,
    fw_iterations=50,
    gap_tolerance=1e-6,
    gamma=1.0
):
    """
    Run the enhanced GNCCP approach.

    Parameters
    ----------
    AG : np.ndarray
        Adjacency matrix of graph G.

    AH : np.ndarray
        Adjacency matrix of graph H.

    G : networkx.Graph
        Original weighted graph G.

    H : networkx.Graph
        Original weighted graph H.

    matching_size : int
        Number of nodes to match.

    alpha : float
        Weight between structural graph cost and node
        feature cost in the existing GNCCP formulation.

    zeta_step : float
        GNCCP continuation step.

    fw_iterations : int
        Maximum Frank-Wolfe iterations.

    gap_tolerance : float
        Frank-Wolfe convergence tolerance.

    gamma : float
        Similarity-kernel parameter.

    Returns
    -------
    result : dict
        Result returned by the existing GNCCP together with
        information about the enhanced feature representation.
    """

    # --------------------------------------------------------
    # Build enhanced cost matrix
    # --------------------------------------------------------

    (
        C_enhanced,
        nodes_G,
        nodes_H,
        features_G,
        features_H
    ) = build_enhanced_cost_matrix(
        G,
        H
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if C_enhanced.shape != (
        len(nodes_G),
        len(nodes_H)
    ):
        raise ValueError(
            "Enhanced cost matrix dimensions do not "
            "match graph dimensions."
        )

    # --------------------------------------------------------
    # Run the EXISTING proposed GNCCP optimizer
    #
    # Important:
    # We are not changing the GNCCP optimization itself.
    # Only the node-feature cost matrix is enhanced.
    # --------------------------------------------------------

    start_time = time.perf_counter()

    result = run_proposed_gnccp(
        AG,
        AH,
        C_enhanced,
        matching_size=matching_size,
        alpha=alpha,
        zeta_step=zeta_step,
        fw_iterations=fw_iterations,
        gap_tolerance=gap_tolerance,
        gamma=gamma
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Attach metadata
    # --------------------------------------------------------

    if isinstance(result, dict):

        result["method"] = (
            "Enhanced GNCCP"
        )

        result["feature_type"] = (
            "Multi-scale structural features"
        )

        result["num_features"] = (
            features_G.shape[1]
            if features_G.ndim == 2
            else 0
        )

        result["feature_names"] = [
            "degree",
            "weighted_degree",
            "clustering",
            "avg_neighbor_degree",
            "avg_neighbor_weighted_degree",
            "avg_2hop_degree",
            "avg_2hop_weighted_degree",
            "pagerank"
        ]

        result["enhanced_runtime"] = (
            elapsed
        )

        result["enhanced_cost_matrix"] = (
            C_enhanced
        )

        result["feature_nodes_G"] = (
            nodes_G
        )

        result["feature_nodes_H"] = (
            nodes_H
        )

        result["enhanced_features_G"] = (
            features_G
        )

        result["enhanced_features_H"] = (
            features_H
        )

        return result

    # --------------------------------------------------------
    # If the existing implementation returns a tuple/list,
    # preserve it and return metadata separately.
    # --------------------------------------------------------

    return {
        "method": "Enhanced GNCCP",
        "gnccp_result": result,
        "feature_type": (
            "Multi-scale structural features"
        ),
        "num_features": features_G.shape[1],
        "feature_names": [
            "degree",
            "weighted_degree",
            "clustering",
            "avg_neighbor_degree",
            "avg_neighbor_weighted_degree",
            "avg_2hop_degree",
            "avg_2hop_weighted_degree",
            "pagerank"
        ],
        "enhanced_runtime": elapsed,
        "enhanced_cost_matrix": C_enhanced,
        "feature_nodes_G": nodes_G,
        "feature_nodes_H": nodes_H,
        "enhanced_features_G": features_G,
        "enhanced_features_H": features_H
    }


# ============================================================
# SANITY TEST
# ============================================================

if __name__ == "__main__":

    import networkx as nx

    print("=" * 70)
    print("ENHANCED GNCCP SANITY TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Small weighted graph G
    # --------------------------------------------------------

    G = nx.Graph()

    G.add_edge(
        0,
        1,
        weight=3.0
    )

    G.add_edge(
        1,
        2,
        weight=2.0
    )

    G.add_edge(
        2,
        3,
        weight=1.0
    )

    G.add_edge(
        0,
        3,
        weight=2.0
    )

    G.add_edge(
        1,
        3,
        weight=4.0
    )

    # --------------------------------------------------------
    # Same graph with different node IDs
    # --------------------------------------------------------

    H = nx.Graph()

    H.add_edge(
        10,
        11,
        weight=3.0
    )

    H.add_edge(
        11,
        12,
        weight=2.0
    )

    H.add_edge(
        12,
        13,
        weight=1.0
    )

    H.add_edge(
        10,
        13,
        weight=2.0
    )

    H.add_edge(
        11,
        13,
        weight=4.0
    )

    # --------------------------------------------------------
    # Convert adjacency matrices
    # --------------------------------------------------------

    nodes_G = list(G.nodes())
    nodes_H = list(H.nodes())

    AG = nx.to_numpy_array(
        G,
        nodelist=nodes_G,
        weight="weight"
    )

    AH = nx.to_numpy_array(
        H,
        nodelist=nodes_H,
        weight="weight"
    )

    # --------------------------------------------------------
    # Run enhanced GNCCP
    # --------------------------------------------------------

    try:

        result = run_enhanced_gnccp(
            AG,
            AH,
            G,
            H,
            matching_size=3
        )

        print()
        print(
            "Enhanced GNCCP executed successfully."
        )

        print(
            "Number of enhanced features:",
            result.get(
                "num_features",
                "unknown"
            )
        )

        print(
            "Feature type:",
            result.get(
                "feature_type",
                "unknown"
            )
        )

        print()
        print("Features:")
        for name in result.get(
            "feature_names",
            []
        ):
            print(
                "  -",
                name
            )

        print()
        print(
            "Test completed successfully."
        )

    except Exception as exc:

        print()
        print(
            "Enhanced GNCCP test failed."
        )

        print(
            "Error:",
            exc
        )

        raise