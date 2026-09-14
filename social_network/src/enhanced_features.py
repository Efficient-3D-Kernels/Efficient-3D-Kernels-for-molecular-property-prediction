"""
Enhanced node features for GNCCP graph matching.

Adds multi-scale structural information to the existing
degree / weighted-degree / clustering features.

Features per node:
    1. degree
    2. weighted degree
    3. clustering coefficient
    4. average neighbor degree
    5. average neighbor weighted degree
    6. average 2-hop degree
    7. average 2-hop weighted degree
    8. PageRank

The features are jointly normalized before constructing
the node-to-node cost matrix.
"""

import numpy as np
import networkx as nx


# ============================================================
# NODE FEATURE EXTRACTION
# ============================================================

def _safe_mean(values):
    """Return mean(values), or 0 when values is empty."""
    if not values:
        return 0.0
    return float(np.mean(values))


def _two_hop_nodes(G, node):
    """
    Return nodes that are exactly two hops away from `node`.
    """
    neighbors = set(G.neighbors(node))
    two_hop = set()

    for nbr in neighbors:
        for nbr2 in G.neighbors(nbr):
            if nbr2 != node and nbr2 not in neighbors:
                two_hop.add(nbr2)

    return two_hop


def extract_enhanced_node_features(G):
    """
    Extract an enhanced structural feature vector for every node.

    Returns
    -------
    nodes : list
        Node ordering.

    features : np.ndarray
        Shape (n_nodes, 8).

    Feature order:
        0 = degree
        1 = weighted degree
        2 = clustering
        3 = average neighbor degree
        4 = average neighbor weighted degree
        5 = average 2-hop degree
        6 = average 2-hop weighted degree
        7 = PageRank
    """

    nodes = list(G.nodes())

    if len(nodes) == 0:
        return nodes, np.empty((0, 8), dtype=float)

    # --------------------------------------------------------
    # PageRank
    # --------------------------------------------------------

    try:
        pagerank = nx.pagerank(
            G,
            weight="weight"
        )
    except Exception:
        pagerank = {
            node: 1.0 / len(nodes)
            for node in nodes
        }

    # --------------------------------------------------------
    # Clustering
    # --------------------------------------------------------

    clustering = nx.clustering(
        G,
        weight="weight"
    )

    features = []

    for node in nodes:

        # ----------------------------------------------------
        # 1. Degree
        # ----------------------------------------------------

        degree = float(
            G.degree(node)
        )

        # ----------------------------------------------------
        # 2. Weighted degree
        # ----------------------------------------------------

        weighted_degree = float(
            G.degree(
                node,
                weight="weight"
            )
        )

        # ----------------------------------------------------
        # 3. Clustering coefficient
        # ----------------------------------------------------

        cluster = float(
            clustering.get(node, 0.0)
        )

        # ----------------------------------------------------
        # 4 & 5. One-hop neighborhood statistics
        # ----------------------------------------------------

        neighbors = list(
            G.neighbors(node)
        )

        neighbor_degrees = [
            float(G.degree(nbr))
            for nbr in neighbors
        ]

        neighbor_weighted_degrees = [
            float(
                G.degree(
                    nbr,
                    weight="weight"
                )
            )
            for nbr in neighbors
        ]

        avg_neighbor_degree = _safe_mean(
            neighbor_degrees
        )

        avg_neighbor_weighted_degree = _safe_mean(
            neighbor_weighted_degrees
        )

        # ----------------------------------------------------
        # 6 & 7. Two-hop neighborhood statistics
        # ----------------------------------------------------

        two_hop = _two_hop_nodes(
            G,
            node
        )

        two_hop_degrees = [
            float(G.degree(nbr))
            for nbr in two_hop
        ]

        two_hop_weighted_degrees = [
            float(
                G.degree(
                    nbr,
                    weight="weight"
                )
            )
            for nbr in two_hop
        ]

        avg_two_hop_degree = _safe_mean(
            two_hop_degrees
        )

        avg_two_hop_weighted_degree = _safe_mean(
            two_hop_weighted_degrees
        )

        # ----------------------------------------------------
        # 8. PageRank
        # ----------------------------------------------------

        pr = float(
            pagerank.get(
                node,
                0.0
            )
        )

        features.append([
            degree,
            weighted_degree,
            cluster,
            avg_neighbor_degree,
            avg_neighbor_weighted_degree,
            avg_two_hop_degree,
            avg_two_hop_weighted_degree,
            pr
        ])

    return (
        nodes,
        np.asarray(
            features,
            dtype=float
        )
    )


# ============================================================
# ROBUST FEATURE NORMALIZATION
# ============================================================

def robust_normalize_features(
    features_G,
    features_H
):
    """
    Jointly normalize features of G and H.

    Uses min-max normalization.

    A small epsilon prevents division by zero for
    constant features.
    """

    combined = np.vstack([
        features_G,
        features_H
    ])

    minimum = np.min(
        combined,
        axis=0
    )

    maximum = np.max(
        combined,
        axis=0
    )

    scale = maximum - minimum

    scale[
        scale < 1e-12
    ] = 1.0

    normalized = (
        combined - minimum
    ) / scale

    n_G = len(features_G)

    return (
        normalized[:n_G],
        normalized[n_G:]
    )


# ============================================================
# ENHANCED COST MATRIX
# ============================================================

def build_enhanced_cost_matrix(
    G,
    H
):
    """
    Construct a node-to-node cost matrix using enhanced
    multi-scale structural features.

    Lower cost means the two nodes are more structurally
    similar.

    Returns
    -------
    C : np.ndarray
        Shape (n_G, n_H).

    nodes_G : list
        Node ordering for G.

    nodes_H : list
        Node ordering for H.

    features_G : np.ndarray
        Normalized features for G.

    features_H : np.ndarray
        Normalized features for H.
    """

    nodes_G, features_G = (
        extract_enhanced_node_features(G)
    )

    nodes_H, features_H = (
        extract_enhanced_node_features(H)
    )

    if len(nodes_G) == 0 or len(nodes_H) == 0:
        return (
            np.empty(
                (len(nodes_G), len(nodes_H))
            ),
            nodes_G,
            nodes_H,
            features_G,
            features_H
        )

    # --------------------------------------------------------
    # Joint normalization
    # --------------------------------------------------------

    features_G, features_H = (
        robust_normalize_features(
            features_G,
            features_H
        )
    )

    # --------------------------------------------------------
    # Feature weights
    #
    # We do not let the large-scale numerical features
    # dominate the smaller features.
    # --------------------------------------------------------

    feature_weights = np.array([
        1.0,   # degree
        1.25,  # weighted degree
        1.0,   # clustering
        1.0,   # neighbor degree
        1.25,  # neighbor weighted degree
        0.75,  # 2-hop degree
        1.0,   # 2-hop weighted degree
        0.75   # PageRank
    ])

    feature_weights = (
        feature_weights
        / np.sum(feature_weights)
    )

    # --------------------------------------------------------
    # Weighted Euclidean cost
    # --------------------------------------------------------

    diff = (
        features_G[:, None, :]
        -
        features_H[None, :, :]
    )

    squared = diff ** 2

    weighted_squared = (
        squared
        * feature_weights[None, None, :]
    )

    C = np.sqrt(
        np.sum(
            weighted_squared,
            axis=2
        )
    )

    # --------------------------------------------------------
    # Normalize cost to [0, 1]
    # --------------------------------------------------------

    c_min = np.min(C)
    c_max = np.max(C)

    if c_max - c_min > 1e-12:
        C = (
            C - c_min
        ) / (
            c_max - c_min
        )
    else:
        C = np.zeros_like(C)

    return (
        C,
        nodes_G,
        nodes_H,
        features_G,
        features_H
    )


# ============================================================
# FEATURE-GUIDED INITIALIZATION
# ============================================================

def build_feature_guided_initialization(
    C,
    matching_size
):
    """
    Build a partial matching initialization from the
    enhanced feature cost matrix.

    The lowest-cost node pairs are selected greedily
    while enforcing one-to-one correspondence.

    Parameters
    ----------
    C : np.ndarray
        Node-to-node cost matrix.

    matching_size : int
        Number of nodes to match.

    Returns
    -------
    X0 : np.ndarray
        Partial permutation initialization.
    """

    n_G, n_H = C.shape

    L = min(
        int(matching_size),
        n_G,
        n_H
    )

    X0 = np.zeros(
        (n_G, n_H),
        dtype=float
    )

    # All possible node pairs sorted by cost.
    pairs = []

    for i in range(n_G):
        for j in range(n_H):
            pairs.append(
                (
                    float(C[i, j]),
                    i,
                    j
                )
            )

    pairs.sort(
        key=lambda x: x[0]
    )

    used_G = set()
    used_H = set()

    selected = 0

    for cost, i, j in pairs:

        if i in used_G:
            continue

        if j in used_H:
            continue

        X0[i, j] = 1.0

        used_G.add(i)
        used_H.add(j)

        selected += 1

        if selected >= L:
            break

    return X0


# ============================================================
# SANITY TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ENHANCED GNCCP FEATURE TEST")
    print("=" * 70)

    G = nx.Graph()

    G.add_edge(0, 1, weight=3.0)
    G.add_edge(1, 2, weight=2.0)
    G.add_edge(2, 3, weight=1.0)
    G.add_edge(0, 3, weight=2.0)
    G.add_edge(1, 3, weight=4.0)

    H = nx.Graph()

    H.add_edge(10, 11, weight=3.0)
    H.add_edge(11, 12, weight=2.0)
    H.add_edge(12, 13, weight=1.0)
    H.add_edge(10, 13, weight=2.0)
    H.add_edge(11, 13, weight=4.0)

    C, nodes_G, nodes_H, features_G, features_H = (
        build_enhanced_cost_matrix(
            G,
            H
        )
    )

    print()
    print("Nodes G:", nodes_G)
    print("Nodes H:", nodes_H)

    print()
    print(
        "Feature matrix shape G:",
        features_G.shape
    )

    print(
        "Feature matrix shape H:",
        features_H.shape
    )

    print()
    print("Cost matrix:")
    print(
        np.round(
            C,
            4
        )
    )

    X0 = build_feature_guided_initialization(
        C,
        matching_size=3
    )

    print()
    print("Feature-guided initialization:")
    print(X0)

    print()
    print(
        "Selected pairs:",
        np.argwhere(X0 > 0.5).tolist()
    )

    print()
    print("Test completed successfully.")