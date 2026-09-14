import numpy as np
import networkx as nx


def compute_node_features(G):
    """
    Compute real social-network features for every node.

    Features:
        1. Degree
        2. Weighted degree (total message volume)
        3. Clustering coefficient

    Returns
    -------
    features : ndarray
        Shape = (number_of_nodes, 3)
    nodes : list
        Node IDs corresponding to feature rows.
    """

    nodes = sorted(G.nodes())

    # 1. Number of social connections
    degree = dict(G.degree())

    # 2. Total communication volume
    weighted_degree = dict(G.degree(weight="weight"))

    # 3. Local network density
    clustering = nx.clustering(G, weight="weight")

    features = []

    for node in nodes:

        feature_vector = [
            float(degree[node]),
            float(weighted_degree[node]),
            float(clustering[node])
        ]

        features.append(feature_vector)

    return np.array(features, dtype=float), nodes


def normalize_features(features):
    """
    Normalize each feature column to [0, 1].
    """

    features = np.asarray(features, dtype=float)

    minimum = np.min(features, axis=0)
    maximum = np.max(features, axis=0)

    denominator = maximum - minimum

    denominator[denominator == 0] = 1.0

    normalized = (features - minimum) / denominator

    return normalized


def build_feature_cost_matrix(
    G,
    H
):
    """
    Construct the node similarity/cost matrix C.

    C[i,j] = Euclidean distance between the social
    features of node i in G and node j in H.
    """

    G_features, G_nodes = compute_node_features(G)
    H_features, H_nodes = compute_node_features(H)

    # Normalize G and H together so that the
    # feature scales are comparable.
    combined = np.vstack([G_features, H_features])

    combined_normalized = normalize_features(combined)

    G_normalized = combined_normalized[
        :len(G_features)
    ]

    H_normalized = combined_normalized[
        len(G_features):
    ]

    C = np.zeros(
        (len(G_nodes), len(H_nodes)),
        dtype=float
    )

    for i in range(len(G_nodes)):
        for j in range(len(H_nodes)):

            difference = (
                G_normalized[i]
                - H_normalized[j]
            )

            C[i, j] = np.linalg.norm(
                difference
            )

    # Final min-max normalization
    c_min = np.min(C)
    c_max = np.max(C)

    if c_max > c_min:
        C = (C - c_min) / (c_max - c_min)

    return C, G_nodes, H_nodes


def print_feature_statistics(
    G,
    H
):
    """
    Print feature information for the two real
    social graphs.
    """

    G_features, G_nodes = compute_node_features(G)
    H_features, H_nodes = compute_node_features(H)

    print("\nSocial Node Features")
    print("=" * 50)

    print("\nGraph G — Early Period")
    print("Feature columns:")
    print("  [Degree, Weighted Degree, Clustering]")

    print("\nFirst 5 nodes:")

    for i in range(min(5, len(G_nodes))):

        print(
            f"User {G_nodes[i]}: "
            f"{G_features[i]}"
        )

    print("\nGraph H — Later Period")

    print("First 5 nodes:")

    for i in range(min(5, len(H_nodes))):

        print(
            f"User {H_nodes[i]}: "
            f"{H_features[i]}"
        )


if __name__ == "__main__":

    from data_loader import (
        load_college_msg,
        split_temporal_data
    )

    from graph_builder import (
        build_real_social_graphs
    )

    # Load real CollegeMsg data
    data = load_college_msg()

    # Split into early/later periods
    early_data, later_data = split_temporal_data(data)

    # Build real social graphs
    G, H, users = build_real_social_graphs(
        early_data,
        later_data,
        num_users=40
    )

    # Display social features
    print_feature_statistics(G, H)

    # Build node cost matrix
    C, G_nodes, H_nodes = build_feature_cost_matrix(
        G,
        H
    )

    print("\nCost Matrix C")
    print("=" * 50)

    print(f"Shape: {C.shape}")
    print(f"Minimum: {C.min():.4f}")
    print(f"Maximum: {C.max():.4f}")
    print(f"Mean:    {C.mean():.4f}")

    print("\nFirst 5 x 5 portion of C:")

    print(
        np.round(
            C[:5, :5],
            4
        )
    )

    print("\nReal social feature construction completed.")