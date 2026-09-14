"""
Weisfeiler-Lehman Graph Kernel
--------------------------------
Computes a WL subtree-style similarity between two NetworkX graphs.

The implementation supports weighted graphs by incorporating edge weights
into the neighborhood representation.

Output:
    similarity : normalized similarity in [0, 1]
"""

import networkx as nx
from collections import Counter


def _initial_labels(G):
    """
    Create initial node labels using weighted degree.

    For weighted graphs, the weighted degree is used.
    For unweighted graphs, ordinary degree is used.
    """
    labels = {}

    for node in G.nodes():
        weighted_degree = G.degree(node, weight="weight")

        # Round to avoid tiny floating-point differences
        labels[node] = f"deg:{round(weighted_degree, 6)}"

    return labels


def _refine_labels(G, labels):
    """
    Perform one Weisfeiler-Lehman label refinement step.

    Each node receives a new label based on:
        current node label
        +
        sorted neighbor labels and edge weights
    """
    new_labels = {}

    for node in G.nodes():

        neighborhood = []

        for neighbor in G.neighbors(node):
            weight = G[node][neighbor].get("weight", 1.0)

            neighborhood.append(
                (
                    labels[neighbor],
                    round(float(weight), 6)
                )
            )

        neighborhood.sort(key=str)

        signature = (
            labels[node],
            tuple(neighborhood)
        )

        new_labels[node] = str(signature)

    # Compress identical signatures into integer labels
    # so that equivalent structures receive the same label.
    unique_signatures = {}
    compressed = {}

    next_label = 0

    for node in G.nodes():

        signature = new_labels[node]

        if signature not in unique_signatures:
            unique_signatures[signature] = next_label
            next_label += 1

        compressed[node] = unique_signatures[signature]

    return compressed


def _feature_vector(G, h=3):
    """
    Generate the WL feature vector.

    At each iteration, count how many nodes have each label.
    """
    labels = _initial_labels(G)

    feature_counts = Counter(labels.values())

    for _ in range(h):

        labels = _refine_labels(G, labels)

        feature_counts.update(labels.values())

    return feature_counts


def _dot_product(counter_a, counter_b):
    """
    Compute dot product between two sparse feature vectors.
    """
    common_keys = set(counter_a.keys()) & set(counter_b.keys())

    return sum(
        counter_a[key] * counter_b[key]
        for key in common_keys
    )


def wl_kernel_similarity(G, H, h=3):
    """
    Compute normalized WL kernel similarity.

    Returns:
        float in [0, 1]
    """

    features_G = _feature_vector(G, h=h)
    features_H = _feature_vector(H, h=h)

    numerator = _dot_product(features_G, features_H)

    norm_G = _dot_product(features_G, features_G)
    norm_H = _dot_product(features_H, features_H)

    if norm_G == 0 or norm_H == 0:
        return 0.0

    similarity = numerator / ((norm_G * norm_H) ** 0.5)

    # Numerical protection
    similarity = max(0.0, min(1.0, similarity))

    return float(similarity)


def run_wl_kernel(G, H, h=3):
    """
    Wrapper used by the benchmark.

    Returns a dictionary so that all kernels can eventually
    use the same interface.
    """

    similarity = wl_kernel_similarity(G, H, h=h)

    return {
        "method": "WL",
        "similarity": similarity,
        "iterations": h
    }


if __name__ == "__main__":

    # Small sanity test
    G = nx.Graph()
    G.add_edge(0, 1, weight=2)
    G.add_edge(1, 2, weight=1)

    H = nx.Graph()
    H.add_edge(10, 11, weight=2)
    H.add_edge(11, 12, weight=1)

    result = run_wl_kernel(G, H, h=3)

    print("WL Kernel Test")
    print("----------------")
    print(f"Similarity: {result['similarity']:.6f}")
    print(f"WL iterations: {result['iterations']}")