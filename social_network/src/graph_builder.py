import numpy as np
import networkx as nx


def build_weighted_social_graph(interactions, users=None):
    """
    Convert temporal message interactions into an undirected
    weighted social graph.

    Each node = a real user.
    Each edge = communication between two users.
    Edge weight = number of messages exchanged.

    Parameters
    ----------
    interactions : pandas.DataFrame
        Columns: source, target, timestamp

    users : list, optional
        If provided, only these users are included.

    Returns
    -------
    G : networkx.Graph
        Weighted undirected social graph.
    """

    G = nx.Graph()

    if users is not None:
        G.add_nodes_from(users)

    for _, row in interactions.iterrows():

        source = int(row["source"])
        target = int(row["target"])

        # Ignore self-messages
        if source == target:
            continue

        # If a user subset is specified, ignore outside users
        if users is not None:
            if source not in users or target not in users:
                continue

        if G.has_edge(source, target):
            G[source][target]["weight"] += 1.0
        else:
            G.add_edge(source, target, weight=1.0)

    return G


def select_active_users(
    early_data,
    later_data,
    num_users=40
):
    """
    Select a manageable set of real users who are active
    in both temporal periods.

    Users are ranked by their total number of interactions
    across both periods.
    """

    early_counts = {}

    for _, row in early_data.iterrows():
        source = int(row["source"])
        target = int(row["target"])

        early_counts[source] = early_counts.get(source, 0) + 1
        early_counts[target] = early_counts.get(target, 0) + 1

    later_counts = {}

    for _, row in later_data.iterrows():
        source = int(row["source"])
        target = int(row["target"])

        later_counts[source] = later_counts.get(source, 0) + 1
        later_counts[target] = later_counts.get(target, 0) + 1

    common_users = set(early_counts.keys()) & set(later_counts.keys())

    # Rank common users by combined activity
    ranked_users = sorted(
        common_users,
        key=lambda user: early_counts[user] + later_counts[user],
        reverse=True
    )

    selected_users = ranked_users[:num_users]

    print("\nUser selection:")
    print(f"Common users available: {len(common_users)}")
    print(f"Users selected:          {len(selected_users)}")

    return selected_users


def remove_isolated_nodes(G):
    """
    Remove users with no connections in the constructed graph.
    """

    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    return G


def print_graph_statistics(G, name="Graph"):
    """
    Print basic statistics for a social graph.
    """

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes > 1:
        density = nx.density(G)
    else:
        density = 0.0

    total_weight = sum(
        data.get("weight", 1.0)
        for _, _, data in G.edges(data=True)
    )

    degrees = dict(G.degree())

    if degrees:
        average_degree = np.mean(list(degrees.values()))
        maximum_degree = max(degrees.values())
    else:
        average_degree = 0.0
        maximum_degree = 0

    print(f"\n{name}")
    print("-" * 40)
    print(f"Nodes:             {num_nodes}")
    print(f"Edges:             {num_edges}")
    print(f"Density:           {density:.4f}")
    print(f"Total edge weight: {total_weight:.2f}")
    print(f"Average degree:    {average_degree:.2f}")
    print(f"Maximum degree:    {maximum_degree}")


def build_real_social_graphs(
    early_data,
    later_data,
    num_users=40
):
    """
    Complete pipeline for constructing the two real
    temporal social graphs.
    """

    # Select users who exist in BOTH time periods
    selected_users = select_active_users(
        early_data,
        later_data,
        num_users=num_users
    )

    # Build early graph
    G = build_weighted_social_graph(
        early_data,
        users=selected_users
    )

    # Build later graph
    H = build_weighted_social_graph(
        later_data,
        users=selected_users
    )

    # Remove users that became isolated
    # in either graph so both graphs retain the same node set.
    common_nodes = set(G.nodes()) & set(H.nodes())

    G = G.subgraph(common_nodes).copy()
    H = H.subgraph(common_nodes).copy()

    print_graph_statistics(G, "Early Social Graph G")
    print_graph_statistics(H, "Later Social Graph H")

    return G, H, sorted(common_nodes)


if __name__ == "__main__":

    # This test imports the real-data loader.
    from data_loader import load_college_msg, split_temporal_data

    data = load_college_msg()

    early_data, later_data = split_temporal_data(data)

    G, H, users = build_real_social_graphs(
        early_data,
        later_data,
        num_users=40
    )

    print("\nGraph construction test completed successfully.")

    print("\nSelected real users:")
    print(users)

    print("\nFirst 10 edges in G:")
    print(list(G.edges(data=True))[:10])

    print("\nFirst 10 edges in H:")
    print(list(H.edges(data=True))[:10])