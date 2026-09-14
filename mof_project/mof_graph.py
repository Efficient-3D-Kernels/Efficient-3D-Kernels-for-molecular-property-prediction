"""
mof_graph.py
============
Step 1 of the pipeline: turn a real crystallographic MOF file (.cif) into a
graph object that the matcher can use.

Two things are extracted from every structure, exactly mirroring what the
two reference papers use:

  (a) 2D topology  -> an adjacency structure (who is bonded to whom, what
                       element each atom is, what its degree is).
                       This is all Stage 1 (the "2D screening gate") is
                       allowed to look at.

  (b) 3D geometry  -> Cartesian coordinates of every atom, used ONLY in
                       Stage 2, to compute bond length / bond angle /
                       torsion angle for the candidates that survive
                       Stage 1 (same three features as Ankit et al.,
                       "Efficient 3D kernels for molecular property
                       prediction", Bioinformatics 2025 - the second
                       paper you uploaded).

Why ASE and not a hand-rolled CIF reader
-----------------------------------------
Real CIF files are periodic (fractional coordinates + unit cell + space
group). Bonding must respect periodic boundary conditions, or atoms near
a cell edge look like isolated outliers. ASE (Atomic Simulation
Environment) correctly expands symmetry and gives us true Cartesian
coordinates; we then do our own bond perception (covalent-radius cutoff)
so that the *graph construction* itself is transparent and is ours, not
a black box.
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from ase.io import read
from ase.data import covalent_radii, atomic_numbers
from ase.neighborlist import NeighborList


# Bond tolerance: two atoms are considered bonded if
#   distance <= tolerance * (covalent_radius_A + covalent_radius_B)
# 1.20 is a standard tolerance used for organic / MOF bond perception
# (e.g. used identically in the OpenBabel / RDKit "xyz2mol" style rules).
BOND_TOLERANCE = 1.20


def load_structure(cif_path: str):
    """Read a CIF (periodic crystal) and return an ASE Atoms object with
    a real 3D unit cell and Cartesian atomic positions."""
    atoms = read(cif_path)
    return atoms


def build_bonded_graph(atoms, tolerance: float = BOND_TOLERANCE) -> nx.Graph:
    """
    Build the atomic graph G = (V, E) of a periodic structure.

    Node attributes:
        element   : str  (e.g. 'Zr', 'O', 'C')
        Z         : int  (atomic number)
        pos       : np.ndarray shape (3,)  -- unwrapped Cartesian coordinates
                    (periodic images are resolved so that every bonded pair
                    has a geometrically meaningful, non-wrapped distance)

    Edge attributes:
        length    : float  (bond length in Angstrom)

    This is the *periodic* analogue of the adjacency matrix A_G used in
    "An Algorithm for Finding the Most Similar Given Sized Subgraphs..."
    (Yang, Qiao & Liu, 2018) -- there it was a molecule's adjacency
    matrix; here it is a crystal's bonded network.

    NOTE ON TYPES: ASE's NeighborList.get_neighbors() returns neighbor
    indices as numpy.int64, not plain Python int. If those numpy scalars
    are used directly as networkx node IDs, every downstream consumer
    that assumes plain-int node IDs (json serialization in
    dataset_loader.save_graph_cache being the one that actually broke)
    can fail with "Object of type int64 is not JSON serializable". We
    cast to int() immediately on read so every node ID in every graph
    built by this function is a native Python int from the start.
    """
    radii = np.array([covalent_radii[z] for z in atoms.get_atomic_numbers()])
    max_radius = radii.max()
    cutoff = max_radius * tolerance + 0.1  # small pad

    nl = NeighborList(
        cutoffs=[cutoff] * len(atoms),
        self_interaction=False,
        bothways=True,
        skin=0.0,
    )
    nl.update(atoms)

    G = nx.Graph()
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    numbers = atoms.get_atomic_numbers()

    for i in range(len(atoms)):
        G.add_node(i, element=symbols[i], Z=int(numbers[i]), pos=positions[i].copy())

    cell = atoms.get_cell()
    for i in range(len(atoms)):
        neighbors, offsets = nl.get_neighbors(i)
        for j, offset in zip(neighbors, offsets):
            j = int(j)  # ASE returns numpy.int64 -- force plain Python int
            if j <= i and not offset.any():
                # avoid double counting the same (i, j) pair with zero offset
                if j < i:
                    continue
            pos_j = positions[j] + offset @ cell
            dist = np.linalg.norm(positions[i] - pos_j)
            r_sum = covalent_radii[numbers[i]] + covalent_radii[numbers[j]]
            if dist <= tolerance * r_sum:
                if not G.has_edge(i, j):
                    G.add_edge(i, j, length=float(dist))

    return G


def describe(G: nx.Graph, name: str = "") -> str:
    elems = {}
    for _, d in G.nodes(data=True):
        elems[d["element"]] = elems.get(d["element"], 0) + 1
    formula = " ".join(f"{e}{n}" for e, n in sorted(elems.items()))
    return f"{name:16s} |V|={G.number_of_nodes():4d}  |E|={G.number_of_edges():4d}   {formula}"


def extract_local_motif(G: nx.Graph, center: int, L: int) -> nx.Graph:
    """
    Extract an L-node query motif around `center` via breadth-first
    expansion. This is the "given-sized subgraph" the WCS formulation in
    the first paper asks for: a subgraph of a *specified* size L, taken
    from one graph, that we then search for inside another, larger graph.
    (Kept for general use; for the MOF demo we use the chemically
    motivated extractor below instead, since a plain BFS star has no
    4-atom chains and can't be scored geometrically for torsion.)
    """
    visited = [center]
    frontier = [center]
    while len(visited) < L and frontier:
        next_frontier = []
        for u in frontier:
            for v in G.neighbors(u):
                if v not in visited:
                    visited.append(v)
                    next_frontier.append(v)
                    if len(visited) >= L:
                        break
            if len(visited) >= L:
                break
        frontier = next_frontier
    nodes = visited[:L]
    return G.subgraph(nodes).copy()


def find_bridging_carboxylate_path(G: nx.Graph, metal_symbol: str):
    """
    Find one concrete metal-O-C-O-metal bridging-carboxylate path in the
    graph -- the real, textbook secondary-building-unit (SBU) linkage
    that holds every carboxylate-based MOF together (Zr6-carboxylate in
    UiO-66, Cu2-paddlewheel-carboxylate in HKUST-1, Zn4O-carboxylate in
    MOF-5/MOF-177, Al-carboxylate chains in MIL-53).

    This is a 5-atom chain M(1)-O(2)-C(3)-O(4)-M(5): exactly two
    overlapping four-consecutive-atom windows, [M,O,C,O] and [O,C,O,M],
    which is what the bond-length/angle/torsion machinery in
    geometric_refinement.py (and Eq. 3-4 of Ankit et al., 2025) needs.

    Returns a networkx.Graph (a 5-node path) or None if no such
    bridging path exists in this structure (e.g. a framework with no
    carboxylate linker at all).
    """
    for m1 in [n for n, d in G.nodes(data=True) if d["element"] == metal_symbol]:
        for o1 in G.neighbors(m1):
            if G.nodes[o1]["element"] != "O":
                continue
            for c in G.neighbors(o1):
                if G.nodes[c]["element"] != "C":
                    continue
                for o2 in G.neighbors(c):
                    if o2 == o1 or G.nodes[o2]["element"] != "O":
                        continue
                    for m2 in G.neighbors(o2):
                        if m2 == m1 or G.nodes[m2]["element"] != metal_symbol:
                            continue
                        path = [m1, o1, c, o2, m2]
                        return G.subgraph(path).copy(), path
    return None, None