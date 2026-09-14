"""
family_labels.py
==================
Step 5: ground truth labels for calibration.

MOFs don't come with "active/inactive" binding labels the way drug
candidates do. The defensible proxy, used by crystallographers
themselves to name and group MOF families, is the metal / secondary
building unit (SBU) composition:

    "UiO-66, UiO-67, UiO-68, NU-1000, PCN-224 ... are all called
     the UiO / Zr6-cluster family BECAUSE they share a Zr6O4(OH)4
     inorganic building block -- not because someone decided to
     group them that way for convenience."

So: two structures get ground-truth label 1 ("should match") if they
contain the SAME primary metal element bonded to the SAME kind of
bridging ligand chemistry (here, simplified to "same metal element AND
both contain a carboxylate-type M-O-C-O-M linkage"); label 0 otherwise.

This file also supports reading a metadata CSV (e.g. QMOF's own
properties table) if you have one -- real metal/topology labels from
the database itself are always better than the heuristic below when
available.
"""

from __future__ import annotations
import csv
import networkx as nx

from mof_graph import find_bridging_carboxylate_path

# Metals that commonly form well-known carboxylate-based MOF families,
# used only for the heuristic fallback labeler. Extend this list with
# whatever metals appear in your slice of QMOF.
KNOWN_SBU_METALS = ["Zr", "Zn", "Cu", "Al", "Fe", "Cr", "Ti", "Mg", "Ni", "Co"]


def primary_metal(G: nx.Graph) -> str | None:
    """Return the most common non-organic (metal) element in the graph,
    or None if no recognized metal is present."""
    counts = {}
    for _, d in G.nodes(data=True):
        e = d["element"]
        if e in KNOWN_SBU_METALS:
            counts[e] = counts.get(e, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def has_carboxylate_bridge(G: nx.Graph, metal: str) -> bool:
    path, _ = find_bridging_carboxylate_path(G, metal)
    return path is not None


def heuristic_label(Gq_metal: str, Hc: nx.Graph) -> int:
    """1 if Hc plausibly belongs to the same SBU family as the query
    metal, else 0. Purely composition + linkage-chemistry based --
    does NOT look at our own matcher's score, which is essential:
    labels used to validate a method must never be derived from that
    same method, or the validation is circular and meaningless."""
    metal = primary_metal(Hc)
    if metal != Gq_metal:
        return 0
    return 1 if has_carboxylate_bridge(Hc, metal) else 0


def load_labels_from_csv(csv_path: str, metal_column: str, id_column: str) -> dict[str, str]:
    """
    If you have QMOF's own metadata CSV (recommended -- real curated
    labels beat our heuristic), load {structure_id: metal_symbol} from
    it directly instead of inferring the metal from the graph.
    """
    labels = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels[row[id_column]] = row[metal_column]
    return labels