"""
dataset_loader.py
===================
Step 4 of the pipeline: load a LARGE folder of CIF files (e.g. the
20,000+ structures from QMOF) instead of a handful of hand-picked demo
files.

Real datasets at this scale always have some broken/unreadable files
(disordered occupancies, missing symmetry data, etc.) -- this loader
does not let one bad file crash the whole run; it records failures and
keeps going, and reports a clean summary at the end so you know exactly
how much of the dataset was usable.
"""

from __future__ import annotations
import os
import glob
import time
import json
import networkx as nx

from mof_graph import load_structure, build_bonded_graph


def bulk_load(
    directory: str,
    limit: int | None = None,
    progress_every: int = 200,
) -> tuple[dict[str, nx.Graph], list[tuple[str, str]]]:
    """
    Load every .cif in `directory` (non-recursive; use directory=**/*.cif
    style glob externally if you need recursion) into a bonded graph.

    Parameters
    ----------
    limit : cap the number of files processed (useful for a quick test
            run on, say, the first 500 of 20,000 before committing to
            the full run, which can take hours on a laptop).
    """
    files = sorted(glob.glob(os.path.join(directory, "*.cif")))
    if limit:
        files = files[:limit]

    graphs: dict[str, nx.Graph] = {}
    failures: list[tuple[str, str]] = []

    t0 = time.perf_counter()
    for i, f in enumerate(files, 1):
        name = os.path.basename(f)
        try:
            atoms = load_structure(f)
            if len(atoms) == 0 or len(atoms) > 2000:
                # skip empty or absurdly large structures (DFT unit
                # cells are occasionally supercells with thousands of
                # atoms; skip for now, revisit if you need them)
                failures.append((name, f"skipped: {len(atoms)} atoms"))
                continue
            G = build_bonded_graph(atoms)
            graphs[name] = G
        except Exception as e:
            failures.append((name, str(e)))

        if progress_every and i % progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed
            remaining = (len(files) - i) / rate if rate > 0 else float("nan")
            print(
                f"  ... {i}/{len(files)} processed "
                f"({len(graphs)} ok, {len(failures)} failed) "
                f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]"
            )

    print(
        f"bulk_load: {len(graphs)}/{len(files)} structures loaded successfully "
        f"({len(failures)} failed/skipped) in {time.perf_counter()-t0:.1f}s"
    )
    return graphs, failures


def save_graph_cache(graphs: dict[str, nx.Graph], path: str) -> None:
    """
    Building the bonded graph for 20,000 CIFs is the slow part (minutes
    to hours depending on your machine). Cache the result so you only
    pay that cost once, not every time you re-run the threshold
    calibration or try a different query motif.

    IMPORTANT: node IDs and edge endpoints coming from ASE/networkx can
    be numpy integer types (e.g. numpy.int64), which Python's built-in
    json module cannot serialize directly -- everything below is
    explicitly cast to plain Python int/float first. We also write to
    a temporary file and only rename it to the real path once writing
    finishes successfully, so a crash mid-write can never again leave
    you with a corrupted, half-written cache file that breaks every
    subsequent run.
    """
    data = {}
    for name, G in graphs.items():
        data[name] = {
            "nodes": [
                {
                    "id": int(n),
                    "element": str(d["element"]),
                    "Z": int(d["Z"]),
                    "pos": [float(x) for x in d["pos"]],
                }
                for n, d in G.nodes(data=True)
            ],
            "edges": [
                [int(u), int(v), float(G.edges[u, v]["length"])]
                for u, v in G.edges()
            ],
        }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp_path, path)  # atomic on both Windows and Linux
    print(f"Cached {len(graphs)} graphs to {path}")


def load_graph_cache(path: str, limit: int | None = None) -> dict[str, nx.Graph]:
    """
    limit : if set, only the first `limit` structures are loaded, using a
            STREAMING JSON parser (ijson) so the full file is never fully
            materialized in memory first -- useful on machines with
            limited RAM, or for a quick partial run while iterating.
            None (default) loads everything, identical to before.
    """
    import numpy as np

    graphs: dict[str, nx.Graph] = {}

    if limit is None:
        with open(path) as fh:
            data = json.load(fh)
        items = data.items()
    else:
        import ijson
        items = []
        with open(path, "rb") as fh:
            # use_float=True: ijson defaults to decimal.Decimal for JSON
            # numbers (to avoid float precision loss), which silently
            # poisons every position into an object-dtype numpy array
            # downstream (np.linalg.svd then fails with a cryptic
            # UFuncInputCastingError). Force plain floats to match the
            # full-load (json.load) path exactly.
            for i, (name, d) in enumerate(ijson.kvitems(fh, "", use_float=True)):
                if i >= limit:
                    break
                items.append((name, d))

    for name, d in items:
        G = nx.Graph()
        for n in d["nodes"]:
            G.add_node(n["id"], element=n["element"], Z=n["Z"], pos=np.array(n["pos"]))
        for u, v, length in d["edges"]:
            G.add_edge(u, v, length=length)
        graphs[name] = G
    return graphs