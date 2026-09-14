"""
run_threshold_calibration.py
==============================
Runs the ACTUAL WCS-GNCCP algorithm (gnccp_matcher.py, implementing
Algorithm 1 from your paper) across the real MOF dataset, then
calibrates the Stage-1 threshold from real labeled data (ROC / Youden's
J), and runs Stage 2 (Kabsch RMSD + bond/angle/torsion) on survivors.

WHAT CHANGED FROM THE OLD VERSION:
  - Stage 1 now uses gnccp_matcher.wcs_gnccp_screen() (real GNCCP
    continuation + Frank-Wolfe + Hungarian oracle + rank-one U +
    closed-form gradient), NOT the old one-shot-Hungarian-only
    combinatorial_matcher.py. Delete that old file -- see chat.
  - Candidates with NO comparable motif (e.g. ZIF-8, which has no
    carboxylate bridge) are correctly excluded rather than forced
    through with a score of 0 that would poison the ROC curve.
"""

import os
import time
import numpy as np

from mof_graph import load_structure, build_bonded_graph, find_bridging_carboxylate_path
from dataset_loader import bulk_load, save_graph_cache, load_graph_cache
from family_labels import primary_metal, heuristic_label
from gnccp_matcher import wcs_gnccp_screen
from geometric_refinement import stage2_refine
from threshold_calibration import (
    youdens_j_threshold,
    recall_constrained_threshold,
    stage2_verdict,
    summarize,
)

# ---------------------------------------------------------------------
# CONFIGURATION -- edit these paths for your machine
# ---------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

QUERY_DIR = os.path.join(_THIS_DIR, "data")
QUERY_STRUCTURE = "UiO-66.cif"
QUERY_METAL = "Zr"

# >>> EDIT THIS to your actual unzipped QMOF folder path <<<
DATA_DIR = r"C:\Amrita\final year project\Efficient-3D-Kernels-for-molecular-property-prediction\qmof_data"

LIMIT = None  # None = full dataset

CACHE_PATH = os.path.join(_THIS_DIR, "qmof_graph_cache.json")
USE_CACHE_IF_EXISTS = True

GAMMA = 10.0  # see gnccp_matcher.py docstring for why 0.5 (paper default) doesn't separate well here


def load_candidates():
    if CACHE_PATH and USE_CACHE_IF_EXISTS and os.path.exists(CACHE_PATH):
        try:
            print(f"Loading cached graphs from {CACHE_PATH} (skipping slow CIF parsing)")
            return load_graph_cache(CACHE_PATH), []
        except Exception as e:
            print(f"Cache file exists but is corrupted/unreadable ({e}).")
            print("Deleting it and re-parsing from the original CIFs instead...")
            os.remove(CACHE_PATH)
    graphs, failures = bulk_load(DATA_DIR, limit=LIMIT)
    if CACHE_PATH:
        save_graph_cache(graphs, CACHE_PATH)
    return graphs, failures


def main():
    print("=" * 78)
    print(f"STEP 1: Loading the QUERY structure ({QUERY_STRUCTURE}) from {QUERY_DIR}")
    print("=" * 78)
    query_path = os.path.join(QUERY_DIR, QUERY_STRUCTURE)
    atoms = load_structure(query_path)
    Gq_full = build_bonded_graph(atoms)
    Gq, path = find_bridging_carboxylate_path(Gq_full, QUERY_METAL)
    if Gq is None:
        raise RuntimeError(f"No {QUERY_METAL}-O-C-O-{QUERY_METAL} bridge found in {QUERY_STRUCTURE}")
    print(f"Query motif path: {'-'.join(Gq_full.nodes[n]['element'] for n in path)}")

    print()
    print("=" * 78)
    print(f"STEP 2: Loading CANDIDATE structures from {DATA_DIR}")
    print("=" * 78)
    graphs, failures = load_candidates()
    if failures:
        print(f"({len(failures)} candidate files failed to load / were skipped)")
    print(f"Total candidates loaded: {len(graphs)}")

    print()
    print("=" * 78)
    print("STEP 3: Running WCS-GNCCP (Algorithm 1) -- Stage 1, exhaustive")
    print("(candidates with no comparable motif, e.g. non-carboxylate")
    print(" frameworks like ZIF-8, are correctly excluded, not scored 0)")
    print("=" * 78)
    names, s1_scores, labels = [], [], []
    s1_details = {}
    excluded_no_motif = 0
    t0 = time.perf_counter()
    for i, (name, Hc) in enumerate(graphs.items(), 1):
        r1 = wcs_gnccp_screen(Gq, Hc, gamma=GAMMA)
        if r1.get("no_motif"):
            excluded_no_motif += 1
            continue
        label = heuristic_label(QUERY_METAL, Hc)
        names.append(name)
        s1_scores.append(r1["score"])
        labels.append(label)
        s1_details[name] = r1
        if label == 1:
            print(f"{name:20s} s={r1['score']:.3f}  metal={primary_metal(Hc)!s:4s}  label=POSITIVE (same family)")
        if i % 200 == 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed
            remaining = (len(graphs) - i) / rate if rate > 0 else float("nan")
            n_pos_so_far = sum(labels)
            print(
                f"  ... scored {i}/{len(graphs)}  ({n_pos_so_far} positives, "
                f"{excluded_no_motif} excluded so far)  "
                f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]"
            )

    print(f"\nExcluded (no comparable motif): {excluded_no_motif}")
    s1_scores = np.array(s1_scores)
    labels = np.array(labels)

    print()
    print("=" * 78)
    print("STEP 4: Calibrating the Stage-1 threshold from this labeled data")
    print("=" * 78)
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    print(f"Labeled sample: {n_pos} positive, {n_neg} negative")
    if n_pos < 10 or n_neg < 10:
        print("*** WARNING: fewer than 10 examples per class. Code-testing only. ***")

    yj = rc = None
    try:
        yj = youdens_j_threshold(s1_scores, labels)
        print("Youden's J (best overall balance):     ", summarize("Stage-1", yj))
    except ValueError as e:
        print("Youden's J: could not compute --", e)

    try:
        rc = recall_constrained_threshold(s1_scores, labels, min_recall=0.95)
        print("Recall-constrained (>=95% recall):     ", summarize("Stage-1", rc))
    except ValueError as e:
        print("Recall-constrained threshold: could not compute --", e)

    print()
    print("=" * 78)
    print("STEP 5: Stage 2 (Kabsch RMSD + bond/angle/torsion) on positives")
    print("=" * 78)
    for name in names:
        idx = names.index(name)
        if labels[idx] == 0:
            continue
        Hc = graphs[name]
        mapping = s1_details[name]["mapping"]
        if len(mapping) < 3:
            print(f"{name:20s} (mapping too small for RMSD, skipped)")
            continue
        r2 = stage2_refine(Gq, Hc, mapping)
        verdict = stage2_verdict(r2["motif_score"], r2["rmsd"], conservative=False)
        print(f"{name:20s} motif_score={r2['motif_score']:.3f}  RMSD={r2['rmsd']:.3f} A  -> {verdict}")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Total candidates screened      : {len(graphs)}")
    print(f"Excluded (no comparable motif) : {excluded_no_motif}")
    print(f"Positive (same-metal) examples : {n_pos}")
    print(f"Negative examples              : {n_neg}")
    if yj:
        print(f"Recommended Stage-1 threshold (Youden's J) : {yj['threshold']:.3f}  (AUC={yj['auc']:.3f})")
    if rc:
        print(f"Recommended Stage-1 threshold (95% recall) : {rc['threshold']:.3f}")


if __name__ == "__main__":
    main()