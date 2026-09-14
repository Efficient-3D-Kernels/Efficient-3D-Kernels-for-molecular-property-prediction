"""
performance_report.py
=======================
The numbers you put on your results slide should come from this file
(or your own re-run of it), not from an unverified script. Every
section below reuses only the already-verified functions from
gnccp_matcher.py / gnccp_naive_baseline.py / combinatorial_cost.py /
threshold_calibration.py -- nothing new is computed here, this file
just times and aggregates them properly.

FIX vs. a naive benchmark: every timing number below is a MEDIAN over
repeated trials, not a single measurement -- single-shot timings on a
shared machine are noisy (garbage collection, OS scheduling, thermal
throttling) and can make a clean O(N) vs O(N^2) comparison look
non-monotonic, which is exactly the artifact to avoid on a slide a
panel will scrutinize.

Run: python performance_report.py
Takes a few minutes on the full QMOF cache; reduce SAMPLE_SIZE below
for a faster pass while iterating.
"""

import os
import time
import statistics
import numpy as np

from dataset_loader import load_graph_cache
from mof_graph import load_structure, build_bonded_graph, find_bridging_carboxylate_path
from family_labels import primary_metal, heuristic_label
from gnccp_matcher import gnccp_match, co_matching_U
from gnccp_naive_baseline import naive_U
from threshold_calibration import youdens_j_threshold, stage2_verdict
from geometric_refinement import stage2_refine

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QUERY_DIR = os.path.join(_THIS_DIR, "data")
QUERY_STRUCTURE = "UiO-66.cif"
QUERY_METAL = "Zr"
CACHE_PATH = os.path.join(_THIS_DIR, "qmof_graph_cache.json")

SAMPLE_SIZE_CLASSIFICATION = 4000  # candidates scored for the AUC/threshold section
SAMPLE_SIZE_THROUGHPUT = 200       # candidates timed for the throughput estimate
TRIALS_PER_N = 25                  # repetitions per matrix size in the Lemma 1 sweep

# If your machine has limited RAM, set this to load only a prefix of the
# cache (streaming, memory-safe -- see dataset_loader.load_graph_cache).
# None = load everything (what you should use on your machine; this ran
# fine there already per your own terminal output).
LOAD_LIMIT = None


def line(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------
# 1. Lemma 1 scaling -- naive U vs rank-one U, MEDIAN over many trials
# ---------------------------------------------------------------------
def section_lemma1_scaling():
    line("1. LEMMA 1 SPEEDUP -- isolated U computation, naive vs rank-one\n"
         f"   (median of {TRIALS_PER_N} trials per N, not a single measurement)")
    rng = np.random.default_rng(0)
    M = 5
    print(f"{'N':>6s} {'naive (ms)':>12s} {'rank-1 (ms)':>12s} {'speedup':>10s}")
    for N in [5, 50, 200, 500, 1000, 2000]:
        X = rng.random((M, N))
        naive_times, fast_times = [], []
        for _ in range(TRIALS_PER_N):
            t0 = time.perf_counter(); naive_U(X); naive_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); co_matching_U(X); fast_times.append(time.perf_counter() - t0)
        t_naive = statistics.median(naive_times)
        t_fast = statistics.median(fast_times)
        print(f"{N:6d} {t_naive*1000:12.4f} {t_fast*1000:12.4f} {t_naive/t_fast:9.1f}x")


# ---------------------------------------------------------------------
# 2. Real pipeline throughput on your actual QMOF cache
# ---------------------------------------------------------------------
def section_throughput(graphs, Gq):
    line("2. FULL PIPELINE THROUGHPUT (real, on your actual dataset)")
    print(f"Cached candidate structures available: {len(graphs)}")
    sample_names = [n for n, g in graphs.items() if g.number_of_nodes() >= Gq.number_of_nodes()][:SAMPLE_SIZE_THROUGHPUT]
    t0 = time.perf_counter()
    for name in sample_names:
        gnccp_match(Gq, graphs[name])
    elapsed = time.perf_counter() - t0
    per_candidate = elapsed / len(sample_names)
    print(f"Timed sample: {len(sample_names)} candidates in {elapsed:.2f}s "
          f"({per_candidate*1000:.2f} ms/candidate)")
    total_est = per_candidate * len(graphs)
    print(f"Estimated time for all {len(graphs)} candidates: {total_est:.1f}s (~{total_est/60:.1f} min)")
    return per_candidate


# ---------------------------------------------------------------------
# 3 & 4. Stage 1 classification performance + Stage 2 match rate
# ---------------------------------------------------------------------
def section_classification(graphs, Gq):
    line("3. STAGE 1 (2D) CLASSIFICATION PERFORMANCE")
    names_all = list(graphs.keys())[:SAMPLE_SIZE_CLASSIFICATION]
    names, scores, labels = [], [], []
    mappings = {}
    t0 = time.perf_counter()
    for name in names_all:
        Hc = graphs[name]
        if Hc.number_of_nodes() < Gq.number_of_nodes():
            continue
        r1 = gnccp_match(Gq, Hc)
        label = heuristic_label(QUERY_METAL, Hc)
        names.append(name)
        scores.append(r1["score"])
        labels.append(label)
        mappings[name] = r1["mapping"]
    elapsed = time.perf_counter() - t0
    scores = np.array(scores)
    labels = np.array(labels)
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    print(f"Scored {len(scores)} candidates in {elapsed:.1f}s")
    print(f"Usable candidates (query motif fits): {len(scores)}  ({n_pos} positive, {n_neg} negative)")

    if n_pos == 0 or n_neg == 0:
        print("Not enough of both classes in this sample -- increase SAMPLE_SIZE_CLASSIFICATION.")
        return None, None, None, None

    yj = youdens_j_threshold(scores, labels)
    print(f"AUC = {yj['auc']:.4f}")
    print(f"Youden's J threshold = {yj['threshold']:.4f}")
    print(f"TPR (recall) at threshold = {yj['tpr']:.4f}")
    print(f"FPR at threshold = {yj['fpr']:.4f}")

    line("4. STAGE 2 (3D) MATCH RATE AMONG STAGE-1 POSITIVES")
    pos_names = [names[i] for i in range(len(names)) if labels[i] == 1]
    match_count, decoy_count = 0, 0
    for name in pos_names:
        Hc = graphs[name]
        mapping = mappings.get(name)
        if not mapping or len(mapping) < 3:
            continue
        r2 = stage2_refine(Gq, Hc, mapping)
        verdict = stage2_verdict(r2["motif_score"], r2["rmsd"], conservative=False)
        if verdict == "MATCH":
            match_count += 1
        else:
            decoy_count += 1
    total2 = match_count + decoy_count
    print(f"Stage-1 positives reaching Stage 2: {total2}")
    if total2:
        print(f"  MATCH (RMSD <= 2.0 A)   : {match_count} ({match_count/total2*100:.1f}%)")
        print(f"  3D-DECOY                : {decoy_count} ({decoy_count/total2*100:.1f}%)")
    return yj, n_pos, n_neg, (match_count, decoy_count)


def main():
    line("Loading query motif")
    query_path_file = os.path.join(QUERY_DIR, QUERY_STRUCTURE)
    if os.path.exists(query_path_file):
        atoms = load_structure(query_path_file)
        Gq_full = build_bonded_graph(atoms)
        Gq, path = find_bridging_carboxylate_path(Gq_full, QUERY_METAL)
        print(f"Query loaded from real {QUERY_STRUCTURE}: "
              f"{'-'.join(Gq_full.nodes[n]['element'] for n in path)}")
    else:
        # sandbox fallback -- on your machine the branch above will run instead
        graphs_tmp = load_graph_cache(CACHE_PATH, limit=2000)
        Gq = None
        for name, G in graphs_tmp.items():
            sub, path = find_bridging_carboxylate_path(G, QUERY_METAL)
            if sub is not None:
                Gq = sub
                print(f"[sandbox fallback] query motif from {name} (no data/{QUERY_STRUCTURE} here)")
                break

    section_lemma1_scaling()

    line("Loading candidate structures")
    graphs = load_graph_cache(CACHE_PATH, limit=LOAD_LIMIT)
    print(f"Loaded {len(graphs)} structures from {CACHE_PATH}"
          + (f" (limited to first {LOAD_LIMIT} -- set LOAD_LIMIT=None for the full dataset)" if LOAD_LIMIT else ""))

    section_throughput(graphs, Gq)
    section_classification(graphs, Gq)

    line("DONE -- use the numbers above on your slide, not hand-typed estimates")


if __name__ == "__main__":
    main()