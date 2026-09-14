"""
threshold_calibration.py
==========================
Step 6, the actual answer to "on what basis do we choose 0.55, or the
3D cutoff": we do NOT choose them by hand. We compute a full score
distribution over labeled data and derive the threshold from that
distribution, using standard, textbook methods:

  - ROC curve (Receiver Operating Characteristic)
  - AUC (Area Under the Curve) -- single number summarizing how well
    the score separates true matches from true non-matches overall
    (0.5 = useless/random, 1.0 = perfect separation)
  - Youden's J statistic -- the standard way to convert an ROC curve
    into ONE recommended threshold: J = TPR - FPR, maximized
  - A "recall-constrained" alternative threshold: the loosest cutoff
    that still keeps >= 95% of true matches -- this is what you use
    for a SCREENING gate (Stage 1), where losing true positives is
    costly and false positives just cost a bit of extra Stage-2 compute
  - For Stage 2 (3D), in addition to whatever your own ROC curve says,
    we report the literature-standard RMSD <= 2.0 A "correct pose"
    criterion used throughout structure-based drug discovery (Bostrom
    et al. 2001 / Glide validation studies / PoseBusters, Deane group,
    Oxford, 2024 / CHARMMing docking benchmarks -- this exact number,
    2.0 A, recurs across all of them as the pass/fail line). For a
    higher-stakes decision (e.g. actually recommending a drug
    candidate) we ALSO report the stricter 1.0 A bar that several of
    those same papers use side-by-side with 2.0 A specifically to show
    results at a more conservative risk tolerance.

IMPORTANT CAVEAT (say this explicitly to your panel): a real ROC curve
needs a reasonably large, balanced labeled sample -- rule-of-thumb
minimums quoted in biostatistics texts are on the order of dozens of
positives AND dozens of negatives before the resulting AUC/threshold
is considered stable rather than noisy. Our 7-structure demo set is
NOT enough for a trustworthy ROC curve by itself; this file is built to
run on the full labeled QMOF slice, where you will have the needed
sample size. Run on 7 structures ONLY to confirm the code works, not to
claim a validated threshold.
"""

from __future__ import annotations
import numpy as np


def roc_points(scores: np.ndarray, labels: np.ndarray):
    """
    scores : array of Stage-1 (or Stage-2) similarity scores, one per
             candidate, for a FIXED query.
    labels : array of 0/1 ground truth (1 = true match), same length.

    Returns arrays of (thresholds, tpr, fpr) swept across every
    distinct score value present in the data -- this is what a
    from-scratch ROC curve actually is; no black box.
    """
    order = np.argsort(-scores)  # descending
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        raise ValueError(
            "Need at least one positive AND one negative example to "
            "compute an ROC curve. Your labeled sample only has one "
            "class -- pull in more structures / more families."
        )

    tp = np.cumsum(labels_sorted == 1)
    fp = np.cumsum(labels_sorted == 0)
    tpr = tp / P
    fpr = fp / N

    thresholds = scores_sorted
    return thresholds, tpr, fpr


def auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """AUC via the trapezoidal rule -- standard, simple, exact for a
    piecewise-linear ROC curve built from discrete score thresholds."""
    order = np.argsort(fpr)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz  # numpy >=2.0 renamed trapz -> trapezoid
    return float(trapz_fn(tpr[order], fpr[order]))


def youdens_j_threshold(scores: np.ndarray, labels: np.ndarray):
    """The single 'best balance' threshold: maximizes TPR - FPR."""
    thresholds, tpr, fpr = roc_points(scores, labels)
    j = tpr - fpr
    best_idx = np.argmax(j)
    return {
        "threshold": float(thresholds[best_idx]),
        "tpr": float(tpr[best_idx]),
        "fpr": float(fpr[best_idx]),
        "youden_j": float(j[best_idx]),
        "auc": auc_trapezoid(fpr, tpr),
    }


def recall_constrained_threshold(scores: np.ndarray, labels: np.ndarray, min_recall: float = 0.95):
    """
    The 'screening funnel' threshold: the LOOSEST cutoff that still
    keeps at least `min_recall` fraction of true matches. Use this for
    Stage 1 -- you WANT to be lenient here on purpose; Stage 2 is your
    real safety net.
    """
    thresholds, tpr, fpr = roc_points(scores, labels)
    valid = np.where(tpr >= min_recall)[0]
    if len(valid) == 0:
        raise ValueError(
            f"No threshold in this data achieves {min_recall:.0%} recall; "
            f"best achievable recall is {tpr.max():.0%}. You likely need "
            f"more / better-separated labeled examples."
        )
    # among thresholds achieving the recall floor, pick the one with
    # the LOWEST false positive rate (least junk let through)
    best_local = valid[np.argmin(fpr[valid])]
    return {
        "threshold": float(thresholds[best_local]),
        "tpr": float(tpr[best_local]),
        "fpr": float(fpr[best_local]),
        "min_recall_target": min_recall,
        "auc": auc_trapezoid(fpr, tpr),
    }


# ---------------------------------------------------------------------
# Stage-2 (3D) threshold: literature value + optional local check
# ---------------------------------------------------------------------

RMSD_STANDARD_A = 2.0   # the field-wide "correct pose" convention
RMSD_CONSERVATIVE_A = 1.0  # stricter bar for high-stakes decisions


def stage2_verdict(motif_score: float, rmsd: float, conservative: bool = False) -> str:
    """
    Literature-grounded 3D decision rule.

    conservative=False -> RMSD <= 2.0 A (the standard docking-pose
                           "success" criterion, used across essentially
                           the entire structure-based drug design field)
    conservative=True  -> RMSD <= 1.0 A (the stricter bar some of the
                           same benchmark papers report side-by-side,
                           for higher-stakes calls)

    motif_score threshold of 0.6 is the ROC-derived cutoff you should
    replace with your own Youden's-J or recall-constrained result once
    you have run the calibration on your labeled QMOF slice -- it is
    NOT hard-coded folklore, it is a placeholder to be replaced by the
    output of youdens_j_threshold() on YOUR Stage-2 score distribution.
    """
    cutoff = RMSD_CONSERVATIVE_A if conservative else RMSD_STANDARD_A
    if motif_score >= 0.6 and rmsd <= cutoff:
        return "MATCH"
    return "3D-DECOY"


def summarize(name: str, result: dict) -> str:
    return (
        f"{name}: threshold={result['threshold']:.3f}  "
        f"TPR(recall)={result['tpr']:.3f}  FPR={result['fpr']:.3f}  "
        f"AUC={result['auc']:.3f}"
    )