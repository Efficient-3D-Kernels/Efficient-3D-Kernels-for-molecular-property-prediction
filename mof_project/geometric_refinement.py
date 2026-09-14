"""
geometric_refinement.py
=========================
Stage 2: the 3D structural confirmation step, applied only to candidates
that passed the Stage-1 GNCCP gate.

Two clearly separated pieces:

  (A) THE PAPER'S ACTUAL STAGE 2 (Sec. X / Theorem 4 / Corollary 3 /
      Definition 5-6 of your theory paper): the bijective atom map X*
      from gnccp_matcher.gnccp_match reduces 3D alignment to a single
      O(n) Kabsch rotation. k3D_score and hybrid_score implement
      Definition 6's exponential kernel and the (1-beta)/beta blend.
      -> kabsch_rmsd, k3D_score, hybrid_score

  (B) A SUPPLEMENTARY diagnostic, not part of the WCS-GNCCP paper: the
      four-atom bond/angle/torsion motif similarity from Ankit, Bhadra
      & Rousu, "Efficient 3D kernels for molecular property prediction"
      (Bioinformatics, 2025), Eq. (3)-(4). Report it alongside RMSD as
      an extra local sanity check ("does the match also agree on bond
      angles/torsions, not just the global rigid overlay") -- but never
      attribute it to the WCS-GNCCP paper.

  stage2_refine() runs both and returns everything together.
"""

from __future__ import annotations
import numpy as np
import networkx as nx


# ---------------------------------------------------------------------
# (A) Theorem 4 / Corollary 3: Kabsch alignment, O(n)
# ---------------------------------------------------------------------

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Optimal-rotation RMSD between matched point sets P, Q (n, 3).
    Theorem 4: R* = V diag(1,1,det(V W^T)) W^T where H = P_centered^T
    Q_centered = W Sigma V^T. Corollary 3: O(n) total -- forming H is
    O(n) vector ops, its SVD is O(1) since H is always 3x3 regardless
    of how many atoms were matched."""
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    W, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ W.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ W.T
    P_rot = (R @ P.T).T
    diff = P_rot - Q
    return float(np.sqrt((diff ** 2).sum() / len(P)))


def k3D_score(rmsd: float, gamma3D: float = 1.0) -> float:
    """k3D(G,H) = exp(-gamma3D * RMSD), the geometric half of the
    hybrid kernel (Definition 6). Calibrate gamma3D from labeled data
    (ROC/Youden's J on the Stage-2 score distribution) the same way the
    Stage-1 threshold is calibrated, rather than guessing it."""
    if rmsd is None or not np.isfinite(rmsd):
        return 0.0
    return float(np.exp(-gamma3D * rmsd))


def hybrid_score(k2D: float, rmsd: float, beta: float = 0.5, gamma3D: float = 1.0) -> float:
    """k_hybrid = (1-beta) k2D + beta k3D  (Definition 6). k2D should be
    gnccp_match(...)['score'] (Proposition 3's exp(-gamma F*/L)). beta
    is meant to be cross-validated per target -- 0.5 is a neutral
    starting point, not a validated value."""
    return (1 - beta) * k2D + beta * k3D_score(rmsd, gamma3D)


# ---------------------------------------------------------------------
# (B) Supplementary diagnostic -- Ankit, Bhadra & Rousu (2025), Eq. 3-4
# ---------------------------------------------------------------------

def bond_length(pA, pB) -> float:
    return float(np.linalg.norm(pA - pB))


def bond_angle_cos(dAB, dBC, dAC) -> float:
    """cos(theta) at B for path A-B-C. Eq. (3)."""
    denom = 2 * dAB * dBC
    if denom == 0:
        return 1.0
    val = (dAB**2 + dBC**2 - dAC**2) / denom
    return float(np.clip(val, -1.0, 1.0))


def torsion_cos(pA, pB, pC, pD) -> float:
    """cos(phi) torsion for four consecutive atoms A-B-C-D, via the
    standard dihedral formula (equivalent to Eq. (4))."""
    b1, b2, b3 = pB - pA, pC - pB, pD - pC
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    n1n, n2n = np.linalg.norm(n1), np.linalg.norm(n2)
    if n1n < 1e-8 or n2n < 1e-8:
        return 1.0
    cos_phi = np.dot(n1, n2) / (n1n * n2n)
    return float(np.clip(cos_phi, -1.0, 1.0))


def four_atom_feature_vector(positions4: list) -> np.ndarray:
    pA, pB, pC, pD = positions4
    dAB, dBC, dCD = bond_length(pA, pB), bond_length(pB, pC), bond_length(pC, pD)
    dAC, dBD = bond_length(pA, pC), bond_length(pB, pD)
    ang_B = bond_angle_cos(dAB, dBC, dAC)
    ang_C = bond_angle_cos(dBC, dCD, dBD)
    tors = torsion_cos(pA, pB, pC, pD)
    return np.array([dAB, dBC, dCD, ang_B, ang_C, tors])


def motif_similarity(fq: np.ndarray, fc: np.ndarray) -> float:
    dist_part = np.exp(-np.abs(fq[:3] - fc[:3]))
    angle_part = np.exp(-np.abs(fq[3:] - fc[3:]))
    return float(np.mean(np.concatenate([dist_part, angle_part])))


# ---------------------------------------------------------------------
# Stage-2 driver
# ---------------------------------------------------------------------

def stage2_refine(Gq: nx.Graph, Hc: nx.Graph, mapping: dict, n_paths: int = 25,
                   beta: float = 0.5, gamma3D: float = 1.0, k2D: float = None):
    """
    Returns dict with:
      'rmsd'         : Kabsch RMSD (Angstrom) over ALL matched atoms (Theorem 4)
      'k3D'          : exp(-gamma3D * rmsd)                          (Def. 6)
      'hybrid'       : (1-beta)*k2D + beta*k3D, only if k2D was passed in
      'motif_score'  : supplementary 4-atom bond/angle/torsion similarity
      'n_motifs'     : number of 4-atom chains sampled
    """
    query_nodes = list(mapping.keys())

    all_paths = []
    for source in query_nodes:
        for target in query_nodes:
            if source == target:
                continue
            for path in nx.all_simple_paths(Gq, source, target, cutoff=3):
                if len(path) == 4:
                    all_paths.append(path)
                if len(all_paths) >= n_paths:
                    break
            if len(all_paths) >= n_paths:
                break
        if len(all_paths) >= n_paths:
            break

    motif_scores = []
    for path in all_paths:
        if any(p not in mapping for p in path):
            continue
        cand_path = [mapping[p] for p in path]
        pq = [Gq.nodes[p]["pos"] for p in path]
        pc = [Hc.nodes[p]["pos"] for p in cand_path]
        motif_scores.append(motif_similarity(four_atom_feature_vector(pq), four_atom_feature_vector(pc)))
    motif_score = float(np.mean(motif_scores)) if motif_scores else float("nan")

    P = np.array([Gq.nodes[q]["pos"] for q in mapping.keys()])
    Q = np.array([Hc.nodes[mapping[q]]["pos"] for q in mapping.keys()])
    rmsd = kabsch_rmsd(P, Q) if len(P) >= 3 else float("nan")

    k3D = k3D_score(rmsd, gamma3D)
    result = {
        "rmsd": rmsd,
        "k3D": k3D,
        "motif_score": motif_score,
        "n_motifs": len(motif_scores),
    }
    if k2D is not None:
        result["hybrid"] = hybrid_score(k2D, rmsd, beta=beta, gamma3D=gamma3D)
    return result