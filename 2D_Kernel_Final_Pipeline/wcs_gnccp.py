# wcs_gnccp.py - GPU ACCELERATED VERSION
# X remains CONTINUOUS during optimization!
# Only discretized to binary at the END!

import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

class WCS_GNCCP:
    """Weighted Common Subgraph Matching using GNCCP with Frank-Wolfe and Optional GPU Support"""
    
    def __init__(self, alpha=0.7, gamma=0.5, max_iter=50, fw_max_iter=20, tol=1e-4, use_gpu=False):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.fw_max_iter = fw_max_iter
        self.tol = tol
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np
        
    def match(self, mol1, mol2, L=None, verbose=True):
        """Main matching function"""
        if verbose:
            print("="*60)
            print("🔬 WCS+GNCCP Matching (Frank-Wolfe)")
            print(f"   Hardware Backend: {'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy)'}")
            print("="*60)
        
        # Build weighted adjacency matrices (CPU)
        A_G = self._build_weighted_adjacency(mol1)
        A_H = self._build_weighted_adjacency(mol2)
        
        M, N = A_G.shape[0], A_H.shape[0]
        
        # Normalize adjacency matrices so structural cost is scale-invariant
        norm_G = np.linalg.norm(A_G, 'fro')
        norm_H = np.linalg.norm(A_H, 'fro')
        if norm_G > 0: A_G = A_G / norm_G
        if norm_H > 0: A_H = A_H / norm_H
        
        # Set L
        if L is None:
            L = max(min(M, N) - 2, 1)
        else:
            L = min(L, M, N)
        
        if verbose:
            print(f"   Matching {L} atoms between {M} and {N} atoms")
            print(f"   Alpha (structure weight): {self.alpha}")
        
        # Build cost matrix (atom features on CPU)
        C = self._build_cost_matrix(mol1, mol2)
        
        # Normalize cost
        if C.max() > C.min():
            C = (C - C.min()) / (C.max() - C.min() + 1e-8)
        else:
            C = C / (C.max() + 1e-8)
        
        # Run GNCCP with Frank-Wolfe (GPU or CPU)
        X = self._gnccp_optimization(A_G, A_H, C, L, verbose)
        
        # Extract matches from binary X (ensure it's back on CPU for pure python logic)
        X_cpu = cp.asnumpy(X) if self.use_gpu else X
        
        matches = []
        for i in range(M):
            for j in range(N):
                if X_cpu[i, j] > 0.5:
                    matches.append((i, j))
        
        # Ensure exactly L matches
        if len(matches) > L:
            matches = matches[:L]
        elif len(matches) < L:
            remaining = []
            for i in range(M):
                for j in range(N):
                    if (i, j) not in matches and X_cpu[i, j] > 0.1:
                        remaining.append((i, j, X_cpu[i, j]))
            remaining.sort(key=lambda x: x[2], reverse=True)
            needed = L - len(matches)
            for i, j, _ in remaining[:needed]:
                matches.append((i, j))
        
        # Calculate final similarity score using the objective F(X)
        score = self._compute_final_score(A_G, A_H, C, X_cpu, L)
        
        if verbose:
            print(f"✅ Found {len(matches)} matches (Sim Score: {score:.3f})")
            print("="*60)
        
        return X_cpu, matches, score

    def _compute_final_score(self, A_G, A_H, C, X, L):
        """Computes the final exponential kernel similarity score."""
        r = X.sum(axis=1)
        U = np.outer(r, r)
        struct_diff = U * A_G - X @ A_H @ X.T
        struct_cost = np.sum(struct_diff ** 2)
        app_cost = np.einsum('ij,ij->', C, X)
        F_obj = self.alpha * struct_cost + (1 - self.alpha) * app_cost
        return np.exp(-self.gamma * F_obj / max(L, 1))
    
    def _build_weighted_adjacency(self, mol):
        """Build weighted adjacency matrix (CPU bound)."""
        n = mol.GetNumAtoms()
        A = np.zeros((n, n))
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bt = bond.GetBondType()
            if bt == Chem.BondType.SINGLE: w = 1.0
            elif bt == Chem.BondType.DOUBLE: w = 2.0
            elif bt == Chem.BondType.TRIPLE: w = 3.0
            elif bt == Chem.BondType.AROMATIC: w = 1.5
            else: w = 1.0
            if bond.IsInRing(): w += 0.1
            if bond.GetIsConjugated(): w += 0.1
            A[i, j] = w
            A[j, i] = w
        return A
    
    def _build_cost_matrix(self, mol1, mol2):
        """Build cost matrix via numpy broadcasting (CPU bound)."""
        f1 = self._extract_atom_features(mol1)
        f2 = self._extract_atom_features(mol2)
        C = np.zeros((len(f1['atomic_num']), len(f2['atomic_num'])))
        C += (f1['atomic_num'][:, None] != f2['atomic_num'][None, :]) * 100
        C += np.abs(f1['degree'][:, None] - f2['degree'][None, :]) * 10
        C += (f1['is_in_ring'][:, None] != f2['is_in_ring'][None, :]) * 5
        C += (f1['is_aromatic'][:, None] != f2['is_aromatic'][None, :]) * 10
        C += np.abs(f1['charge'][:, None] - f2['charge'][None, :]) * 20
        C += np.abs(f1['mass'][:, None] - f2['mass'][None, :]) * 0.5
        C += (f1['hybridization'][:, None] != f2['hybridization'][None, :]) * 15
        C += (f1['num_hs'][:, None] != f2['num_hs'][None, :]) * 5
        C += (f1['chiral_tag'][:, None] != f2['chiral_tag'][None, :]) * 5
        C += (f1['isotope'][:, None] != f2['isotope'][None, :]) * 2
        return C
    
    def _extract_atom_features(self, mol):
        """Extract features into numpy arrays (CPU bound)."""
        atoms = list(mol.GetAtoms())
        return {
            'atomic_num': np.array([a.GetAtomicNum() for a in atoms]),
            'degree': np.array([a.GetDegree() for a in atoms], dtype=float),
            'is_in_ring': np.array([a.IsInRing() for a in atoms]),
            'is_aromatic': np.array([a.GetIsAromatic() for a in atoms]),
            'mass': np.array([a.GetMass() for a in atoms]),
            'charge': np.array([a.GetFormalCharge() for a in atoms], dtype=float),
            'hybridization': np.array([str(a.GetHybridization()) for a in atoms]),
            'num_hs': np.array([a.GetTotalNumHs() for a in atoms]),
            'chiral_tag': np.array([str(a.GetChiralTag()) for a in atoms]),
            'isotope': np.array([a.GetIsotope() for a in atoms]),
        }
    
    def _gnccp_optimization(self, A_G, A_H, C, L, verbose=True):
        """GNCCP Optimization (Dynamically routes to CPU or GPU)"""
        xp = self.xp
        
        if self.use_gpu:
            A_G = xp.asarray(A_G)
            A_H = xp.asarray(A_H)
            C = xp.asarray(C)
        
        M, N = A_G.shape[0], A_H.shape[0]
        
        # Initialize X uniformly
        X = xp.ones((M, N)) * (L / (M * N))
        zeta_values = xp.linspace(1.0, -1.0, self.max_iter)
        
        for idx, zeta in enumerate(zeta_values):
            if verbose and idx % 10 == 0:
                print(f"   GNCCP Iteration {idx+1}/{self.max_iter}, ζ = {zeta:.3f}")
            
            for fw_iter in range(self.fw_max_iter):
                grad = self._compute_gradient_Jzeta(X, A_G, A_H, C, zeta)
                Y = self._solve_lp_direction(grad, L)
                
                D_fw = X - Y
                fw_gap = xp.sum(grad * D_fw)
                if fw_gap < self.tol:
                    break
                
                lambda_opt = self._line_search_Jzeta(X, Y, A_G, A_H, C, zeta, L)
                X_new = X + lambda_opt * (Y - X)
                X_new = self._project_to_relaxation(X_new, L)
                X = X_new
        
        # Final Discretization
        X_binary = self._discretize_to_binary(X, L)
        return X_binary
    
    def _compute_gradient_Jzeta(self, X, A_G, A_H, C, zeta):
        xp = self.xp
        grad_F = self._compute_gradient_F(X, A_G, A_H, C)
        if zeta >= 0:
            grad = (1 - zeta) * grad_F + 2 * zeta * X
        else:
            grad = (1 + zeta) * grad_F + 2 * zeta * X
        
        grad = xp.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
        return grad
    
    def _compute_gradient_F(self, X, A_G, A_H, C):
        xp = self.xp
        r = X.sum(axis=1)
        U = xp.outer(r, r)
        
        R = (U * A_G) - (X @ A_H @ X.T)
        RA_X = (R * A_G) @ X
        term1 = RA_X.sum(axis=1, keepdims=True)
        term2 = R @ X @ A_H
        
        grad_struct = 4.0 * term1 - 4.0 * term2
        grad_F = self.alpha * grad_struct + (1 - self.alpha) * C
        return grad_F
    
    def _solve_lp_direction(self, grad, L):
        xp = self.xp
        M, N = grad.shape
        
        # Linear Sum Assignment ONLY runs on CPU via SciPy
        grad_cpu = cp.asnumpy(grad) if self.use_gpu else grad
        row_ind, col_ind = linear_sum_assignment(grad_cpu)
        
        Y = xp.zeros((M, N))
        for r, c in zip(row_ind, col_ind):
            if r < M and c < N:
                Y[r, c] = 1.0
        
        Y = self._project_to_relaxation(Y, L)
        return Y
    
    def _line_search_Jzeta(self, X, Y, A_G, A_H, C, zeta, L):
        D = Y - X
        lambda_opt = 1.0
        J_old = self._compute_Jzeta(X, A_G, A_H, C, zeta)
        
        for _ in range(20):
            X_new = X + lambda_opt * D
            X_new = self._project_to_relaxation(X_new, L)
            J_new = self._compute_Jzeta(X_new, A_G, A_H, C, zeta)
            if J_new < J_old:
                return lambda_opt
            lambda_opt *= 0.5
        return 0.0
    
    def _compute_Jzeta(self, X, A_G, A_H, C, zeta):
        xp = self.xp
        F = self._compute_F(X, A_G, A_H, C)
        x_norm_sq = xp.sum(X ** 2)
        if zeta >= 0:
            J = (1 - zeta) * F + zeta * x_norm_sq
        else:
            J = (1 + zeta) * F + zeta * x_norm_sq
        return J
    
    def _compute_F(self, X, A_G, A_H, C):
        xp = self.xp
        r = X.sum(axis=1)
        U = xp.outer(r, r)
        diff = (U * A_G) - (X @ A_H @ X.T)
        struct_term = xp.sum(diff ** 2)
        app_term = xp.einsum('ij,ij->', C, X)
        F = self.alpha * struct_term + (1 - self.alpha) * app_term
        return F
    
    def _project_to_relaxation(self, X, L):
        xp = self.xp
        X = xp.maximum(X, 0)
        current_sum = X.sum()
        if current_sum > 0:
            X = X * (L / current_sum)
        
        for _ in range(20):
            row_sums = X.sum(axis=1, keepdims=True)
            over_rows = row_sums > 1.0
            X = xp.where(over_rows, X / row_sums, X)
            
            col_sums = X.sum(axis=0, keepdims=True)
            over_cols = col_sums > 1.0
            X = xp.where(over_cols, X / col_sums, X)
            
            current_sum = X.sum()
            if current_sum > 0:
                X = X * (L / current_sum)
            
            if (xp.all(X.sum(axis=1) <= 1.0 + 1e-6) and
                xp.all(X.sum(axis=0) <= 1.0 + 1e-6)):
                break
        
        X = xp.clip(X, 0, 1)
        return X
    
    def _discretize_to_binary(self, X, L):
        xp = self.xp
        M, N = X.shape
        L = min(L, M, N)
        if L == 0:
            return xp.zeros((M, N))
        
        # Hungarian for final assignment (CPU bound)
        C_matrix = -X
        C_cpu = cp.asnumpy(C_matrix) if self.use_gpu else C_matrix
        row_ind, col_ind = linear_sum_assignment(C_cpu)
        
        Y_cpu = np.zeros((M, N))
        for r, c in zip(row_ind, col_ind):
            if r < M and c < N:
                Y_cpu[r, c] = 1
                
        # Ensure exactly L matches (pure python/numpy logic)
        X_cpu = cp.asnumpy(X) if self.use_gpu else X
        current_sum = int(Y_cpu.sum())
        
        if current_sum > L:
            ones = np.where(Y_cpu == 1)
            if len(ones[0]) > 0:
                vals = [(X_cpu[ones[0][k], ones[1][k]], k) for k in range(len(ones[0]))]
                vals.sort(key=lambda x: x[0])
                for _, k in vals[:current_sum - L]:
                    Y_cpu[ones[0][k], ones[1][k]] = 0
        elif current_sum < L:
            zeros = np.where(Y_cpu == 0)
            if len(zeros[0]) > 0:
                vals = [(X_cpu[zeros[0][k], zeros[1][k]], k) for k in range(len(zeros[0]))]
                vals.sort(key=lambda x: x[0], reverse=True)
                added = 0
                for _, k in vals:
                    if added >= L - current_sum:
                        break
                    i, j = zeros[0][k], zeros[1][k]
                    if Y_cpu[i, :].sum() == 0 and Y_cpu[:, j].sum() == 0:
                        Y_cpu[i, j] = 1
                        added += 1
                        
        Y = xp.asarray(Y_cpu) if self.use_gpu else Y_cpu
        return Y

    # ------------------------------------------------------------------
    # Precomputation API — avoids redundant work during hyperparameter sweeps
    # ------------------------------------------------------------------
    @staticmethod
    def precompute_molecular_data(mol):
        instance = WCS_GNCCP.__new__(WCS_GNCCP)
        return {
            'A': instance._build_weighted_adjacency(mol),
            'features': instance._extract_atom_features(mol),
            'n_atoms': mol.GetNumAtoms(),
        }
    
    def match_from_precomputed(self, data1, data2, L=None):
        A_G = data1['A'].copy()
        A_H = data2['A'].copy()
        M, N = data1['n_atoms'], data2['n_atoms']
        
        norm_G = np.linalg.norm(A_G, 'fro')
        norm_H = np.linalg.norm(A_H, 'fro')
        if norm_G > 0: A_G /= norm_G
        if norm_H > 0: A_H /= norm_H
        
        f1, f2 = data1['features'], data2['features']
        C = np.zeros((M, N))
        C += (f1['atomic_num'][:, None] != f2['atomic_num'][None, :]) * 100
        C += np.abs(f1['degree'][:, None] - f2['degree'][None, :]) * 10
        C += (f1['is_in_ring'][:, None] != f2['is_in_ring'][None, :]) * 5
        C += (f1['is_aromatic'][:, None] != f2['is_aromatic'][None, :]) * 10
        C += np.abs(f1['charge'][:, None] - f2['charge'][None, :]) * 20
        C += np.abs(f1['mass'][:, None] - f2['mass'][None, :]) * 0.5
        C += (f1['hybridization'][:, None] != f2['hybridization'][None, :]) * 15
        C += (f1['num_hs'][:, None] != f2['num_hs'][None, :]) * 5
        C += (f1['chiral_tag'][:, None] != f2['chiral_tag'][None, :]) * 5
        C += (f1['isotope'][:, None] != f2['isotope'][None, :]) * 2
        
        if C.max() > C.min():
            C = (C - C.min()) / (C.max() - C.min() + 1e-8)
        else:
            C = C / (C.max() + 1e-8)
        
        if L is None:
            L = max(min(M, N) - 2, 1)
        else:
            L = min(L, M, N)
        
        X = self._gnccp_optimization(A_G, A_H, C, L, verbose=False)
        X_cpu = cp.asnumpy(X) if self.use_gpu else X
        
        score = self._compute_final_score(A_G, A_H, C, X_cpu, L)
        return score
