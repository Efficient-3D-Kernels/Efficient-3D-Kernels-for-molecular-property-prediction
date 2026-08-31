# wcs_gnccp.py - CORRECTED VERSION
# X remains CONTINUOUS during optimization!
# Only discretized to binary at the END!

import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

class WCS_GNCCP:
    """Weighted Common Subgraph Matching using GNCCP with Frank-Wolfe"""
    
    def __init__(self, alpha=0.7, max_iter=50, fw_max_iter=20, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.fw_max_iter = fw_max_iter
        self.tol = tol
        
    def match(self, mol1, mol2, L=None, verbose=True):
        """Main matching function"""
        if verbose:
            print("="*60)
            print("🔬 WCS+GNCCP Matching (Frank-Wolfe)")
            print("   X remains CONTINUOUS during optimization")
            print("="*60)
        
        # Build weighted adjacency matrices
        A_G = self._build_weighted_adjacency(mol1)
        A_H = self._build_weighted_adjacency(mol2)
        
        M, N = A_G.shape[0], A_H.shape[0]
        
        # Set L
        if L is None:
            L = min(M, N) - 2
            L = max(L, 1)
        else:
            L = min(L, M, N)
        
        if verbose:
            print(f"   Matching {L} atoms between {M} and {N} atoms")
            print(f"   Alpha (structure weight): {self.alpha}")
        
        # Build cost matrix (atom features)
        C = self._build_cost_matrix(mol1, mol2)
        
        # Normalize cost
        if C.max() > C.min():
            C = (C - C.min()) / (C.max() - C.min() + 1e-8)
        else:
            C = C / (C.max() + 1e-8)
        
        # Run GNCCP with Frank-Wolfe
        X = self._gnccp_optimization(A_G, A_H, C, L, verbose)
        
        # Extract matches from binary X
        matches = []
        for i in range(M):
            for j in range(N):
                if X[i, j] > 0.5:
                    matches.append((i, j))
        
        # Ensure exactly L matches
        if len(matches) > L:
            matches = matches[:L]
        elif len(matches) < L:
            remaining = []
            for i in range(M):
                for j in range(N):
                    if (i, j) not in matches and X[i, j] > 0.1:
                        remaining.append((i, j, X[i, j]))
            remaining.sort(key=lambda x: x[2], reverse=True)
            needed = L - len(matches)
            for i, j, _ in remaining[:needed]:
                matches.append((i, j))
        
        # Calculate actual mathematical similarity using the objective F(X)
        M, N = X.shape
        I = np.ones((N, N))
        U = X @ I @ X.T
        struct_diff = U * A_G - X @ A_H @ X.T
        struct_cost = np.sum(struct_diff ** 2)
        app_cost = np.trace(C.T @ X)
        
        # The objective to minimize:
        F_obj = self.alpha * struct_cost + (1 - self.alpha) * app_cost
        
        # Convert distance F_obj to a similarity score [0, 1]
        # We use a scaled exponential to avoid immediate zeroing out
        score = np.exp(-F_obj / (L * 10.0))
        
        if verbose:
            print(f"✅ Found {len(matches)} matches (Sim Score: {score:.3f} | F_obj: {F_obj:.3f})")
            print("="*60)
        
        return X, matches, score
    
    def _build_weighted_adjacency(self, mol):
        """Build weighted adjacency matrix with bond strengths"""
        n = mol.GetNumAtoms()
        A = np.zeros((n, n))
        
        if mol.GetNumConformers() == 0:
            mol2d = Chem.Mol(mol)
            mol2d.RemoveAllConformers()
            AllChem.Compute2DCoords(mol2d)
            conf = mol2d.GetConformer()
        else:
            conf = mol.GetConformer()
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            bt = bond.GetBondType()
            if bt == Chem.BondType.SINGLE:
                w = 1.0
            elif bt == Chem.BondType.DOUBLE:
                w = 2.0
            elif bt == Chem.BondType.TRIPLE:
                w = 3.0
            elif bt == Chem.BondType.AROMATIC:
                w = 1.5
            else:
                w = 1.0
                
            # Add micro-weights for 2D bond topology
            if bond.IsInRing():
                w += 0.1
            if bond.GetIsConjugated():
                w += 0.1
            
            pos_i = conf.GetAtomPosition(i)
            pos_j = conf.GetAtomPosition(j)
            dist = np.sqrt((pos_i.x - pos_j.x)**2 + 
                          (pos_i.y - pos_j.y)**2 + 
                          (pos_i.z - pos_j.z)**2)
            
            A[i, j] = w * (1.0 + 0.1 * dist)
            A[j, i] = w * (1.0 + 0.1 * dist)
        
        return A
    
    def _build_cost_matrix(self, mol1, mol2):
        """Build cost matrix from atom features"""
        M = mol1.GetNumAtoms()
        N = mol2.GetNumAtoms()
        C = np.zeros((M, N))
        
        features1 = self._extract_atom_features(mol1)
        features2 = self._extract_atom_features(mol2)
        
        for i in range(M):
            for j in range(N):
                f1 = features1[i]
                f2 = features2[j]
                
                if f1['atomic_num'] != f2['atomic_num']:
                    C[i, j] += 100
                C[i, j] += abs(f1['degree'] - f2['degree']) * 10
                if f1['is_in_ring'] != f2['is_in_ring']:
                    C[i, j] += 5
                if f1['is_aromatic'] != f2['is_aromatic']:
                    C[i, j] += 10
                C[i, j] += abs(f1['charge'] - f2['charge']) * 20
                
                # Incorporate remaining 2D features: mass and hybridization
                C[i, j] += abs(f1['mass'] - f2['mass']) * 0.5
                if f1['hybridization'] != f2['hybridization']:
                    C[i, j] += 15
                    
                # Incorporate exhaustive 2D features
                if f1['num_hs'] != f2['num_hs']:
                    C[i, j] += 5
                if f1['chiral_tag'] != f2['chiral_tag']:
                    C[i, j] += 5
                if f1['isotope'] != f2['isotope']:
                    C[i, j] += 2
        
        return C
    
    def _extract_atom_features(self, mol):
        """Extract atom features"""
        features = []
        for atom in mol.GetAtoms():
            features.append({
                'atomic_num': atom.GetAtomicNum(),
                'degree': atom.GetDegree(),
                'is_in_ring': atom.IsInRing(),
                'is_aromatic': atom.GetIsAromatic(),
                'mass': atom.GetMass(),
                'charge': atom.GetFormalCharge(),
                'hybridization': str(atom.GetHybridization()),
                'num_hs': atom.GetTotalNumHs(),
                'chiral_tag': str(atom.GetChiralTag()),
                'isotope': atom.GetIsotope()
            })
        return features
    
    def _gnccp_optimization(self, A_G, A_H, C, L, verbose=True):
        """
        GNCCP with Frank-Wolfe
        
        KEY: X remains CONTINUOUS during optimization!
        """
        M, N = A_G.shape[0], A_H.shape[0]
        
        # Initialize X as continuous relaxation
        X = np.ones((M, N)) * (L / (M * N))
        
        # GNCCP path: ζ from 1 to -1
        zeta_values = np.linspace(1.0, -1.0, self.max_iter)
        
        for idx, zeta in enumerate(zeta_values):
            if verbose and idx % 10 == 0:
                print(f"   GNCCP Iteration {idx+1}/{self.max_iter}, ζ = {zeta:.3f}")
            
            # Frank-Wolfe inner loop
            for fw_iter in range(self.fw_max_iter):
                grad = self._compute_gradient_Jzeta(X, A_G, A_H, C, zeta)
                Y = self._solve_lp_direction(grad, L)
                lambda_opt = self._line_search_Jzeta(X, Y, A_G, A_H, C, zeta)
                
                X_new = X + lambda_opt * (Y - X)
                
                # Project to RELAXATION D (continuous, NOT binary!)
                X_new = self._project_to_relaxation(X_new, L)
                
                if np.linalg.norm(X_new - X) < self.tol:
                    X = X_new
                    break
                
                X = X_new
        
        # FINAL: Discretize to binary partial permutation
        X_binary = self._discretize_to_binary(X, L)
        
        return X_binary
    
    def _compute_gradient_Jzeta(self, X, A_G, A_H, C, zeta):
        """Compute gradient of J_ζ(X) (Equation 6)"""
        grad_F = self._compute_gradient_F(X, A_G, A_H, C)
        
        if zeta >= 0:
            grad = (1 - zeta) * grad_F + 2 * zeta * X
        else:
            grad = (1 + zeta) * grad_F + 2 * zeta * X
        
        grad = np.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
        return grad
    
    def _compute_gradient_F(self, X, A_G, A_H, C):
        """
        Compute gradient of FULL objective F(X)
        Includes BOTH structure and appearance!
        """
        M, N = X.shape
        
        # U = X * I * X^T
        I = np.ones((N, N))
        U = X @ I @ X.T
        
        # Structural term
        diff = (U * A_G) - (X @ A_H @ X.T)
        grad_struct = -2 * (A_G.T @ diff @ X @ A_H.T + A_G @ diff @ X @ A_H)
        
        # Appearance term
        grad_app = (1 - self.alpha) * C
        
        # Full gradient
        grad_F = self.alpha * grad_struct + grad_app
        
        return grad_F
    
    def _solve_lp_direction(self, grad, L):
        """Solve LP for direction Y (Equation 5)"""
        M, N = grad.shape
        
        row_ind, col_ind = linear_sum_assignment(grad)
        
        Y = np.zeros((M, N))
        for r, c in zip(row_ind, col_ind):
            if r < M and c < N:
                Y[r, c] = 1.0
        
        Y = self._project_to_relaxation(Y, L)
        return Y
    
    def _line_search_Jzeta(self, X, Y, A_G, A_H, C, zeta):
        """Line search (Equation 8)"""
        D = Y - X
        lambda_opt = 1.0
        
        for _ in range(20):
            X_new = X + lambda_opt * D
            X_new = self._project_to_relaxation(X_new, int(X.sum()))
            
            J_new = self._compute_Jzeta(X_new, A_G, A_H, C, zeta)
            J_old = self._compute_Jzeta(X, A_G, A_H, C, zeta)
            
            if J_new < J_old:
                break
            
            lambda_opt *= 0.5
        
        return max(lambda_opt, 0.01)
    
    def _compute_Jzeta(self, X, A_G, A_H, C, zeta):
        """Compute J_ζ(X)"""
        F = self._compute_F(X, A_G, A_H, C)
        
        if zeta >= 0:
            J = (1 - zeta) * F + zeta * np.trace(X.T @ X)
        else:
            J = (1 + zeta) * F + zeta * np.trace(X.T @ X)
        
        return J
    
    def _compute_F(self, X, A_G, A_H, C):
        """Compute FULL objective F(X) (Equation 2)"""
        M, N = X.shape
        
        I = np.ones((N, N))
        U = X @ I @ X.T
        
        diff = (U * A_G) - (X @ A_H @ X.T)
        struct_term = np.linalg.norm(diff, 'fro') ** 2
        app_term = np.trace(C.T @ X)
        
        F = self.alpha * struct_term + (1 - self.alpha) * app_term
        return F
    
    def _project_to_relaxation(self, X, L):
        """
        Project X to convex hull D (continuous relaxation)
        X remains CONTINUOUS here.
        
        D = {X | row sums ≤ 1, col sums ≤ 1, total sum = L, X ≥ 0}
        """
        M, N = X.shape
        
        # Ensure non-negative
        X = np.maximum(X, 0)
        
        # Scale to total sum = L
        current_sum = X.sum()
        if current_sum > 0:
            X = X * (L / current_sum)
        
        # Project rows
        for i in range(M):
            row_sum = X[i, :].sum()
            if row_sum > 1.0:
                X[i, :] = X[i, :] / row_sum
        
        # Project columns
        for j in range(N):
            col_sum = X[:, j].sum()
            if col_sum > 1.0:
                X[:, j] = X[:, j] / col_sum
        
        # Re-normalize
        current_sum = X.sum()
        if current_sum > 0:
            X = X * (L / current_sum)
        
        # Clamp to [0, 1]
        X = np.clip(X, 0, 1)
        
        return X
    
    def _discretize_to_binary(self, X, L):
        """
        Convert continuous X to binary partial permutation
        ONLY called at the END of GNCCP optimization!
        """
        M, N = X.shape
        L = min(L, M, N)
        
        if L == 0:
            return np.zeros((M, N))
        
        # Hungarian for final assignment
        C = -X
        row_ind, col_ind = linear_sum_assignment(C)
        
        Y = np.zeros((M, N))
        for r, c in zip(row_ind, col_ind):
            if r < M and c < N:
                Y[r, c] = 1
        
        # Ensure exactly L matches
        current_sum = int(Y.sum())
        if current_sum > L:
            ones = np.where(Y == 1)
            if len(ones[0]) > 0:
                vals = [(X[ones[0][k], ones[1][k]], k) for k in range(len(ones[0]))]
                vals.sort(key=lambda x: x[0])
                for _, k in vals[:current_sum - L]:
                    Y[ones[0][k], ones[1][k]] = 0
        elif current_sum < L:
            zeros = np.where(Y == 0)
            if len(zeros[0]) > 0:
                vals = [(X[zeros[0][k], zeros[1][k]], k) for k in range(len(zeros[0]))]
                vals.sort(key=lambda x: x[0], reverse=True)
                added = 0
                for _, k in vals:
                    if added >= L - current_sum:
                        break
                    i, j = zeros[0][k], zeros[1][k]
                    if Y[i, :].sum() == 0 and Y[:, j].sum() == 0:
                        Y[i, j] = 1
                        added += 1
        
        return Y
    
    def _is_binary(self, X, tol=1e-3):
        """Check if X is binary"""
        return np.all((X < tol) | (X > 1 - tol))
    
    def compute_similarity_score(self, mol1, mol2, L=None):
        """Convenience method"""
        _, _, score = self.match(mol1, mol2, L, verbose=False)
        return score