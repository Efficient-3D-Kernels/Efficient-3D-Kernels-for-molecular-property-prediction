# wcs_gnccp.py - Fixed and Optimized
import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

class WCS_GNCCP:
    """
    Weighted Common Subgraph Matching using GNCCP
    Optimized version with bug fixes
    """
    
    def __init__(self, alpha=0.7, max_iter=50, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        
    def match(self, mol1, mol2, L=None, verbose=True):
        """Main matching function"""
        if verbose:
            print("="*60)
            print("🔬 WCS+GNCCP Matching")
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
            L = min(L, M, N)  # Ensure L doesn't exceed dimensions
        
        if verbose:
            print(f"   Matching {L} atoms between {M} and {N} atoms")
        
        # Build cost matrix
        C = self._build_cost_matrix(mol1, mol2)
        
        # Normalize cost
        if C.max() > C.min():
            C = (C - C.min()) / (C.max() - C.min() + 1e-8)
        else:
            C = C / (C.max() + 1e-8)
        
        # Run GNCCP
        X = self._gnccp_optimization(A_G, A_H, C, L, verbose)
        
        # Extract matches
        matches = []
        for i in range(M):
            for j in range(N):
                if X[i, j] > 0.5:
                    matches.append((i, j))
        
        # Ensure we have exactly L matches
        if len(matches) > L:
            # Keep only L best matches
            matches = matches[:L]
        elif len(matches) < L:
            # Add more matches if possible
            # Get remaining candidates
            remaining = []
            for i in range(M):
                for j in range(N):
                    if (i, j) not in matches and X[i, j] > 0.1:
                        remaining.append((i, j, X[i, j]))
            remaining.sort(key=lambda x: x[2], reverse=True)
            needed = L - len(matches)
            for i, j, _ in remaining[:needed]:
                matches.append((i, j))
        
        score = len(matches) / L if L > 0 else 0
        
        if verbose:
            print(f"✅ Found {len(matches)} matches (Score: {score:.3f})")
            print("="*60)
        
        return X, matches, score
    
    def _build_weighted_adjacency(self, mol):
        """Build weighted adjacency matrix"""
        n = mol.GetNumAtoms()
        A = np.zeros((n, n))
        
        # Get coordinates
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
            
            # Bond type weight
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
            
            # Distance
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
                
                # Atomic number
                if f1['atomic_num'] != f2['atomic_num']:
                    C[i, j] += 100
                
                # Degree
                C[i, j] += abs(f1['degree'] - f2['degree']) * 10
                
                # Ring membership
                if f1['is_in_ring'] != f2['is_in_ring']:
                    C[i, j] += 5
                
                # Aromaticity
                if f1['is_aromatic'] != f2['is_aromatic']:
                    C[i, j] += 10
        
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
            })
        return features
    
    def _gnccp_optimization(self, A_G, A_H, C, L, verbose=True):
        """GNCCP optimization with fixed projection"""
        M, N = A_G.shape[0], A_H.shape[0]
        
        # Initialize X
        X = np.ones((M, N)) * (L / (M * N))
        
        # GNCCP path
        zeta_values = np.linspace(1.0, -1.0, self.max_iter)
        
        for idx, zeta in enumerate(zeta_values):
            if verbose and idx % 10 == 0:
                print(f"   Iteration {idx+1}/{self.max_iter}, ζ = {zeta:.3f}")
            
            # Compute gradient
            grad = self._compute_gradient(X, A_G, A_H, C, zeta)
            
            # Find direction
            Y = self._solve_linear_programming(grad, L)
            
            # Line search
            lambda_opt = self._line_search(X, Y, A_G, A_H, C, zeta)
            
            # Update
            X_new = X + lambda_opt * (Y - X)
            
            # Project
            X_new = self._project_to_partial_permutation_fast(X_new, L)
            
            # Check convergence
            if np.linalg.norm(X_new - X) < self.tol:
                if verbose:
                    print(f"   Converged at iteration {idx+1}")
                X = X_new
                break
            
            X = X_new
        
        # Final discretization
        X = self._discretize_fast(X, L)
        
        return X
    
    def _compute_gradient(self, X, A_G, A_H, C, zeta):
        """Compute gradient"""
        M, N = X.shape
        
        # U = X * I * X^T
        I = np.ones((N, N))
        U = X @ I @ X.T
        
        # Structural term
        diff = (U * A_G) - (X @ A_H @ X.T)
        
        # Gradient of structural term
        grad_struct = -2 * (A_G.T @ diff @ X @ A_H.T + A_G @ diff @ X @ A_H)
        
        # Gradient of appearance term
        grad_app = (1 - self.alpha) * C
        
        # Full gradient
        grad_F = self.alpha * grad_struct + grad_app
        
        # GNCCP scaling
        if zeta >= 0:
            grad = (1 - zeta) * grad_F + 2 * zeta * X
        else:
            grad = (1 + zeta) * grad_F + 2 * zeta * X
        
        # Handle NaN/Inf
        grad = np.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return grad
    
    def _solve_linear_programming(self, grad, L):
        """Solve LP using Hungarian algorithm"""
        M, N = grad.shape
        
        # Use Hungarian
        row_ind, col_ind = linear_sum_assignment(grad)
        
        # Build Y
        Y = np.zeros((M, N))
        for r, c in zip(row_ind, col_ind):
            if r < M and c < N:
                Y[r, c] = 1
        
        # Ensure L matches
        Y = self._project_to_partial_permutation_fast(Y, L)
        
        return Y
    
    def _line_search(self, X, Y, A_G, A_H, C, zeta):
        """Line search with backtracking"""
        D = Y - X
        lambda_opt = 1.0
        
        for _ in range(20):
            X_new = X + lambda_opt * D
            X_new = self._project_to_partial_permutation_fast(X_new, int(X.sum()))
            
            # Check if valid
            if self._is_valid(X_new):
                break
            
            lambda_opt *= 0.5
        
        return max(lambda_opt, 0.01)
    
    def _is_valid(self, X):
        """Check if X is valid"""
        if np.any(X < -1e-6) or np.any(np.isnan(X)):
            return False
        if np.any(X.sum(axis=1) > 1.0 + 1e-4):
            return False
        if np.any(X.sum(axis=0) > 1.0 + 1e-4):
            return False
        return True
    
    def _project_to_partial_permutation_fast(self, X, L):
        """
        Fast projection to partial permutation matrices
        Uses greedy selection instead of sorting all elements
        """
        M, N = X.shape
        
        # Ensure non-negative
        X = np.maximum(X, 0)
        
        # If L is larger than matrix size, adjust
        L = min(L, M * N)
        
        # Flatten and find top L
        flat = X.flatten()
        
        # Use argpartition for O(n) instead of O(n log n)
        if L < len(flat):
            threshold = np.partition(flat, -L)[-L]
            Y = np.zeros_like(X)
            Y[X >= threshold] = 1
        else:
            Y = (X > 0).astype(float)
        
        # Enforce row/column constraints
        # Row constraints
        for i in range(M):
            row_ones = np.where(Y[i, :] == 1)[0]
            if len(row_ones) > 1:
                # Keep only highest value
                keep_idx = row_ones[np.argmax(X[i, row_ones])]
                Y[i, :] = 0
                Y[i, keep_idx] = 1
        
        # Column constraints
        for j in range(N):
            col_ones = np.where(Y[:, j] == 1)[0]
            if len(col_ones) > 1:
                keep_idx = col_ones[np.argmax(X[col_ones, j])]
                Y[:, j] = 0
                Y[keep_idx, j] = 1
        
        # Adjust total sum to L
        current_sum = int(Y.sum())
        if current_sum > L:
            # Remove extra ones
            ones = np.where(Y == 1)
            if len(ones[0]) > 0:
                # Remove from lowest values
                vals = [(X[ones[0][k], ones[1][k]], k) for k in range(len(ones[0]))]
                vals.sort(key=lambda x: x[0])
                for _, k in vals[:current_sum - L]:
                    Y[ones[0][k], ones[1][k]] = 0
        elif current_sum < L and L > 0:
            # Add more ones
            zeros = np.where(Y == 0)
            if len(zeros[0]) > 0:
                # Add to highest values
                vals = [(X[zeros[0][k], zeros[1][k]], k) for k in range(len(zeros[0]))]
                vals.sort(key=lambda x: x[0], reverse=True)
                added = 0
                for val, k in vals:
                    if added >= L - current_sum:
                        break
                    i, j = zeros[0][k], zeros[1][k]
                    if Y[i, :].sum() == 0 and Y[:, j].sum() == 0:
                        Y[i, j] = 1
                        added += 1
        
        return Y
    
    def _discretize_fast(self, X, L):
        """Fast discretization"""
        M, N = X.shape
        
        # Use Hungarian for final assignment
        C = -X  # Maximize X, so minimize -X
        row_ind, col_ind = linear_sum_assignment(C)
        
        Y = np.zeros((M, N))
        for r, c in zip(row_ind, col_ind):
            if r < M and c < N:
                Y[r, c] = 1
        
        # Ensure L matches
        Y = self._project_to_partial_permutation_fast(Y, L)
        
        return Y

    def compute_similarity_score(self, mol1, mol2, L=None):
        """Convenience method"""
        _, _, score = self.match(mol1, mol2, L, verbose=False)
        return score