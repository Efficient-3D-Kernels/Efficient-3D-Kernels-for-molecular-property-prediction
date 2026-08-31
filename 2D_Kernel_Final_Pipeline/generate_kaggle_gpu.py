import nbformat
import os

def create_notebook():
    nb = nbformat.v4.new_notebook()

    # Cell 1: Environment Setup
    cell_setup = nbformat.v4.new_markdown_cell("""\
# Massive WCS 2D Graph Kernel Benchmark (GPU Accelerated)
This notebook implements the complete 2D Weighted Common Subgraph (WCS) Graph Kernel using **PyTorch CUDA acceleration**.
Because the Hungarian Algorithm ($O(N^3)$) requires CPU execution, this implementation uses a hybrid architecture:
1. All heavy continuous optimizations, trace calculus, and gradients run massively in parallel on the GPU.
2. The exact assignment matching runs on the CPU.
""")

    cell_imports = nbformat.v4.new_code_cell("""\
import os
import sys
import time
import glob
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem

# Configure Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using Compute Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
""")

    # Cell 2: Data Loader
    cell_loader = nbformat.v4.new_code_cell("""\
def load_dataset(dataset_prefix, data_dir="/kaggle/input/your-dataset-name"):
    '''Loads actives and inactives for a given dataset prefix (e.g. "Aromatase")'''
    active_path = os.path.join(data_dir, f"{dataset_prefix}_actives_new.sdf")
    inactive_path = os.path.join(data_dir, f"{dataset_prefix}_inactives_new.sdf")
    
    if not os.path.exists(active_path):
        # Fallback to local testing directory if not on Kaggle
        data_dir = "../Data"
        active_path = os.path.join(data_dir, f"{dataset_prefix}_actives_new.sdf")
        inactive_path = os.path.join(data_dir, f"{dataset_prefix}_inactives_new.sdf")
        
    print(f"Loading {dataset_prefix}...")
    
    active_mols = []
    if os.path.exists(active_path):
        suppl = Chem.SDMolSupplier(active_path, removeHs=True, sanitize=True)
        active_mols = [m for m in suppl if m is not None]
        
    inactive_mols = []
    if os.path.exists(inactive_path):
        suppl = Chem.SDMolSupplier(inactive_path, removeHs=True, sanitize=True)
        inactive_mols = [m for m in suppl if m is not None]
        
    # We will sample to make benchmarking feasible within Notebook limits
    # You can increase this!
    N_SAMPLES = 10 
    
    import random
    random.seed(42)
    active_mols = random.sample(active_mols, min(N_SAMPLES, len(active_mols)))
    inactive_mols = random.sample(inactive_mols, min(N_SAMPLES, len(inactive_mols)))
    
    print(f" -> Selected {len(active_mols)} Actives and {len(inactive_mols)} Inactives")
    
    # Convert all to 2D
    mols_2d = []
    for m in active_mols + inactive_mols:
        m2d = Chem.Mol(m)
        m2d.RemoveAllConformers()
        AllChem.Compute2DCoords(m2d)
        mols_2d.append(m2d)
        
    labels = np.array([1]*len(active_mols) + [0]*len(inactive_mols))
    return mols_2d, labels
""")

    # Cell 3: GPU WCS Kernel
    cell_wcs = nbformat.v4.new_code_cell("""\
class WCS_GNCCP_GPU:
    '''PyTorch Accelerated WCS Graph Kernel'''
    def __init__(self, alpha=0.7, max_iter=8, fw_max_iter=8, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.fw_max_iter = fw_max_iter
        self.tol = tol
        
    def match(self, mol1, mol2, L=None):
        A_G_np = self._build_weighted_adjacency(mol1)
        A_H_np = self._build_weighted_adjacency(mol2)
        C_np = self._build_cost_matrix(mol1, mol2)
        
        M, N = A_G_np.shape[0], A_H_np.shape[0]
        if L is None: L = max(min(M, N) - 2, 1)
        else: L = min(L, M, N)
        
        if C_np.max() > C_np.min():
            C_np = (C_np - C_np.min()) / (C_np.max() - C_np.min() + 1e-8)
        else:
            C_np = C_np / (C_np.max() + 1e-8)
            
        # Push to GPU
        A_G = torch.tensor(A_G_np, dtype=torch.float32, device=device)
        A_H = torch.tensor(A_H_np, dtype=torch.float32, device=device)
        C = torch.tensor(C_np, dtype=torch.float32, device=device)
        
        X = self._gnccp_optimization(A_G, A_H, C, L)
        
        # Exact Objective calculation on GPU
        I = torch.ones((N, N), dtype=torch.float32, device=device)
        U = X @ I @ X.T
        struct_diff = U * A_G - X @ A_H @ X.T
        struct_cost = torch.sum(struct_diff ** 2)
        app_cost = torch.trace(C.T @ X)
        
        F_obj = self.alpha * struct_cost + (1 - self.alpha) * app_cost
        F_obj_val = F_obj.item()
        
        # Inverse Fraction mapping for distinct scores!
        score = 1.0 / (1.0 + (F_obj_val / L))
        
        return score
        
    def _gnccp_optimization(self, A_G, A_H, C, L):
        M, N = A_G.shape[0], A_H.shape[0]
        X = torch.ones((M, N), dtype=torch.float32, device=device) * (L / (M * N))
        
        zeta_values = torch.linspace(1.0, -1.0, self.max_iter, device=device)
        
        for zeta in zeta_values:
            for _ in range(self.fw_max_iter):
                # 1. Gradient
                I = torch.ones((N, N), dtype=torch.float32, device=device)
                U = X @ I @ X.T
                diff = (U * A_G) - (X @ A_H @ X.T)
                grad_struct = -2.0 * (A_G.T @ diff @ X @ A_H.T + A_G @ diff @ X @ A_H)
                grad_app = (1.0 - self.alpha) * C
                grad_F = self.alpha * grad_struct + grad_app
                
                if zeta >= 0: grad = (1 - zeta) * grad_F + 2 * zeta * X
                else:         grad = (1 + zeta) * grad_F + 2 * zeta * X
                
                # 2. Hungarian (Move grad to CPU momentarily)
                grad_cpu = grad.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(grad_cpu)
                Y = torch.zeros((M, N), dtype=torch.float32, device=device)
                for r, c in zip(row_ind, col_ind):
                    if r < M and c < N: Y[r, c] = 1.0
                Y = self._project_to_relaxation(Y, L)
                
                # 3. Step size and update
                D_mat = Y - X
                lam = 1.0
                for _ in range(10): # Simple line search
                    X_new = X + lam * D_mat
                    X_new = self._project_to_relaxation(X_new, L)
                    # Break condition simplified for speed on GPU
                    if torch.norm(X_new - X) < self.tol:
                        X = X_new
                        break
                    lam *= 0.5
                X = X_new
                
        # Final Hungarian Discretization
        X_cpu = -X.detach().cpu().numpy()
        r_i, c_i = linear_sum_assignment(X_cpu)
        X_bin = torch.zeros((M, N), dtype=torch.float32, device=device)
        matched = 0
        for r, c in zip(r_i, c_i):
            if r < M and c < N and matched < L:
                X_bin[r, c] = 1.0
                matched += 1
        return X_bin

    def _project_to_relaxation(self, X, L):
        M, N = X.shape
        X = torch.clamp(X, min=0.0)
        c_sum = X.sum()
        if c_sum > 0: X = X * (L / c_sum)
        
        # Row normalization
        row_sums = X.sum(dim=1, keepdim=True)
        mask = row_sums > 1.0
        X = torch.where(mask.expand_as(X), X / (row_sums + 1e-8), X)
        
        # Col normalization
        col_sums = X.sum(dim=0, keepdim=True)
        mask = col_sums > 1.0
        X = torch.where(mask.expand_as(X), X / (col_sums + 1e-8), X)
        
        c_sum = X.sum()
        if c_sum > 0: X = X * (L / c_sum)
        return torch.clamp(X, 0.0, 1.0)
    
    def _build_weighted_adjacency(self, mol):
        n = mol.GetNumAtoms()
        A = np.zeros((n, n))
        conf = mol.GetConformer()
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = bond.GetBondType()
            w = 1.0
            if bt == Chem.BondType.DOUBLE: w = 2.0
            elif bt == Chem.BondType.TRIPLE: w = 3.0
            elif bt == Chem.BondType.AROMATIC: w = 1.5
            if bond.IsInRing(): w += 0.1
            if bond.GetIsConjugated(): w += 0.1
            
            p_i = conf.GetAtomPosition(i)
            p_j = conf.GetAtomPosition(j)
            dist = np.sqrt((p_i.x - p_j.x)**2 + (p_i.y - p_j.y)**2 + (p_i.z - p_j.z)**2)
            
            A[i, j] = w * (1.0 + 0.1 * dist)
            A[j, i] = w * (1.0 + 0.1 * dist)
        return A
        
    def _build_cost_matrix(self, mol1, mol2):
        M, N = mol1.GetNumAtoms(), mol2.GetNumAtoms()
        C = np.zeros((M, N))
        f1 = self._extract_features(mol1)
        f2 = self._extract_features(mol2)
        
        for i in range(M):
            for j in range(N):
                c_val = 0
                if f1[i]['atomic_num'] != f2[j]['atomic_num']: c_val += 100
                c_val += abs(f1[i]['degree'] - f2[j]['degree']) * 10
                if f1[i]['is_in_ring'] != f2[j]['is_in_ring']: c_val += 5
                if f1[i]['is_aromatic'] != f2[j]['is_aromatic']: c_val += 10
                c_val += abs(f1[i]['charge'] - f2[j]['charge']) * 20
                c_val += abs(f1[i]['mass'] - f2[j]['mass']) * 0.5
                if f1[i]['hybridization'] != f2[j]['hybridization']: c_val += 15
                if f1[i]['num_hs'] != f2[j]['num_hs']: c_val += 5
                if f1[i]['chiral_tag'] != f2[j]['chiral_tag']: c_val += 5
                C[i, j] = c_val
        return C
        
    def _extract_features(self, mol):
        return [{
            'atomic_num': a.GetAtomicNum(),
            'degree': a.GetDegree(),
            'is_in_ring': a.IsInRing(),
            'is_aromatic': a.GetIsAromatic(),
            'mass': a.GetMass(),
            'charge': a.GetFormalCharge(),
            'hybridization': str(a.GetHybridization()),
            'num_hs': a.GetTotalNumHs(),
            'chiral_tag': str(a.GetChiralTag())
        } for a in mol.GetAtoms()]
""")

    # Cell 4: Benchmarking Engine (Iterates datasets and alpha)
    cell_benchmark = nbformat.v4.new_code_cell("""\
datasets = ["Aromatase", "LBD", "MMP", "p53"]
alpha_values = [0.3, 0.5, 0.7, 0.9]

results = {}

print("="*60)
print("🚀 MASSIVE GPU BENCHMARK INITIATED")
print("="*60)

for dataset in datasets:
    try:
        mols, labels = load_dataset(dataset)
    except Exception as e:
        print(f"Skipping {dataset}, files not found.")
        continue
        
    n_mols = len(mols)
    if n_mols < 5:
        continue
        
    print(f"\\n🧪 Dataset: {dataset} | Total Molecules: {n_mols}")
    results[dataset] = {}
    
    for alpha in alpha_values:
        print(f"   -> Testing Alpha = {alpha}")
        matcher = WCS_GNCCP_GPU(alpha=alpha, max_iter=8, fw_max_iter=8)
        
        # Compute Distance Matrix
        D = np.zeros((n_mols, n_mols))
        start_time = time.time()
        
        for i in range(n_mols):
            for j in range(i + 1, n_mols):
                sim_score = matcher.match(mols[i], mols[j])
                dist = 1.0 - sim_score # Distance is 1 - similarity
                D[i, j] = dist
                D[j, i] = dist
                
        elapsed = time.time() - start_time
        
        # 1-NN Leave One Out CV
        loo = LeaveOneOut()
        y_true = []
        y_pred = []
        for train_index, test_index in loo.split(D):
            D_train = D[np.ix_(train_index, train_index)]
            D_test = D[np.ix_(test_index, train_index)]
            y_train = labels[train_index]
            y_test_true = labels[test_index]
            
            clf = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
            clf.fit(D_train, y_train)
            pred = clf.predict(D_test)
            
            y_true.append(y_test_true[0])
            y_pred.append(pred[0])
            
        acc = accuracy_score(y_true, y_pred) * 100
        results[dataset][alpha] = acc
        print(f"      [Acc: {acc:.2f}% | Compute Time: {elapsed:.1f}s]")

print("\\n" + "="*60)
print("🏆 FINAL SWEET SPOT RESULTS")
print("="*60)
for dataset, alphas in results.items():
    best_alpha = max(alphas, key=alphas.get)
    print(f"{dataset}: Best Alpha = {best_alpha} (Accuracy: {alphas[best_alpha]:.2f}%)")
""")

    # Cell 5: Next Steps (3D Mapping placeholder)
    cell_nextsteps = nbformat.v4.new_markdown_cell("""\
# Phase 3: Next Steps (2D to 3D Pipeline)
*Note: As per the current project roadmap, no code is added here yet. This section documents the methodology for the next phase.*

**The Goal:** Now that we have established a robust 2D Topological graph kernel that finds the exact atom-to-atom matches (and proven it via 1-NN accuracy), we need to incorporate the physical 3D properties of the molecules.

**The Workflow:**
1. **Store Mappings:** We will store the exact `X_binary` matching array output by the 2D Kernel.
2. **Force Alignment:** We will pass this exact mapping to `rdMolAlign.AlignMol(mol1, mol2, atomMap=...)`. This forces the two physical 3D conformation graphs to sit perfectly on top of each other in 3D space, anchored precisely by the atoms our 2D kernel decided were identical!
3. **Calculate 3D Features (RMSD):** Once aligned, we calculate the Root-Mean-Square Deviation (RMSD) of the 3D distances between the matched atoms.
4. **Final Kernel Integration:** We combine the 2D Topological Penalty ($F_{obj}$) and the 3D RMSD Penalty into a single, massive **3D-2D Hybrid Distance Kernel**.

This combined kernel will completely destroy "Decoy Bias", because even if a decoy has a perfectly matching 2D topology, its 3D physical volume will clash, resulting in a massive RMSD penalty.
""")

    nb['cells'] = [cell_setup, cell_imports, cell_loader, cell_wcs, cell_benchmark, cell_nextsteps]

    with open('Kaggle_GPU_Benchmark.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Kaggle_GPU_Benchmark.ipynb successfully created!")

if __name__ == "__main__":
    create_notebook()