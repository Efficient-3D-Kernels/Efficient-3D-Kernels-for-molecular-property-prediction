import os
import sys
import time
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = sys.stdout

from wcs_gnccp import WCS_GNCCP
import random

def load_random_mols(sdf_name, num_mols):
    """Loads a random subset of molecules from an SDF file."""
    sdf_path = Path("../Data") / sdf_name
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)
    
    valid_mols = [mol for mol in supplier if mol is not None]
    
    # Shuffle and pick num_mols
    random.seed(42) # For reproducibility
    random.shuffle(valid_mols)
    return valid_mols[:num_mols]

def convert_to_2d(mol_3d):
    mol_2d = Chem.Mol(mol_3d)
    mol_2d.RemoveAllConformers()
    AllChem.Compute2DCoords(mol_2d)
    return mol_2d

def run_benchmark():
    print("="*70)
    print("🧪 SVM KERNEL BENCHMARK (WCS 2D GRAPH KERNEL)")
    print("="*70)
    
    # Configuration
    dataset_prefix = "LBD"  # You have Aromatase, LBD, MMP, p53
    n_actives = 10
    n_inactives = 10
    n_total = n_actives + n_inactives
    
    print(f"📂 Loading Dataset: {dataset_prefix}")
    print(f"   Selecting {n_actives} random Actives and {n_inactives} random Inactives...")
    
    active_mols = load_random_mols(f"{dataset_prefix}_actives_new.sdf", n_actives)
    inactive_mols = load_random_mols(f"{dataset_prefix}_inactives_new.sdf", n_inactives)
    
    mols_2d = [convert_to_2d(m) for m in active_mols] + [convert_to_2d(m) for m in inactive_mols]
    labels = np.array([1]*n_actives + [0]*n_inactives)
    
    print(f"\n⚙️ Computing {n_total}x{n_total} Precomputed Kernel Matrix K...")
    print(f"   (Computing {(n_total * (n_total - 1)) // 2} unique pairs)")
    
    K = np.zeros((n_total, n_total))
    matcher = WCS_GNCCP(alpha=0.7, max_iter=8, fw_max_iter=8)
    
    start_time = time.time()
    for i in range(n_total):
        K[i, i] = 1.0 # Self-similarity
        for j in range(i + 1, n_total):
            mol1, mol2 = mols_2d[i], mols_2d[j]
            L = min(mol1.GetNumAtoms(), mol2.GetNumAtoms()) - 2
            
            _, _, score = matcher.match(mol1, mol2, L=L, verbose=False)
            
            K[i, j] = score
            K[j, i] = score
            
            elapsed = time.time() - start_time
            print(f"   Computed Pair ({i},{j}) -> Sim: {score:.3f} | Elapsed: {elapsed:.1f}s")
            
    print(f"\n✅ Kernel Matrix Computation Complete in {time.time() - start_time:.1f}s")
    
    print("\n🧠 Training Support Vector Machine (SVM)...")
    print("   Method: 5-Fold Stratified Cross-Validation")
    
    # Use an SVM with a precomputed kernel (This is exactly what the paper does!)
    clf = SVC(kernel='precomputed', C=1.0)
    
    # Stratified 5-Fold CV ensures each fold has an equal ratio of Actives/Inactives
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate accuracy scores for all 5 folds
    scores = cross_val_score(clf, K, labels, cv=cv, scoring='accuracy')
    
    print("\n" + "="*70)
    print("🏆 FINAL BENCHMARK RESULTS")
    print("="*70)
    print(f"Dataset: {dataset_prefix} ({n_actives} Actives, {n_inactives} Inactives)")
    print(f"Folds Accuracies: {[f'{s*100:.1f}%' for s in scores]}")
    print("-" * 70)
    print(f"⭐ MEAN ACCURACY: {np.mean(scores) * 100:.2f}% (± {np.std(scores) * 100:.2f}%)")
    print("-" * 70)

if __name__ == "__main__":
    run_benchmark()