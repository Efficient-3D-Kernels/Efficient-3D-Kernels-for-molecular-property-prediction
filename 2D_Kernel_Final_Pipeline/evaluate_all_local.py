import os
import sys
import time
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = sys.stdout

from wcs_gnccp import WCS_GNCCP

def load_mols(sdf_name, max_mols):
    sdf_path = Path("../Data") / sdf_name
    if not sdf_path.exists():
        return []
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)
    mols = []
    for mol in supplier:
        if mol is not None:
            mols.append(mol)
            if len(mols) == max_mols:
                break
    return mols

def convert_to_2d(mol_3d):
    mol_2d = Chem.Mol(mol_3d)
    mol_2d.RemoveAllConformers()
    AllChem.Compute2DCoords(mol_2d)
    return mol_2d

def run_all_evaluation():
    datasets = ["Aromatase", "LBD", "MMP", "p53"]
    n_per_class = 10 # 10 actives, 10 inactives = 20 total per dataset
    
    print("="*60)
    print(f"🚀 MULTI-DATASET 2D KERNEL BENCHMARK (LOCAL CPU)")
    print(f"   Testing: {datasets}")
    print(f"   Sample: {n_per_class} Actives + {n_per_class} Inactives per dataset")
    print("="*60)
    
    results = {}
    matcher = WCS_GNCCP(alpha=0.7, max_iter=8, fw_max_iter=8) 
    
    for dataset in datasets:
        print(f"\n📂 Processing Dataset: {dataset}...")
        
        act_name = f"{dataset}_actives_new.sdf"
        inact_name = f"{dataset}_inactives_new.sdf"
        
        active_mols = load_mols(act_name, max_mols=n_per_class)
        inactive_mols = load_mols(inact_name, max_mols=n_per_class)
        
        if len(active_mols) == 0 or len(inactive_mols) == 0:
            print(f"   [!] Skipping {dataset} (files not found or empty).")
            continue
            
        mols_2d = [convert_to_2d(m) for m in active_mols] + [convert_to_2d(m) for m in inactive_mols]
        labels = np.array([1]*len(active_mols) + [0]*len(inactive_mols))
        n_total = len(mols_2d)
        
        print(f"   -> Loaded {n_total} molecules. Computing {n_total}x{n_total} Kernel...")
        
        D = np.zeros((n_total, n_total))
        start_time = time.time()
        
        for i in range(n_total):
            D[i, i] = 0.0 
            for j in range(i + 1, n_total):
                mol1, mol2 = mols_2d[i], mols_2d[j]
                L = min(mol1.GetNumAtoms(), mol2.GetNumAtoms()) - 2
                _, _, sim_score = matcher.match(mol1, mol2, L=L, verbose=False)
                
                dist = 1.0 - sim_score
                D[i, j] = dist
                D[j, i] = dist 
                
        elapsed = time.time() - start_time
        print(f"   -> Kernel Computation Complete in {elapsed:.1f}s")
        
        loo = LeaveOneOut()
        y_true = []
        y_pred = []
        
        for train_index, test_index in loo.split(D):
            D_train = D[np.ix_(train_index, train_index)]
            D_test = D[np.ix_(test_index, train_index)]
            y_train = labels[train_index]
            
            clf = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
            clf.fit(D_train, y_train)
            pred = clf.predict(D_test)
            
            y_true.append(labels[test_index][0])
            y_pred.append(pred[0])
            
        accuracy = accuracy_score(y_true, y_pred) * 100
        results[dataset] = accuracy
        print(f"   ⭐ ACCURACY for {dataset}: {accuracy:.2f}%")

    print("\n" + "="*60)
    print("🏆 FINAL BENCHMARK SUMMARY")
    print("="*60)
    for ds, acc in results.items():
        print(f" - {ds}: {acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_all_evaluation()
