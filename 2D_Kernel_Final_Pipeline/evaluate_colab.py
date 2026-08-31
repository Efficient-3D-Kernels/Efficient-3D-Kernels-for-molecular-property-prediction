import os
import sys
import time
import glob
import argparse
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

from wcs_gnccp import WCS_GNCCP

def load_mols(sdf_path, max_mols):
    if not os.path.exists(sdf_path):
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

def compute_kernel_parallel(mol_data_list, matcher, n_jobs=-1):
    n = len(mol_data_list)
    K = np.zeros((n, n))
    np.fill_diagonal(K, 1.0)
    
    def match_pair(i, j):
        score = matcher.match_from_precomputed(mol_data_list[i], mol_data_list[j])
        return i, j, score

    # Generate all pairs
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Run in parallel
    # NOTE: If GPU is used, n_jobs will be forced to 1 to prevent CUDA memory/context crashes
    results = Parallel(n_jobs=n_jobs)(delayed(match_pair)(i, j) for i, j in pairs)
    
    for i, j, score in results:
        K[i, j] = score
        K[j, i] = score
        
    return K

def evaluate_1nn(K, labels):
    D = 1.0 - K
    np.fill_diagonal(D, 0.0)
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(D):
        D_train = D[np.ix_(train_idx, train_idx)]
        D_test = D[np.ix_(test_idx, train_idx)]
        clf = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
        clf.fit(D_train, labels[train_idx])
        pred = clf.predict(D_test)
        y_true.append(labels[test_idx][0])
        y_pred.append(pred[0])
    return accuracy_score(y_true, y_pred) * 100

def evaluate_svm(K, labels):
    K_sym = (K + K.T) / 2
    K_psd = K_sym + np.eye(len(K)) * 1e-6
    clf = SVC(kernel='precomputed', C=1.0)
    
    n_splits = min(5, sum(labels==0), sum(labels==1))
    if n_splits < 2:
        return 0.0
        
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, K_psd, labels, cv=cv, scoring='accuracy')
    return np.mean(scores) * 100

def run_evaluation(args):
    data_dir = args.data_dir
    n_per_class = args.max_mols
    use_gpu = args.use_gpu
    n_jobs = args.n_jobs
    
    # Find all active datasets
    active_files = glob.glob(os.path.join(data_dir, "*_actives_new.sdf"))
    datasets = []
    for f in active_files:
        basename = os.path.basename(f)
        dataset_name = basename.replace("_actives_new.sdf", "")
        inact_file = os.path.join(data_dir, f"{dataset_name}_inactives_new.sdf")
        if os.path.exists(inact_file):
            datasets.append(dataset_name)
    
    datasets = sorted(datasets)
    
    alpha_values = [0.5, 0.7]
    gamma_values = [0.5, 1.0]
    max_iter = 25
    fw_max_iter = 12
    
    print("=" * 70)
    print("🚀 COLAB MULTI-CPU & GPU BENCHMARK")
    print(f"   Data Directory: {data_dir}")
    print(f"   Datasets Found: {len(datasets)} total")
    print(f"   Hardware:       {'GPU (CuPy)' if use_gpu else f'CPU (Joblib with {n_jobs} cores)'}")
    print("=" * 70)
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"📂 Dataset: {dataset}")
        
        act_name = os.path.join(data_dir, f"{dataset}_actives_new.sdf")
        inact_name = os.path.join(data_dir, f"{dataset}_inactives_new.sdf")
        
        active_mols = load_mols(act_name, max_mols=n_per_class)
        inactive_mols = load_mols(inact_name, max_mols=n_per_class)
        
        mols_2d = [convert_to_2d(m) for m in active_mols] + [convert_to_2d(m) for m in inactive_mols]
        labels = np.array([1] * len(active_mols) + [0] * len(inactive_mols))
        n_total = len(mols_2d)
        
        print(f"   Loaded {n_total} molecules ({len(active_mols)} active, {len(inactive_mols)} inactive)")
        if n_total < 4:
            continue
            
        print(f"   📦 Precomputing features...")
        mol_data_list = [WCS_GNCCP.precompute_molecular_data(m) for m in mols_2d]
        
        best_1nn = 0.0
        best_svm = 0.0
        best_1nn_config = ""
        best_svm_config = ""
        
        for alpha in alpha_values:
            for gamma in gamma_values:
                config_str = f"α={alpha}, γ={gamma}"
                print(f"   ⚙️  Testing {config_str}...")
                
                matcher = WCS_GNCCP(alpha=alpha, gamma=gamma, max_iter=max_iter, fw_max_iter=fw_max_iter, use_gpu=use_gpu)
                
                start_time = time.time()
                # If GPU is enabled, fallback to single process to prevent CUDA context crashes
                safe_n_jobs = 1 if use_gpu else n_jobs
                K = compute_kernel_parallel(mol_data_list, matcher, n_jobs=safe_n_jobs)
                elapsed = time.time() - start_time
                
                acc_1nn = evaluate_1nn(K, labels)
                acc_svm = evaluate_svm(K, labels)
                
                print(f"      1-NN: {acc_1nn:>5.1f}% | SVM: {acc_svm:>5.1f}% | Time: {elapsed:.1f}s")
                
                if acc_1nn > best_1nn:
                    best_1nn = acc_1nn
                    best_1nn_config = config_str
                if acc_svm > best_svm:
                    best_svm = acc_svm
                    best_svm_config = config_str
                    
        all_results[dataset] = {
            'best_1nn': best_1nn, 'best_1nn_config': best_1nn_config,
            'best_svm': best_svm, 'best_svm_config': best_svm_config,
        }
        
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY")
    for ds, res in all_results.items():
        print(f"{ds:<15} 1-NN: {res['best_1nn']:>5.1f}% ({res['best_1nn_config']:>11}) | SVM: {res['best_svm']:>5.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/datasets')
    parser.add_argument('--max_mols', type=int, default=10000)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--n_jobs', type=int, default=-1)
    args = parser.parse_args()
    
    run_evaluation(args)
