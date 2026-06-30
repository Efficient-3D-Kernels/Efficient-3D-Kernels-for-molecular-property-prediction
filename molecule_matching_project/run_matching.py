# run_matching.py
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
OUT_ROOT = Path("outputs")
ADJ_DIR = OUT_ROOT / "adjacency"
META_DIR = OUT_ROOT / "metadata"
MATCH_OUT_DIR = OUT_ROOT / "matches"
MATCH_OUT_DIR.mkdir(parents=True, exist_ok=True)
# ===================================

def load_molecule_data(stem_name):
    """Load adjacency matrices and metadata for a molecule set"""
    adj_path = ADJ_DIR / f"{stem_name}_adjacency.npy"
    meta_path = META_DIR / f"{stem_name}_metadata.csv"
    
    adjacency_list = np.load(adj_path, allow_pickle=True)
    metadata = pd.read_csv(meta_path)
    
    return adjacency_list, metadata

def load_original_molecules(sdf_name):
    """Load original molecules with 2D coordinates"""
    sdf_path = Path("data") / sdf_name
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    molecules = []
    
    for mol in supplier:
        if mol is not None:
            mol2d = Chem.Mol(mol)
            mol2d.RemoveAllConformers()
            AllChem.Compute2DCoords(mol2d)
            molecules.append(mol2d)
    
    return molecules

def build_atom_features(mol):
    """Extract atom features for similarity comparison"""
    features = []
    for atom in mol.GetAtoms():
        features.append({
            'atomic_num': atom.GetAtomicNum(),
            'degree': atom.GetDegree(),
            'is_in_ring': atom.IsInRing(),
            'hybridization': str(atom.GetHybridization()),
            'mass': atom.GetMass(),
            'charge': atom.GetFormalCharge()
        })
    return features

def build_cost_matrix(mol1, mol2):
    """Build cost matrix for atom matching"""
    M = mol1.GetNumAtoms()
    N = mol2.GetNumAtoms()
    C = np.zeros((M, N))
    
    # Pre-compute atom features
    features1 = build_atom_features(mol1)
    features2 = build_atom_features(mol2)
    
    for i in range(M):
        for j in range(N):
            f1 = features1[i]
            f2 = features2[j]
            
            # Atomic number mismatch (most important)
            if f1['atomic_num'] != f2['atomic_num']:
                C[i, j] += 100
            
            # Degree mismatch
            C[i, j] += abs(f1['degree'] - f2['degree']) * 10
            
            # Ring membership mismatch
            if f1['is_in_ring'] != f2['is_in_ring']:
                C[i, j] += 5
            
            # Charge mismatch
            C[i, j] += abs(f1['charge'] - f2['charge']) * 20
    
    return C

def find_common_substructure(mol1, mol2, L=None):
    """
    Find the most similar substructure between two molecules
    """
    M = mol1.GetNumAtoms()
    N = mol2.GetNumAtoms()
    
    # Set L (number of matches to find)
    if L is None:
        L = min(M, N) - 2  # Leave some outliers
    
    # Build adjacency matrices
    def get_adj(mol):
        n = mol.GetNumAtoms()
        mat = np.zeros((n, n), dtype=np.float32)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            mat[i, j] = 1
            mat[j, i] = 1
        return mat
    
    A1 = get_adj(mol1)
    A2 = get_adj(mol2)
    
    # Build cost matrix
    C = build_cost_matrix(mol1, mol2)
    
    # Normalize cost matrix
    C = (C - C.min()) / (C.max() - C.min() + 1e-8)
    
    # Run greedy matching
    matches = greedy_matching(A1, A2, C, L)
    
    return matches

def greedy_matching(A1, A2, C, L):
    """Greedy algorithm for finding best matches"""
    M, N = A1.shape[0], A2.shape[0]
    
    # Initialize assignment
    row_ind, col_ind = linear_sum_assignment(C)
    
    # Sort by cost
    costs = [C[r, c] for r, c in zip(row_ind, col_ind)]
    sorted_pairs = sorted(zip(row_ind, col_ind, costs), key=lambda x: x[2])
    
    # Keep only L best matches
    selected = sorted_pairs[:L]
    
    return [(r, c) for r, c, _ in selected]

def compare_molecules(mol1, mol2, L=None):
    """Compare two molecules and return similarity score"""
    matches = find_common_substructure(mol1, mol2, L)
    
    if not matches:
        return 0, []
    
    # Calculate similarity score
    score = len(matches) / min(mol1.GetNumAtoms(), mol2.GetNumAtoms())
    
    return score, matches

def visualize_matches(mol1, mol2, matches, save_path=None):
    """Visualize matched atoms between two molecules"""
    if not matches:
        print("No matches to visualize")
        return
    
    matched_atoms1 = [i for i, j in matches]
    matched_atoms2 = [j for i, j in matches]
    
    # Create copies for highlighting
    mol1_copy = Chem.Mol(mol1)
    mol2_copy = Chem.Mol(mol2)
    
    # Draw molecules
    img = Draw.MolsToGridImage(
        [mol1_copy, mol2_copy],
        molsPerRow=2,
        subImgSize=(300, 300),
        highlightAtomLists=[matched_atoms1, matched_atoms2],
        legends=[f"Molecule 1 ({len(matched_atoms1)} matched)", 
                 f"Molecule 2 ({len(matched_atoms2)} matched)"]
    )
    
    if save_path:
        img.save(save_path)
        print(f"✅ Saved visualization to {save_path}")
    else:
        img.show()
    
    return img

def find_similar_molecules(active_mols, inactive_mols, L=10, threshold=0.3):
    """Find pairs of similar molecules between active and inactive sets"""
    similar_pairs = []
    
    total = len(active_mols) * len(inactive_mols)
    count = 0
    
    print(f"🔍 Comparing {len(active_mols)} active vs {len(inactive_mols)} inactive molecules...")
    
    for i, mol1 in enumerate(active_mols):
        for j, mol2 in enumerate(inactive_mols):
            count += 1
            if count % 50 == 0:
                print(f"   Progress: {count}/{total}")
            
            score, matches = compare_molecules(mol1, mol2, L)
            
            if score > threshold and len(matches) >= L * 0.7:
                similar_pairs.append({
                    'active_idx': i,
                    'inactive_idx': j,
                    'score': score,
                    'matches_count': len(matches),
                    'matches': matches
                })
    
    # Sort by score (higher is better)
    similar_pairs.sort(key=lambda x: x['score'], reverse=True)
    
    return similar_pairs

# ========== MAIN EXECUTION ==========
def main():
    print("="*60)
    print("🔬 MOLECULE MATCHING WITH WCS ALGORITHM")
    print("="*60)
    
    # Load data
    print("\n📂 Loading molecule data...")
    active_adj, active_meta = load_molecule_data("Aromatase_actives_new")
    inactive_adj, inactive_meta = load_molecule_data("Aromatase_inactives_new")
    
    print(f"   Active molecules: {len(active_adj)}")
    print(f"   Inactive molecules: {len(inactive_adj)}")
    
    # Load original molecules for matching
    print("\n📂 Loading original molecules with features...")
    active_mols = load_original_molecules("Aromatase_actives_new.sdf")
    inactive_mols = load_original_molecules("Aromatase_inactives_new.sdf")
    
    print(f"   Loaded {len(active_mols)} active molecules")
    print(f"   Loaded {len(inactive_mols)} inactive molecules")
    
    # Find similar molecules
    print("\n🔍 Finding similar molecule pairs...")
    similar_pairs = find_similar_molecules(active_mols, inactive_mols, L=10, threshold=0.3)
    
    print(f"\n✅ Found {len(similar_pairs)} similar molecule pairs!")
    
    # Display top results
    print("\n🏆 TOP 10 MOST SIMILAR PAIRS:")
    print("-"*60)
    for idx, pair in enumerate(similar_pairs[:10]):
        print(f"{idx+1}. Active {pair['active_idx']} ↔ Inactive {pair['inactive_idx']}")
        print(f"   Score: {pair['score']:.3f}")
        print(f"   Matched atoms: {pair['matches_count']}")
        print("-"*40)
    
    # Visualize the best match
    if similar_pairs:
        best_pair = similar_pairs[0]
        print(f"\n🎨 Visualizing best match...")
        mol1 = active_mols[best_pair['active_idx']]
        mol2 = inactive_mols[best_pair['inactive_idx']]
        
        save_path = MATCH_OUT_DIR / "best_match.png"
        visualize_matches(mol1, mol2, best_pair['matches'], save_path)
        
        # Save all results
        results_df = pd.DataFrame([{
            'active_idx': p['active_idx'],
            'inactive_idx': p['inactive_idx'],
            'score': p['score'],
            'matches_count': p['matches_count']
        } for p in similar_pairs])
        
        results_df.to_csv(MATCH_OUT_DIR / "matching_results.csv", index=False)
        print(f"✅ Saved results to {MATCH_OUT_DIR / 'matching_results.csv'}")
    
    print("\n" + "="*60)
    print("✅ MATCHING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()