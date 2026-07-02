# run_matching.py - Simplified version for testing
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

from wcs_gnccp import WCS_GNCCP

# ========== CONFIGURATION ==========
OUT_ROOT = Path("outputs")
MATCH_OUT_DIR = OUT_ROOT / "matches"
MATCH_OUT_DIR.mkdir(parents=True, exist_ok=True)
# ===================================

def load_original_molecules(sdf_name, max_mols=None):
    """Load molecules from SDF file"""
    sdf_path = Path("data") / sdf_name
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    molecules = []
    
    for mol in supplier:
        if mol is not None and (max_mols is None or len(molecules) < max_mols):
            mol2d = Chem.Mol(mol)
            mol2d.RemoveAllConformers()
            AllChem.Compute2DCoords(mol2d)
            molecules.append(mol2d)
    
    return molecules

def visualize_matches(mol1, mol2, matches, save_path=None):
    """Visualize matched atoms"""
    if not matches:
        print("No matches to visualize")
        return
    
    matched_atoms1 = [int(i) for i, j in matches]
    matched_atoms2 = [int(j) for i, j in matches]
    
    img = Draw.MolsToGridImage(
        [mol1, mol2],
        molsPerRow=2,
        subImgSize=(300, 300),
        highlightAtomLists=[matched_atoms1, matched_atoms2],
        legends=[f"Molecule 1 ({len(matched_atoms1)} matched)", 
                 f"Molecule 2 ({len(matched_atoms2)} matched)"]
    )
    
    if save_path:
        img.save(str(save_path))
        print(f"✅ Saved visualization to {save_path}")
    
    return img

def main():
    print("="*60)
    print("🔬 MOLECULE MATCHING WITH WCS+GNCCP")
    print("="*60)
    
    # Load a small subset for testing
    print("\n📂 Loading molecules...")
    active_mols = load_original_molecules("Aromatase_actives_new.sdf", max_mols=5)
    inactive_mols = load_original_molecules("Aromatase_inactives_new.sdf", max_mols=10)
    
    print(f"   Active molecules: {len(active_mols)}")
    print(f"   Inactive molecules: {len(inactive_mols)}")
    
    # Initialize matcher
    matcher = WCS_GNCCP(alpha=0.7, max_iter=30)
    
    # Find similar pairs
    similar_pairs = []
    total = len(active_mols) * len(inactive_mols)
    count = 0
    
    print("\n🔄 Running WCS+GNCCP matching...")
    
    for i, mol1 in enumerate(active_mols):
        for j, mol2 in enumerate(inactive_mols):
            count += 1
            print(f"   Pair {count}/{total}: Active {i} ↔ Inactive {j}")
            
            try:
                _, matches, score = matcher.match(mol1, mol2, L=6, verbose=False)
                
                if score > 0.2:
                    similar_pairs.append({
                        'active_idx': i,
                        'inactive_idx': j,
                        'score': score,
                        'matches_count': len(matches),
                        'matches': matches
                    })
            except Exception as e:
                print(f"   ⚠️ Error: {str(e)[:50]}")
                continue
    
    # Sort by score
    similar_pairs.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n✅ Found {len(similar_pairs)} similar molecule pairs!")
    
    # Display results
    if similar_pairs:
        print("\n🏆 TOP RESULTS:")
        print("-"*50)
        for idx, pair in enumerate(similar_pairs[:5]):
            print(f"{idx+1}. Active {pair['active_idx']} ↔ Inactive {pair['inactive_idx']}")
            print(f"   Score: {pair['score']:.3f}")
            print(f"   Matched atoms: {pair['matches_count']}")
            print("-"*40)
        
        # Visualize best match
        best = similar_pairs[0]
        print(f"\n🎨 Visualizing best match...")
        mol1 = active_mols[best['active_idx']]
        mol2 = inactive_mols[best['inactive_idx']]
        
        save_path = MATCH_OUT_DIR / "wcs_gnccp_best_match.png"
        visualize_matches(mol1, mol2, best['matches'], save_path)
        
        # Save results
        results_df = pd.DataFrame(similar_pairs)
        results_df.to_csv(MATCH_OUT_DIR / "wcs_gnccp_results.csv", index=False)
        print(f"✅ Saved results to {MATCH_OUT_DIR / 'wcs_gnccp_results.csv'}")
    
    print("\n" + "="*60)
    print("✅ MATCHING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()