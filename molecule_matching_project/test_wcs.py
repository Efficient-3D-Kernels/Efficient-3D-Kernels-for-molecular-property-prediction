# test_wcs.py
"""
Test script to compare simplified vs full WCS+GNCCP
"""
from run_matching import *
from wcs_gnccp import WCS_GNCCP
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import time

def test_single_pair():
    """Test WCS+GNCCP on a single molecule pair"""
    print("="*60)
    print("🧪 TESTING WCS+GNCCP ON A SINGLE PAIR")
    print("="*60)
    
    # Load molecules
    active_mols = load_original_molecules("Aromatase_actives_new.sdf")
    inactive_mols = load_original_molecules("Aromatase_inactives_new.sdf")
    
    mol1 = active_mols[0]
    mol2 = inactive_mols[0]
    
    print(f"Active molecule: {mol1.GetNumAtoms()} atoms")
    print(f"Inactive molecule: {mol2.GetNumAtoms()} atoms")
    
    # Run WCS+GNCCP
    matcher = WCS_GNCCP(alpha=0.7, max_iter=30)
    
    print("\n🔬 Running WCS+GNCCP...")
    start_time = time.time()
    X, matches, score = matcher.match(mol1, mol2, L=8, verbose=True)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️ Time: {elapsed:.2f} seconds")
    print(f"✅ Found {len(matches)} matches (Score: {score:.3f})")
    print(f"   Matched atoms: {matches[:5]}...")
    
    # Visualize
    img = Draw.MolsToGridImage(
        [mol1, mol2],
        molsPerRow=2,
        subImgSize=(300, 300),
        highlightAtomLists=[[i for i, j in matches[:8]], [j for i, j in matches[:8]]],
        legends=["Active (WCS+GNCCP)", "Inactive (WCS+GNCCP)"]
    )
    img.save("wcs_gnccp_test.png")
    print("✅ Saved visualization to wcs_gnccp_test.png")
    
    return matches, score

def test_comparison():
    """Compare simplified vs WCS+GNCCP on multiple pairs"""
    print("="*60)
    print("📊 COMPARING SIMPLIFIED vs WCS+GNCCP")
    print("="*60)
    
    # Load molecules
    active_mols = load_original_molecules("Aromatase_actives_new.sdf")[:10]
    inactive_mols = load_original_molecules("Aromatase_inactives_new.sdf")[:10]
    
    results = []
    
    for i, mol1 in enumerate(active_mols):
        for j, mol2 in enumerate(inactive_mols):
            # Simplified
            score_simple, matches_simple = compare_molecules_simplified(mol1, mol2, L=8)
            
            # WCS+GNCCP
            matcher = WCS_GNCCP(alpha=0.7, max_iter=20)
            _, matches_wcs, score_wcs = matcher.match(mol1, mol2, L=8, verbose=False)
            
            results.append({
                'pair': f"{i}-{j}",
                'simplified_score': score_simple,
                'simplified_matches': len(matches_simple),
                'wcs_score': score_wcs,
                'wcs_matches': len(matches_wcs),
                'improvement': score_wcs - score_simple
            })
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(results)
    
    print("\n📊 Results:")
    print(df[['pair', 'simplified_score', 'wcs_score', 'improvement']].to_string())
    
    print("\n📈 Summary:")
    print(f"Average simplified score: {df['simplified_score'].mean():.3f}")
    print(f"Average WCS score: {df['wcs_score'].mean():.3f}")
    print(f"Average improvement: {df['improvement'].mean():.3f}")
    
    return df

if __name__ == "__main__":
    # Run tests
    test_single_pair()
    print("\n" + "="*60)
    test_comparison()