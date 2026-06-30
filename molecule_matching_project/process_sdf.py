# process_sdf.py
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import os

# ========== CONFIGURATION ==========
DATA_DIR = Path("data")
OUT_ROOT = Path("outputs")

# Create output directories
SDF_OUT_DIR = OUT_ROOT / "sdf_2d"
ADJ_OUT_DIR = OUT_ROOT / "adjacency"
META_OUT_DIR = OUT_ROOT / "metadata"

for directory in (SDF_OUT_DIR, ADJ_OUT_DIR, META_OUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# SDF files to process
INPUT_SDF_FILES = [
    "1843_actives_new.sdf",
    "Aromatase_actives_new.sdf",
    "Aromatase_inactives_new.sdf",
]
# ===================================

def load_valid_molecules(sdf_path: Path):
    """Load molecules from SDF file, skip invalid ones"""
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    return [(i, m) for i, m in enumerate(supplier) if m is not None]

def get_molecule_name(mol: Chem.Mol, fallback_index: int) -> str:
    """Extract molecule name from properties"""
    if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        return mol.GetProp("_Name").strip()
    if mol.HasProp("PUBCHEM_COMPOUND_CID"):
        return f"CID_{mol.GetProp('PUBCHEM_COMPOUND_CID')}"
    return f"mol_{fallback_index:06d}"

def to_2d_mol(mol: Chem.Mol) -> Chem.Mol:
    """Convert molecule to 2D"""
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    AllChem.Compute2DCoords(m)
    return m

def adjacency_binary(mol: Chem.Mol) -> np.ndarray:
    """Create binary adjacency matrix from molecule"""
    n = mol.GetNumAtoms()
    mat = np.zeros((n, n), dtype=np.uint8)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        mat[i, j] = 1
        mat[j, i] = 1
    return mat

def process_all_molecules():
    """Main processing function"""
    summaries = []
    
    for sdf_name in INPUT_SDF_FILES:
        sdf_path = DATA_DIR / sdf_name
        
        if not sdf_path.exists():
            print(f"[SKIP] Missing file: {sdf_path}")
            continue
        
        print(f"\n📂 Processing: {sdf_name}")
        
        # Load molecules
        mols = load_valid_molecules(sdf_path)
        if not mols:
            print(f"[SKIP] No valid molecules in: {sdf_path}")
            continue
        
        stem = sdf_path.stem
        out_sdf = SDF_OUT_DIR / f"{stem}_2d.sdf"
        out_adj = ADJ_OUT_DIR / f"{stem}_adjacency.npy"
        out_meta = META_OUT_DIR / f"{stem}_metadata.csv"
        
        adjacency_mats = []
        records = []
        max_abs_z = 0.0
        
        # Process each molecule
        writer = Chem.SDWriter(str(out_sdf))
        for idx, mol in mols:
            mol2d = to_2d_mol(mol)
            writer.write(mol2d)
            
            adj = adjacency_binary(mol2d)
            adjacency_mats.append(adj)
            
            # Get Z-coordinate info (for 3D to 2D conversion check)
            conf = mol2d.GetConformer()
            z_abs = [abs(conf.GetAtomPosition(a).z) for a in range(mol2d.GetNumAtoms())]
            if z_abs:
                max_abs_z = max(max_abs_z, max(z_abs))
            
            records.append({
                "molecule_index": idx,
                "name": get_molecule_name(mol2d, idx),
                "atoms": mol2d.GetNumAtoms(),
                "bonds": mol2d.GetNumBonds(),
            })
        writer.close()
        
        # Save outputs
        np.save(out_adj, np.array(adjacency_mats, dtype=object), allow_pickle=True)
        pd.DataFrame(records).to_csv(out_meta, index=False)
        
        summaries.append({
            "input": str(sdf_path),
            "molecules": len(mols),
            "output_sdf": str(out_sdf),
            "adjacency_file": str(out_adj),
            "metadata_file": str(out_meta),
            "max_abs_z": max_abs_z,
        })
        
        print(f"✅ Processed {len(mols)} molecules")
        print(f"   SDF: {out_sdf}")
        print(f"   Adjacency: {out_adj}")
        print(f"   Metadata: {out_meta}")
    
    # Save summary
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_ROOT / "processing_summary.csv", index=False)
    print("\n" + "="*50)
    print("✅ ALL DONE!")
    print(summary_df)
    print("="*50)

if __name__ == "__main__":
    process_all_molecules()