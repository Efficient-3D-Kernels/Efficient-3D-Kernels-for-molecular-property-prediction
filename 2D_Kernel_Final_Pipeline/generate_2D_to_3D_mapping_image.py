import os
import sys
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from wcs_gnccp import WCS_GNCCP

def load_mols(sdf_name, max_mols):
    sdf_path = Path("../Data") / sdf_name
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)
    mols = []
    for mol in supplier:
        if mol is not None:
            mols.append(mol)
            if len(mols) == max_mols:
                break
    return mols

def create_3d_image():
    print("Loading molecules...")
    active_mols = load_mols("Aromatase_actives_new.sdf", max_mols=2)
    mol1_3d = active_mols[0]
    mol2_3d = active_mols[1]

    # Convert to 2D for the Kernel Mapping
    mol1_2d = Chem.Mol(mol1_3d)
    mol1_2d.RemoveAllConformers()
    AllChem.Compute2DCoords(mol1_2d)

    mol2_2d = Chem.Mol(mol2_3d)
    mol2_2d.RemoveAllConformers()
    AllChem.Compute2DCoords(mol2_2d)

    print("Running 2D Kernel Mapping...")
    matcher = WCS_GNCCP(alpha=0.7, max_iter=8, fw_max_iter=8)
    L = min(mol1_2d.GetNumAtoms(), mol2_2d.GetNumAtoms()) - 2
    _, matches, score = matcher.match(mol1_2d, mol2_2d, L=L, verbose=False)

    print(f"Aligning 3D conformations based on {len(matches)} matches...")
    atom_map = [(j, i) for i, j in matches]
    rmsd = rdMolAlign.AlignMol(mol2_3d, mol1_3d, atomMap=atom_map)
    print(f"3D Alignment RMSD: {rmsd:.3f}")

    # Plot 3D Overlay using Matplotlib
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get coordinates
    conf1 = mol1_3d.GetConformer()
    conf2 = mol2_3d.GetConformer()

    # Plot Molecule 1 (Green)
    x1, y1, z1 = [], [], []
    for i in range(mol1_3d.GetNumAtoms()):
        pos = conf1.GetAtomPosition(i)
        x1.append(pos.x)
        y1.append(pos.y)
        z1.append(pos.z)
    ax.scatter(x1, y1, z1, c='green', s=100, label='Molecule 1 (Active 0)', alpha=0.6, edgecolors='k')

    # Draw bonds for Mol 1
    for bond in mol1_3d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        pos_i = conf1.GetAtomPosition(i)
        pos_j = conf1.GetAtomPosition(j)
        ax.plot([pos_i.x, pos_j.x], [pos_i.y, pos_j.y], [pos_i.z, pos_j.z], c='green', linewidth=2, alpha=0.5)

    # Plot Molecule 2 (Cyan)
    x2, y2, z2 = [], [], []
    for i in range(mol2_3d.GetNumAtoms()):
        pos = conf2.GetAtomPosition(i)
        x2.append(pos.x)
        y2.append(pos.y)
        z2.append(pos.z)
    ax.scatter(x2, y2, z2, c='cyan', s=100, label='Molecule 2 (Active 1)', alpha=0.6, edgecolors='k')

    # Draw bonds for Mol 2
    for bond in mol2_3d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        pos_i = conf2.GetAtomPosition(i)
        pos_j = conf2.GetAtomPosition(j)
        ax.plot([pos_i.x, pos_j.x], [pos_i.y, pos_j.y], [pos_i.z, pos_j.z], c='cyan', linewidth=2, alpha=0.5)

    ax.set_title(f"3D Overlay of Molecules based on 2D Kernel Mapping (RMSD: {rmsd:.2f}A)", fontsize=14)
    ax.legend()
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    out_path = os.path.abspath("3D_Mapping_Visualization.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved 3D visualization to: {out_path}")

if __name__ == "__main__":
    create_3d_image()