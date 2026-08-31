import nbformat
import os

nb = nbformat.v4.new_notebook()

# Cell 1: Imports
cell_1 = nbformat.v4.new_code_cell("""\
import os
import sys
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolAlign
import py3Dmol
from IPython.display import display

# Import our custom kernel
sys.path.append(os.path.abspath("."))
from wcs_gnccp import WCS_GNCCP
""")

# Cell 2: Load Data
cell_2 = nbformat.v4.new_code_cell("""\
# Load two Active molecules
sdf_path = Path("../Data/Aromatase_actives_new.sdf")
supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)

mols = []
for mol in supplier:
    if mol is not None:
        mols.append(mol)
    if len(mols) == 2:
        break

mol1_3d = mols[0]
mol2_3d = mols[1]

# Convert copies to 2D for the topological mapping
mol1_2d = Chem.Mol(mol1_3d)
mol1_2d.RemoveAllConformers()
AllChem.Compute2DCoords(mol1_2d)

mol2_2d = Chem.Mol(mol2_3d)
mol2_2d.RemoveAllConformers()
AllChem.Compute2DCoords(mol2_2d)

print("Molecules loaded successfully!")
""")

# Cell 3: Run WCS
cell_3 = nbformat.v4.new_code_cell("""\
# Run the WCS 2D Graph Kernel
matcher = WCS_GNCCP(alpha=0.7, max_iter=8, fw_max_iter=8)

L = min(mol1_2d.GetNumAtoms(), mol2_2d.GetNumAtoms()) - 2
print(f"Searching for {L} mapping points...")

_, matches, score = matcher.match(mol1_2d, mol2_2d, L=L, verbose=False)
print(f"Mapping Complete! Similarity Score: {score:.3f}")
""")

# Cell 4: 2D Visualization
cell_4 = nbformat.v4.new_code_cell("""\
# Visualize the 2D Topological Mapping
matched_atoms1 = [m[0] for m in matches]
matched_atoms2 = [m[1] for m in matches]

img = Draw.MolsToGridImage(
    [mol1_2d, mol2_2d], 
    molsPerRow=2, 
    subImgSize=(400, 400),
    highlightAtomLists=[matched_atoms1, matched_atoms2],
    legends=["Molecule 1 (Active)", "Molecule 2 (Active)"]
)
display(img)
""")

# Cell 5: 3D Visualization
cell_5 = nbformat.v4.new_code_cell("""\
# 3D Interactive Visualization

# 1. Align 3D coordinates based strictly on the 2D matches
atom_map = [(j, i) for i, j in matches]
rmsd = rdMolAlign.AlignMol(mol2_3d, mol1_3d, atomMap=atom_map)
print(f"3D Alignment RMSD: {rmsd:.3f} Angstroms")

# 2. Extract MolBlocks
mb1 = Chem.MolToMolBlock(mol1_3d)
mb2 = Chem.MolToMolBlock(mol2_3d)

# 3. Render interactively in Jupyter
view = py3Dmol.view(width=800, height=600)
view.addModel(mb1, 'sdf')
view.setStyle({'model': 0}, {'stick': {'color': 'green'}})

view.addModel(mb2, 'sdf')
view.setStyle({'model': 1}, {'stick': {'color': 'cyan'}})

view.zoomTo()
view.show()
""")

nb['cells'] = [cell_1, cell_2, cell_3, cell_4, cell_5]

with open('Interactive_Mapping_Demo.ipynb', 'w') as f:
    nbformat.write(nb, f)
print("Notebook created successfully.")
