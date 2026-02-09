from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import deepchem as dc
import networkx as nx
import json
import os

def ensure_2d_coordinates(mol):
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    return mol

def mol_to_nx(mol):
    mol = ensure_2d_coordinates(mol)
    G = nx.Graph()
    conf = mol.GetConformer()

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        G.add_node(
            idx,
            atomic_num=atom.GetAtomicNum(),
            symbol=atom.GetSymbol(),
            degree=atom.GetDegree(),
            formal_charge=atom.GetFormalCharge(),
            is_aromatic=bool(atom.GetIsAromatic()),
            x=float(pos.x),
            y=float(pos.y),
            z=float(pos.z),
        )

    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=str(bond.GetBondType()),
            is_aromatic=bool(bond.GetIsAromatic()),
        )

    G.graph["n_nodes"] = mol.GetNumAtoms()
    return G

def save_json(G, sim, smiles, path):
    A = nx.to_numpy_array(G, dtype=int)
    data = {
        "smiles": smiles,
        "similarity": float(sim),
        "n_nodes": G.graph["n_nodes"],
        "adjacency": A.tolist(),
        "node_features": [dict(G.nodes[i]) for i in G.nodes()],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

print("Loading Tox21 dataset...")
tasks, datasets, _ = dc.molnet.load_tox21(featurizer="Raw", splitter=None)
smiles_list = datasets[0].ids
print(f"Loaded {len(smiles_list)} molecules")

reference_smiles = "c1ccc(cc1)O" # Phenol
ref_mol = Chem.MolFromSmiles(reference_smiles)
ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, radius=2, nBits=2048)

results = []

print("Computing similarities...")
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    sim = DataStructs.TanimotoSimilarity(ref_fp, fp)
    results.append((smi, sim))

results.sort(key=lambda x: x[1], reverse=True)

SIM_THRESHOLD_ALL = 0.5
SIM_THRESHOLD_HIGH = 0.9

similar_all = [(smi, sim) for smi, sim in results if sim >= SIM_THRESHOLD_ALL]
similar_high = [(smi, sim) for smi, sim in results if sim >= SIM_THRESHOLD_HIGH]

os.makedirs("output_similar_graphs/all_similar", exist_ok=True)
os.makedirs("output_similar_graphs/highly_similar", exist_ok=True)

print("\n================ RESULTS (≥ 0.5) =================")
print(f"Total molecules: {len(similar_all)}")

for i, (smi, sim) in enumerate(similar_all):
    mol = Chem.MolFromSmiles(smi)
    G = mol_to_nx(mol)

    print(f"{i+1}. SMILES={smi}  Similarity={sim:.3f}")

    save_json(
        G,
        sim,
        smi,
        f"output_similar_graphs/all_similar/mol_{i}_sim_{sim:.3f}.json"
    )

print("\n================ RESULTS (≥ 0.9) =================")
print(f"Highly similar molecules: {len(similar_high)}")

for i, (smi, sim) in enumerate(similar_high):
    mol = Chem.MolFromSmiles(smi)
    G = mol_to_nx(mol)

    print(f"{i+1}. SMILES={smi}  Similarity={sim:.3f}")

    save_json(
        G,
        sim,
        smi,
        f"output_similar_graphs/highly_similar/mol_{i}_sim_{sim:.3f}.json"
    )

print("\nAll files saved in: output_similar_graphs/")

