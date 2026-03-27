
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

INPUT_SDF_FILES = [
    "1843_actives_new.sdf",
    "Aromatase_actives_new.sdf",
    "Aromatase_inactives_new.sdf",
]

SDF2D_ROOT = Path("outputs_sdf2d")
CIF2D_ROOT = Path("outputs_cif2d")

# Suppress noisy SD field warnings from some source files.
RDLogger.DisableLog("rdApp.warning")


def load_valid_molecules(sdf_path: Path):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    return [m for m in supplier if m is not None]


def canonical_smiles(mol: Chem.Mol) -> str:
    m = Chem.Mol(mol)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


def bond_order_matrix(mol: Chem.Mol) -> np.ndarray:
    n = mol.GetNumAtoms()
    mat = np.zeros((n, n), dtype=float)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        order = float(bond.GetBondTypeAsDouble())
        mat[i, j] = order
        mat[j, i] = order
    return mat


def check_adjacency_matches_mol(mol: Chem.Mol, mat: np.ndarray, tag: str):
    n = mol.GetNumAtoms()
    if mat.shape != (n, n):
        raise AssertionError(f"{tag}: adjacency shape {mat.shape} != ({n}, {n})")
    if not np.array_equal(mat, mat.T):
        raise AssertionError(f"{tag}: adjacency matrix is not symmetric")
    if not np.all(np.diag(mat) == 0):
        raise AssertionError(f"{tag}: adjacency diagonal is not all zeros")
    if not np.isin(mat, [0, 1]).all():
        raise AssertionError(f"{tag}: adjacency matrix has non-binary values")

    expected = np.zeros((n, n), dtype=np.uint8)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        expected[i, j] = 1
        expected[j, i] = 1

    if not np.array_equal(expected, mat.astype(np.uint8)):
        raise AssertionError(f"{tag}: adjacency does not match bond topology")


def parse_cif_blocks(cif_path: Path):
    lines = cif_path.read_text(encoding="utf-8").splitlines()
    blocks = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if not line.startswith("data_"):
            i += 1
            continue

        block_name = line[5:].strip()
        i += 1
        atom_count = 0
        z_values = []

        while i < n and not lines[i].strip().startswith("data_"):
            s = lines[i].strip()
            if s != "loop_":
                i += 1
                continue

            j = i + 1
            headers = []
            while j < n and lines[j].strip().startswith("_"):
                headers.append(lines[j].strip())
                j += 1

            if headers[:5] == [
                "_atom_site_label",
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
            ]:
                i = j
                while i < n:
                    row = lines[i].strip()
                    if (not row) or row.startswith("loop_") or row.startswith("_") or row.startswith("data_"):
                        break
                    parts = row.split()
                    if len(parts) >= 5:
                        atom_count += 1
                        z_values.append(float(parts[4]))
                    i += 1
                continue

            i = j
            while i < n:
                row = lines[i].strip()
                if (not row) or row.startswith("loop_") or row.startswith("_") or row.startswith("data_"):
                    break
                i += 1

        blocks.append({"name": block_name, "atom_count": atom_count, "z_values": z_values})

    return blocks


def check_sdf2d_outputs():
    print("Checking SDF2D outputs...")
    for sdf_name in INPUT_SDF_FILES:
        source = Path(sdf_name)
        if not source.exists():
            print(f"  [SKIP] Missing source: {source}")
            continue

        stem = source.stem
        out_sdf = SDF2D_ROOT / "sdf_2d" / f"{stem}_2d.sdf"
        out_adj = SDF2D_ROOT / "adjacency" / f"{stem}_adjacency.npy"

        if not out_sdf.exists():
            raise FileNotFoundError(f"Missing output SDF: {out_sdf}")
        if not out_adj.exists():
            raise FileNotFoundError(f"Missing output adjacency file: {out_adj}")

        in_mols = load_valid_molecules(source)
        out_mols = load_valid_molecules(out_sdf)
        mats = np.load(out_adj, allow_pickle=True)

        if len(in_mols) != len(out_mols):
            raise AssertionError(f"{stem}: source molecules {len(in_mols)} != output molecules {len(out_mols)}")
        if len(out_mols) != len(mats):
            raise AssertionError(f"{stem}: output molecules {len(out_mols)} != adjacency matrices {len(mats)}")

        for idx, (m_in, m_out, mat) in enumerate(zip(in_mols, out_mols, mats)):
            if m_in.GetNumAtoms() != m_out.GetNumAtoms():
                raise AssertionError(f"{stem}#{idx}: atom count mismatch after 2D conversion")
            if m_in.GetNumBonds() != m_out.GetNumBonds():
                raise AssertionError(f"{stem}#{idx}: bond count mismatch after 2D conversion")

            symbols_in = [a.GetSymbol() for a in m_in.GetAtoms()]
            symbols_out = [a.GetSymbol() for a in m_out.GetAtoms()]
            if symbols_in != symbols_out:
                raise AssertionError(f"{stem}#{idx}: atom symbol ordering changed after 2D conversion")

            if not np.allclose(bond_order_matrix(m_in), bond_order_matrix(m_out)):
                raise AssertionError(f"{stem}#{idx}: bond order topology changed after 2D conversion")

            conf = m_out.GetConformer()
            z_values = [abs(conf.GetAtomPosition(a).z) for a in range(m_out.GetNumAtoms())]
            if z_values and max(z_values) > 1e-4:
                raise AssertionError(f"{stem}#{idx}: non-2D geometry detected (z max {max(z_values):.6f})")

            check_adjacency_matches_mol(m_out, np.array(mat), f"{stem}#{idx}")

        print(f"  [OK] {stem}: {len(out_mols)} molecules")


def check_cif2d_outputs():
    print("Checking CIF2D outputs...")
    for sdf_name in INPUT_SDF_FILES:
        source = Path(sdf_name)
        if not source.exists():
            print(f"  [SKIP] Missing source: {source}")
            continue

        stem = source.stem
        out_cif = CIF2D_ROOT / "cif_2d" / f"{stem}_2d.cif"
        out_adj = CIF2D_ROOT / "adjacency" / f"{stem}_adjacency.npy"

        if not out_cif.exists():
            raise FileNotFoundError(f"Missing output CIF: {out_cif}")
        if not out_adj.exists():
            raise FileNotFoundError(f"Missing output adjacency file: {out_adj}")

        in_mols = load_valid_molecules(source)
        mats = np.load(out_adj, allow_pickle=True)
        blocks = parse_cif_blocks(out_cif)

        if len(in_mols) != len(blocks):
            raise AssertionError(f"{stem}: source molecules {len(in_mols)} != CIF blocks {len(blocks)}")
        if len(in_mols) != len(mats):
            raise AssertionError(f"{stem}: source molecules {len(in_mols)} != adjacency matrices {len(mats)}")

        for idx, (mol, block, mat) in enumerate(zip(in_mols, blocks, mats)):
            n_atoms = mol.GetNumAtoms()
            if block["atom_count"] != n_atoms:
                raise AssertionError(
                    f"{stem}#{idx}: CIF atom count {block['atom_count']} != molecule atom count {n_atoms}"
                )
            if any(abs(z) > 1e-9 for z in block["z_values"]):
                raise AssertionError(f"{stem}#{idx}: CIF has non-zero z fractional coordinates")

            check_adjacency_matches_mol(mol, np.array(mat), f"{stem}#{idx}")

        print(f"  [OK] {stem}: {len(in_mols)} molecules")


if __name__ == "__main__":
    check_sdf2d_outputs()
    check_cif2d_outputs()
    print("All sanity checks passed.")
