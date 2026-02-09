from rdkit import Chem
import pandas as pd
import os

def convert_sdf_to_excel(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    # Read the SDF file
    supplier = Chem.SDMolSupplier(filename)
    all_data = []

    for mol in supplier:
        if mol is None:
            continue
        
        # Get all the internal data fields (Activity, CID, Mass, etc.)
        data = mol.GetPropsAsDict()
        
        # Add basic info
        data['Molecule_Name'] = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"
        data['SMILES'] = Chem.MolToSmiles(mol)
        
        all_data.append(data)

    # Convert to Excel
    df = pd.DataFrame(all_data)
    output_name = filename.replace('.sdf', '.xlsx')
    df.to_excel(output_name, index=False)
    print(f"Done! Created: {output_name}")

# Run for your specific files
convert_sdf_to_excel('1843_actives_new.sdf')
convert_sdf_to_excel('1798_actives_new.sdf')