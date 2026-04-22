import os
import random
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import RDLogger

# Suppress RDKit C++ warnings and errors from cluttering the terminal
RDLogger.DisableLog('rdApp.*')

# --- Define the Multi-Atom Motif Library ---
MOTIF_LIBRARY = {
    "methyl": "C",
    "amino": "N",
    "hydroxyl": "O",
    "fluoro": "F",
    "trifluoromethyl": "C(F)(F)F",
    "phenyl": "c1ccccc1",
    "amide": "C(=O)N",
    "cyano": "C#N",
    "methoxy": "OC",
    "pyridine": "c1ccncc1"
}

def mutate_molecule(smiles, max_retries=20):
    for _ in range(max_retries):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: continue
        
        rw_mol = Chem.RWMol(mol)
        atoms = list(rw_mol.GetAtoms())
        action = random.choice(["add", "delete", "change"])
        action_detail = ""

        try:
            if action == "change":
                target = random.choice(atoms)
                if target.GetIsAromatic(): continue 
                
                new_atomic_num = random.choice([6, 7, 8, 9, 16, 17])
                target.SetAtomicNum(new_atomic_num)
                target.SetFormalCharge(0)
                action_detail = f"Changed atom {target.GetIdx()} to {target.GetSymbol()}"

            elif action == "add":
                # UPDATED: Use GetTotalNumHs() to check for open valence without triggering deprecation warnings
                valid_targets = [a for a in atoms if a.GetTotalNumHs() > 0]
                if not valid_targets: continue
                target_idx = random.choice(valid_targets).GetIdx()
                
                motif_name = random.choice(list(MOTIF_LIBRARY.keys()))
                motif_mol = Chem.MolFromSmiles(MOTIF_LIBRARY[motif_name])
                
                base_atom_count = rw_mol.GetNumAtoms()
                rw_mol.InsertMol(motif_mol)
                
                rw_mol.AddBond(target_idx, base_atom_count, Chem.BondType.SINGLE)
                action_detail = f"Added {motif_name} motif to atom {target_idx}"

            elif action == "delete":
                terminal_atoms = [a for a in atoms if a.GetDegree() == 1]
                if not terminal_atoms: continue
                
                target = random.choice(terminal_atoms)
                rw_mol.RemoveAtom(target.GetIdx())
                action_detail = f"Deleted terminal atom {target.GetIdx()}"

                pass
            Chem.SanitizeMol(rw_mol)
            new_smiles = Chem.MolToSmiles(rw_mol)
            
            if "." not in new_smiles and new_smiles != smiles:
                return new_smiles, f"{action.upper()}: {action_detail}"
                
        except Exception:
            continue

        pass
    return smiles, "FAILED"

def run_mutation_framework(start_smiles, num_steps=10):
    seed = random.randint(1000, 9999)
    random.seed(seed)
    output_dir = os.path.join("testing", f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    
    current_smiles = start_smiles
    Draw.MolToFile(Chem.MolFromSmiles(current_smiles), os.path.join(output_dir, "step_0.png"))
    
    print(f"Starting Walk from ATV (Seed: {seed})")
    print("-" * 40)
    
    for step in range(1, num_steps + 1):
        new_smiles, log = mutate_molecule(current_smiles)
        if new_smiles == current_smiles: break
        
        print(f"Step {step}: {log}")
        Draw.MolToFile(Chem.MolFromSmiles(new_smiles), os.path.join(output_dir, f"step_{step}.png"))
        current_smiles = new_smiles
        
    print("-" * 40)
    print(f"Walk complete. Check {output_dir}/ for images.")

if __name__ == "__main__":
    atv_smiles = "CC(C)(C)[C@H](NC(=O)OC)C(=O)N[N@](CC1=CC=C(C=C1)C2=CC=CC=N2)C[C@H](O)[C@H](CC3=CC=CC=C3)NC(=O)[C@@H](NC(=O)OC)C(C)(C)C"
    run_mutation_framework(atv_smiles, num_steps=30)
