import os
import random
import csv
import subprocess
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress RDKit C++ warnings
RDLogger.DisableLog('rdApp.*')

# ---------------------------------------------------------
# 1. MUTATION FRAMEWORK
# ---------------------------------------------------------
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
        action = random.choice(["a", "d", "c"])
        action_detail = ""

        try:
            if action == "c":
                target = random.choice(atoms)
                if target.GetIsAromatic(): continue 
                
                new_atomic_num = random.choice([6, 7, 8, 9, 16, 17])
                target.SetAtomicNum(new_atomic_num)
                target.SetFormalCharge(0)
                action_detail = f"Changed atom {target.GetIdx()} to {target.GetSymbol()}"

            elif action == "a":
                valid_targets = [a for a in atoms if a.GetTotalNumHs() > 0]
                if not valid_targets: continue
                target_idx = random.choice(valid_targets).GetIdx()
                
                motif_name = random.choice(list(MOTIF_LIBRARY.keys()))
                motif_mol = Chem.MolFromSmiles(MOTIF_LIBRARY[motif_name])
                
                base_atom_count = rw_mol.GetNumAtoms()
                rw_mol.InsertMol(motif_mol)
                rw_mol.AddBond(target_idx, base_atom_count, Chem.BondType.SINGLE)
                action_detail = f"Added {motif_name} motif to atom {target_idx}"

            elif action == "d":
                terminal_atoms = [a for a in atoms if a.GetDegree() == 1]
                if not terminal_atoms: continue
                
                target = random.choice(terminal_atoms)
                rw_mol.RemoveAtom(target.GetIdx())
                action_detail = f"Deleted terminal atom {target.GetIdx()}"

            Chem.SanitizeMol(rw_mol)
            new_smiles = Chem.MolToSmiles(rw_mol)
            
            if "." not in new_smiles and new_smiles != smiles:
                return new_smiles, f"{action.upper()}: {action_detail}"
                
        except Exception:
            continue

    return smiles, "FAILED"

# ---------------------------------------------------------
# 2. DOCKING FRAMEWORK
# ---------------------------------------------------------
def dock_and_score_smiles(smiles_string, mutant_pdb_path, ligand_name, output_dir):
    """
    Safely handles 3D generation of random SMILES and docks via Smina.
    Incorporates parameters from Loops & Strands tutorial.
    """
    ligand_sdf_path = os.path.join(output_dir, f"{ligand_name}_3D.sdf")
    docked_output_path = os.path.join(output_dir, f"{ligand_name}_docked.sdf")
    
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return None
        
    # Adds hydrogens equivalent to Open Babel's -p flag functionality
    mol = Chem.AddHs(mol)
    
    # Catch 3D embedding failures for physically impossible mutants
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    
    res = AllChem.EmbedMolecule(mol, params)
    if res == -1:
        print("  -> DOCKING SKIPPED: Molecule is too strained to exist in 3D.")
        return None
        
    AllChem.MMFFOptimizeMolecule(mol)
    
    writer = Chem.SDWriter(ligand_sdf_path)
    writer.write(mol)
    writer.close()
    
    smina_path = os.path.join("dependencies", "smina")
    
    # UPDATED: Parameters integrated from the provided tutorial
    smina_cmd = [
        smina_path,
        "-r", mutant_pdb_path,                 
        "-l", ligand_sdf_path,                 
        "--autobox_ligand", mutant_pdb_path,   
        "--autobox_add", "6",                   
        "--exhaustiveness", "16",               
        "-o", docked_output_path               
    ]
    
    try:
        result = subprocess.run(smina_cmd, capture_output=True, text=True, check=True)
        best_score = None
        
        # ROBUST PARSING: Ignore leading spaces and look for token "1"
        for line in result.stdout.split('\n'):
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "1":
                try:
                    best_score = float(parts[1])
                    break
                except ValueError:
                    continue
                    
        # [CRITICAL DEBUGGING BLOCK]
        if best_score is None:
            print("  -> [SMINA SILENT FAILURE]: Smina ran but produced no scores. Here is what it said:")
            print("======== Smina Output ========")
            print(result.stdout.strip())
            if result.stderr.strip():
                print("======== Smina Error ========")
                print(result.stderr.strip())
            print("==============================")
            
        return best_score
            
    except subprocess.CalledProcessError as e:
        print("  -> Smina crashed on this molecule.")
        print(f"  -> [SMINA ERROR LOG]:\n{e.stderr}")        
        return None

# ---------------------------------------------------------
# 3. UNIFIED PIPELINE
# ---------------------------------------------------------
def run_drug_discovery_loop(start_smiles, target_pdb, num_steps=10):
    seed = random.randint(1000, 9999)
    random.seed(seed)
    
    output_dir = os.path.join("testing", f"run_seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "docking_trajectory.csv")
    csv_data = [["Step", "Action", "SMILES", "Smina_Score_kcal_mol", "Status"]]
    
    current_smiles = start_smiles
    target_name = os.path.basename(target_pdb)
    
    print(f"\n[START] Random Drug Walk | Seed: {seed} | Target: {target_name}")
    print("-" * 60)
    
    # Score the Base Molecule
    print("Step 0: Scoring Base Molecule...")
    Draw.MolToFile(Chem.MolFromSmiles(current_smiles), os.path.join(output_dir, "step_0.png"))
    base_score = dock_and_score_smiles(current_smiles, target_pdb, "step_0_base", output_dir)
    print(f"  -> Base Score: {base_score} kcal/mol")
    csv_data.append(["0", "Starting Molecule", current_smiles, base_score, "ACCEPTED"])
    
    for step in range(1, num_steps + 1):
        print(f"\nStep {step}: Mutating...")
        new_smiles, log = mutate_molecule(current_smiles)
        
        if new_smiles == current_smiles: 
            print("  -> Mutation failed to find valid path. Stopping.")
            break
            
        print(f"  -> {log}")
        
        score = dock_and_score_smiles(new_smiles, target_pdb, f"step_{step}_mutant", output_dir)
        
        # [CRITICAL UPDATE] Acceptance Logic
        if score is not None:
            print(f"  -> Binding Affinity: {score} kcal/mol")
            print("  -> [ACCEPTED]: Molecule is physically valid. Moving forward.")
            current_smiles = new_smiles  # Advance the walk ONLY if valid
            Draw.MolToFile(Chem.MolFromSmiles(new_smiles), os.path.join(output_dir, f"step_{step}.png"))
            csv_data.append([str(step), log, new_smiles, str(score), "ACCEPTED"])
        else:
            print("  -> [REJECTED]: Trashing this mutation. Reverting to previous state for next step.")
            csv_data.append([str(step), log, new_smiles, "Failed", "REJECTED"])
        
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
        
    print("-" * 60)
    print(f"Walk complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    atv_smiles = "CC(C)(C)[C@H](NC(=O)OC)C(=O)N[N@](CC1=CC=C(C=C1)C2=CC=CC=N2)C[C@H](O)[C@H](CC3=CC=CC=C3)NC(=O)[C@@H](NC(=O)OC)C(C)(C)C"
    
    target_protein = "data/sanitized_mutated_apo_panels/1fb7/1fb7_WT_relaxed.pdb" 
    
    if os.path.exists(target_protein) and os.path.exists("dependencies/smina"):
        run_drug_discovery_loop(atv_smiles, target_protein, num_steps=10)
    else:
        print(f"[!] Please ensure {target_protein} and dependencies/smina exist before running.")
