import os
import random
import subprocess
import pandas as pd
import concurrent.futures
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# ---------------------------------------------------------
# 1. RDKIT MUTATION ENGINE (Single Kick)
# ---------------------------------------------------------
def mutate_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return smiles, "Invalid input SMILES"
    
    rw_mol = Chem.RWMol(mol)
    num_atoms = rw_mol.GetNumAtoms()
    if num_atoms == 0:
        return smiles, "Empty molecule"

    action = random.choice(["DELETE", "CHANGE", "ADD"])
    target_idx = random.randint(0, num_atoms - 1)
    
    if action == "DELETE" and num_atoms > 5:
        atom = rw_mol.GetAtomWithIdx(target_idx)
        if atom.GetDegree() == 1:
            rw_mol.RemoveAtom(target_idx)
            return Chem.MolToSmiles(rw_mol), f"DELETE: Deleted terminal atom {target_idx}"
            
    elif action == "CHANGE":
        atom = rw_mol.GetAtomWithIdx(target_idx)
        if atom.GetSymbol() == 'C' and not atom.IsInRing():
            new_element = random.choice(['O', 'N', 'S'])
            atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_element))
            return Chem.MolToSmiles(rw_mol), f"CHANGE: Changed atom {target_idx} to {new_element}"
            
    elif action == "ADD":
        atom = rw_mol.GetAtomWithIdx(target_idx)
        if atom.GetSymbol() in ['C', 'N'] and atom.GetValence(Chem.ValenceType.IMPLICIT) > 0:
            new_idx = rw_mol.AddAtom(Chem.Atom(6)) 
            rw_mol.AddBond(target_idx, new_idx, Chem.BondType.SINGLE)
            return Chem.MolToSmiles(rw_mol), f"ADD: Added methyl motif to atom {target_idx}"

    return smiles, "None"

# ---------------------------------------------------------
# 2. SYNFORMER PROJECTION PIPELINE (Expansion)
# ---------------------------------------------------------
def project_synformer_batch(smiles_list, output_dir):
    input_csv = os.path.abspath(os.path.join(output_dir, "synformer_input.csv"))
    output_csv = os.path.abspath(os.path.join(output_dir, "synformer_output.csv"))
    
    pd.DataFrame({"smiles": smiles_list}).to_csv(input_csv, index=False)
    
    synformer_dir = os.path.abspath("dependencies/synformer")
    sample_script = "sample.py" if os.path.exists(os.path.join(synformer_dir, "sample.py")) else "scripts/sample.py"

    synformer_cmd = [
        "python", sample_script,
        "--model-path", "data/trained_weights/sf_ed_default.ckpt", 
        "--input", input_csv,
        "--output", output_csv
    ]
    
    print(f"  -> Running SynFormer to expand {len(smiles_list)} base molecule(s)...")
    try:
        subprocess.run(synformer_cmd, check=True, capture_output=True, text=True, cwd=synformer_dir)
        
        if os.path.exists(output_csv):
            df_out = pd.read_csv(output_csv)
            projected_col = df_out.columns[1] if df_out.shape[1] > 1 else df_out.columns[0]
            valid_projections = df_out[projected_col].dropna().tolist()
            return valid_projections
    except subprocess.CalledProcessError as e:
        print(f"  -> [SYNFORMER ERROR]: {e.stderr}")
        
    return []

# ---------------------------------------------------------
# 3. 3D FOLDING (Runs ONCE per ligand)
# ---------------------------------------------------------
def fold_ligand_to_3d(smiles, idx, output_dir):
    """
    Folds the 1D SMILES into a 3D SDF file for docking.
    Returns (smiles, sdf_path) if successful, or (smiles, None) if it fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return smiles, None
    
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.maxIterations = 1000  
    
    res = AllChem.EmbedMolecule(mol, params)
    if res == -1:
        params.useRandomCoords = True
        params.boxSizeMult = 2.0
        res = AllChem.EmbedMolecule(mol, params)
        
    if res == -1:
        return smiles, None 
        
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass 

    sdf_path = os.path.join(output_dir, f"ligand_{idx}_3d.sdf")
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()

    return smiles, sdf_path

# ---------------------------------------------------------
# 4. SMINA DOCKING (Runs per Receptor-Ligand pair)
# ---------------------------------------------------------
def run_smina_docking(sdf_path, target_pdb, target_name, idx, output_dir):
    """
    Takes a pre-folded SDF and docks it into a specific receptor.
    """
    out_sdf = os.path.join(output_dir, f"docked_{target_name}_ligand_{idx}.sdf")
    smina_cmd = [
        "dependencies/smina", 
        "--receptor", target_pdb,
        "--ligand", sdf_path,
        "--autobox_ligand", sdf_path,
        "--exhaustiveness", "4",
        "--out", out_sdf,
        "--quiet"
    ]
    
    try:
        result = subprocess.run(smina_cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if line.strip().startswith('1'):
                parts = line.split()
                if len(parts) >= 2:
                    return target_name, sdf_path, float(parts[1])
    except Exception:
        pass
        
    return target_name, sdf_path, None

# ---------------------------------------------------------
# 5. MULTI-RECEPTOR SWARM PIPELINE
# ---------------------------------------------------------
def cross_docking_pipeline(start_smiles, target_pdbs, max_workers=6):
    seed = random.randint(1000, 9999)
    random.seed(seed)
    
    output_dir = os.path.join("results", f"cross_dock_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[START] Cross-Docking Swarm Pipeline | Seed: {seed}")
    print(f"Targeting {len(target_pdbs)} distinct receptors.")
    print("-" * 60)
    
    # 1. Single RDKit Mutation
    print("Step 1: Generating a single RDKit mutation...")
    mutated_smi = start_smiles
    action_taken = "None"
    
    attempts = 0
    while mutated_smi == start_smiles and attempts < 10:
        mutated_smi, action_taken = mutate_molecule(start_smiles)
        attempts += 1
        
    if mutated_smi == start_smiles:
        print("  -> Failed to generate a valid mutation after 10 attempts. Exiting.")
        return
        
    print(f"  -> Action: {action_taken}")
    print(f"  -> Base Mutation: {mutated_smi}")
    
    # 2. SynFormer Expansion
    print("\nStep 2: Passing mutation to SynFormer for expansion...")
    projected_batch = project_synformer_batch([mutated_smi], output_dir)
    projected_batch = list(set(projected_batch))
    print(f"  -> SynFormer generated {len(projected_batch)} unique synthesizable molecules.")
    
    if not projected_batch:
        print("  -> No valid projections. Exiting.")
        return
        
    # 2.5 Save 2D Structures
    print("\nStep 2.5: Saving 2D structures as PNGs...")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    for i, smi in enumerate(projected_batch):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            Draw.MolToFile(mol, os.path.join(images_dir, f"ligand_{i}.png"), size=(400, 400))
            
    # 3. Parallel 3D Folding
    print(f"\nStep 3: Folding ligands into 3D (Workers: {max_workers})...")
    valid_3d_ligands = [] # Will hold tuples of (smiles, sdf_path)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fold_ligand_to_3d, smi, i, output_dir) for i, smi in enumerate(projected_batch)]
        for future in concurrent.futures.as_completed(futures):
            smi, sdf_path = future.result()
            if sdf_path is not None:
                valid_3d_ligands.append((smi, sdf_path))
                
    print(f"  -> Successfully folded {len(valid_3d_ligands)} out of {len(projected_batch)} ligands.")

    # 4. Pairwise Parallel Docking Matrix
    print(f"\nStep 4: Cross-Docking Matrix (Receptors: {len(target_pdbs)} x Ligands: {len(valid_3d_ligands)})...")
    
    # Create a dictionary to organize results by target PDB
    results_by_target = {os.path.basename(pdb).replace('.pdb', ''): [] for pdb in target_pdbs}
    total_tasks = len(target_pdbs) * len(valid_3d_ligands)
    completed_tasks = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_ligand = {}
        
        # Matrix Generation: Queue up every Receptor + Ligand combination
        for pdb_path in target_pdbs:
            target_name = os.path.basename(pdb_path).replace('.pdb', '')
            for i, (smi, sdf_path) in enumerate(valid_3d_ligands):
                future = executor.submit(run_smina_docking, sdf_path, pdb_path, target_name, i, output_dir)
                futures_to_ligand[future] = smi # Map future back to the SMILES string
                
        # Process as they finish
        for future in concurrent.futures.as_completed(futures_to_ligand):
            smi = futures_to_ligand[future]
            target_name, sdf_path, score = future.result()
            completed_tasks += 1
            
            if score is not None:
                results_by_target[target_name].append((smi, score))
                
            if completed_tasks % 20 == 0 or completed_tasks == total_tasks:
                print(f"  -> Docked {completed_tasks}/{total_tasks} interactions...")

    # Summary Display
    print("\n" + "=" * 60)
    print("CROSS-DOCKING RESULTS (Top 5 per Receptor)")
    print("=" * 60)
    
    for target_name, scores_list in results_by_target.items():
        print(f"\nRECEPTOR: {target_name}")
        print("-" * 40)
        if not scores_list:
            print("  No successful docking events.")
            continue
            
        sorted_scores = sorted(scores_list, key=lambda x: x[1])
        for smi, score in sorted_scores[:5]:
            print(f"  Score: {score:7.2f} kcal/mol | {smi}")
            
    print("\n" + "-" * 60)
    print(f"Pipeline complete. All files saved to {output_dir}/")

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # You can now add as many PDBs to this list as you want!
    TARGET_RECEPTORS = [
        "data/sanitized_mutated_apo_panels/1fb7/1fb7_WT_relaxed.pdb",
        # "data/sanitized_mutated_apo_panels/1fb7/1fb7_MUTANT_A.pdb",  <-- Add your second PDB here
        # "data/sanitized_mutated_apo_panels/1fb7/1fb7_MUTANT_B.pdb"   <-- Add your third PDB here
    ]
    
    STARTING_LIGAND = "COC(=O)N[C@H](C(=O)NN(Cc1ccc(-c2ccccn2)cc1)C[C@H](O)[C@H](Cc1ccccc1)NC(=O)[C@@H](NC(=O)OC)C(C)(C)C)C(C)(C)C"
    
    # Filter out missing PDBs to prevent pipeline crashes
    valid_receptors = [pdb for pdb in TARGET_RECEPTORS if os.path.exists(pdb)]
    
    if not valid_receptors:
        print("ERROR: Cannot find any valid receptor files. Please check your paths.")
    else:
        cross_docking_pipeline(
            start_smiles=STARTING_LIGAND,
            target_pdbs=valid_receptors,
            max_workers=6  # Adjust based on your CPU cores
        )
