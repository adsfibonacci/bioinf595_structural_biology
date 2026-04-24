import os
import glob
import random
import csv
import multiprocessing
from functools import partial
import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.toolbox import mutate_residue

# Initialize PyRosetta silently - This happens once per process
init("-ex1 -ex2 -relax:default_repeats 2 -ignore_unrecognized_res 1 -mute all")

# ---------------------------------------------------------
# THE 1-SNP MUTATION MATRIX
# ---------------------------------------------------------
SNP_MAP = {
    'A': ['S', 'P', 'T', 'V', 'D', 'E', 'G'],
    'C': ['F', 'S', 'Y', 'R', 'G', 'W'],
    'D': ['Y', 'H', 'N', 'E', 'V', 'A', 'G'],
    'E': ['Q', 'K', 'D', 'V', 'A', 'G'],
    'F': ['L', 'I', 'V', 'S', 'Y', 'C'],
    'G': ['V', 'A', 'D', 'E', 'S', 'R', 'C', 'W'],
    'H': ['Y', 'Q', 'N', 'L', 'P', 'R', 'D'],
    'I': ['F', 'L', 'M', 'V', 'S', 'T', 'N', 'K', 'R'],
    'K': ['Q', 'N', 'E', 'I', 'M', 'T', 'R'],
    'L': ['F', 'I', 'M', 'V', 'S', 'P', 'H', 'Q', 'R', 'W'],
    'M': ['I', 'L', 'V', 'T', 'K', 'R'],
    'N': ['Y', 'H', 'K', 'D', 'I', 'S', 'T'],
    'P': ['S', 'T', 'A', 'L', 'H', 'Q', 'R'],
    'Q': ['H', 'K', 'E', 'L', 'P', 'R'],
    'R': ['L', 'P', 'H', 'Q', 'I', 'M', 'T', 'S', 'K', 'C', 'G', 'W'],
    'S': ['F', 'L', 'I', 'P', 'T', 'A', 'Y', 'N', 'C', 'R', 'G'],
    'T': ['S', 'P', 'A', 'I', 'M', 'N', 'K', 'R'],
    'V': ['F', 'L', 'I', 'M', 'A', 'G', 'D', 'E'],
    'W': ['L', 'R', 'C', 'G'],
    'Y': ['F', 'S', 'C', 'H', 'N', 'D']
}

def generate_biologically_constrained_panel(pdb_file, num_singles=5, num_doubles=5, stability_threshold=7.5, max_attempts=150):
    base_name = os.path.basename(pdb_file).replace(".pdb", "")
    output_dir = os.path.join("data", "mutated_apo_panels", base_name)
    csv_path = os.path.join(output_dir, f"{base_name}_mutations_summary.csv")

    # --- SKIPPING LOGIC ---
    if os.path.exists(csv_path):
        print(f" >>> [SKIP] {base_name} already processed.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[START] Processing: {base_name} | Goal: {num_singles}S, {num_doubles}D")

    csv_data = [["Filename", "Mutation_Type", "Mutation", "ddG_fold_REU"]]

    try:
        pose = pose_from_pdb(pdb_file)
        pyrosetta.rosetta.core.pose.remove_nonprotein_residues(pose)

        scorefxn = get_fa_scorefxn()
        relax = FastRelax()
        relax.set_scorefxn(scorefxn)
        
        relax.apply(pose)
        wt_stability = scorefxn(pose)
        
        wt_filename = f"{base_name}_WT_relaxed.pdb"
        pose.dump_pdb(os.path.join(output_dir, wt_filename))
        csv_data.append([wt_filename, "Wild-Type", "None", "0.00"])
        
        total_residues = pose.total_residue()

        # PHASE 1: SINGLE MUTANTS
        singles_found = 0
        attempts = 0
        while singles_found < num_singles and attempts < max_attempts:
            attempts += 1
            target_pos = random.randint(1, total_residues)
            orig_aa = pose.residue(target_pos).name1()
            if orig_aa not in SNP_MAP: continue
            
            new_aa = random.choice(SNP_MAP[orig_aa])
            mut_name = f"{orig_aa}{target_pos}{new_aa}"
            
            mutant_pose = pose.clone()
            mutate_residue(mutant_pose, target_pos, new_aa, pack_radius=8.0, pack_scorefxn=scorefxn)
            relax.apply(mutant_pose)
            
            penalty = scorefxn(mutant_pose) - wt_stability
            if penalty <= stability_threshold:
                singles_found += 1
                filename = f"{base_name}_single_{singles_found}_{mut_name}.pdb"
                mutant_pose.dump_pdb(os.path.join(output_dir, filename))
                csv_data.append([filename, "Single", mut_name, f"{penalty:.2f}"])

        # PHASE 2: DOUBLE MUTANTS
        doubles_found = 0
        attempts = 0
        while doubles_found < num_doubles and attempts < max_attempts:
            attempts += 1
            pos1, pos2 = random.sample(range(1, total_residues + 1), 2)
            orig1, orig2 = pose.residue(pos1).name1(), pose.residue(pos2).name1()
            if orig1 not in SNP_MAP or orig2 not in SNP_MAP: continue
            
            aa1, aa2 = random.choice(SNP_MAP[orig1]), random.choice(SNP_MAP[orig2])
            mut_name = f"{orig1}{pos1}{aa1}_{orig2}{pos2}{aa2}"
            
            mutant_pose = pose.clone()
            mutate_residue(mutant_pose, pos1, aa1, pack_radius=8.0, pack_scorefxn=scorefxn)
            mutate_residue(mutant_pose, pos2, aa2, pack_radius=8.0, pack_scorefxn=scorefxn)
            relax.apply(mutant_pose)
            
            penalty = scorefxn(mutant_pose) - wt_stability
            if penalty <= stability_threshold:
                doubles_found += 1
                filename = f"{base_name}_double_{doubles_found}_{mut_name}.pdb"
                mutant_pose.dump_pdb(os.path.join(output_dir, filename))
                csv_data.append([filename, "Double", mut_name, f"{penalty:.2f}"])

        # SAVE RESULTS
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)
        print(f"[DONE] Finished {base_name}. Yielded {singles_found}S and {doubles_found}D.")

    except Exception as e:
        print(f"[ERROR] Failed to process {base_name}: {e}")

if __name__ == "__main__":
    pdb_files = glob.glob("data/cleaned_pdb_files/*.pdb", recursive=True)
    
    if not pdb_files:
        print("[!] No PDB files found.")
    else:
        # Pre-filter files to skip so we don't spin up pool workers for nothing
        queue = []
        for f in pdb_files:
            name = os.path.basename(f).replace(".pdb", "")
            if not os.path.exists(os.path.join("data", "mutated_apo_panels", name, f"{name}_mutations_summary.csv")):
                queue.append(f)

        print(f"Total PDBs: {len(pdb_files)} | Completed: {len(pdb_files)-len(queue)} | Remaining: {len(queue)}")

        if queue:
            # Use all but one CPU core to keep the system responsive
            num_cores = max(1, multiprocessing.cpu_count() - 1)
            print(f"Spinning up {num_cores} cores...\n")
            
            # Use partial to set our specific desired arguments
            worker_func = partial(generate_biologically_constrained_panel, 
                                 num_singles=5, 
                                 num_doubles=5, 
                                 stability_threshold=7.5, 
                                 max_attempts=150)
            
            with multiprocessing.Pool(processes=num_cores) as pool:
                pool.map(worker_func, queue)
            
            print("\nAll tasks completed.")
        else:
            print("Everything is already up to date!")
