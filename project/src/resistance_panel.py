import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.toolbox import mutate_residue

# 1. Initialize PyRosetta (include flags for accurate sidechain packing)
init("-ex1 -ex2 -relax:default_repeats 2 -ignore_unrecognized_res 1")

def evaluate_complex(pose, interface_chains="A_X"):
    """
    Analyzes the interface between the protein (e.g., Chain A) 
    and the ligand (e.g., Chain X).
    """
    iam = InterfaceAnalyzerMover(interface_chains)
    iam.apply(pose)
    
    # Extract binding energy (dG_separated)
    ddg = iam.get_interface_dG()
    return ddg

def run_phase2_pipeline(pdb_file, mutate_pos, new_amino_acid, ligand_chain="X"):
    # 2. Load the wild-type complex
    pose = pose_from_pdb(pdb_file)
    print(f"Loaded {pdb_file} with {pose.total_residue()} residues.")

    # 3. FastRelax the Wild-Type
    # This relieves steric clashes from crystal structures before scoring
    scorefxn = get_fa_scorefxn()
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    
    print("Relaxing Wild-Type complex... (This may take a few minutes)")
    relax.apply(pose)
    
    wt_ddg = evaluate_complex(pose, f"A_{ligand_chain}")
    print(f"Wild-Type Binding Energy (ΔΔG): {wt_ddg:.2f} REU")

    # 4. Introduce Point Mutation 
    # mutate_residue uses PackRotamersMover under the hood to repack the surrounding area
    print(f"Mutating position {mutate_pos} to {new_amino_acid}...")
    mutant_pose = pose.clone()
    mutate_residue(mutant_pose, mutate_pos, new_amino_acid, pack_radius=8.0, pack_scorefxn=scorefxn)
    
    # 5. FastRelax the Mutant to allow the backbone to adjust to the new sidechain
    print("Relaxing Mutant complex...")
    relax.apply(mutant_pose)

    mut_ddg = evaluate_complex(mutant_pose, f"A_{ligand_chain}")
    print(f"Mutant Binding Energy (ΔΔG): {mut_ddg:.2f} REU")
    
    # 6. Output structures for your static dataset
    pose.dump_pdb("relaxed_WT_complex.pdb")
    mutant_pose.dump_pdb(f"relaxed_mutant_{mutate_pos}{new_amino_acid}_complex.pdb")
    
    return wt_ddg, mut_ddg

if __name__ == "__main__":
    # Example usage: Mutate residue 82 to Alanine, assuming ligand is chain X
    # Note: Rosetta residue indices might differ from PDB numbering!
    run_phase2_pipeline("hiv_complex.pdb", mutate_pos=82, new_amino_acid="A", ligand_chain="X")
