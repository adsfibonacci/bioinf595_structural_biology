import os
import pandas as pd
import pyrosetta

os.chdir("/home/alex/Documents/structural_biology/lab7")
pyrosetta.init("-mute all")

relaxed_pose = pyrosetta.rosetta.core.import_pose.pose_from_file(
    filename="data/relaxed.pdb", # relaxed PDB or the distributed cif file? 
    read_fold_tree=False,
    type=pyrosetta.rosetta.core.import_pose.FileType.PDB_file
)
relaxed_pose_backup = relaxed_pose.clone()

sfxn = pyrosetta.create_score_function(weights_tag="ref2015")
relax = pyrosetta.rosetta.protocols.relax.FastRelax(
    scorefxn_in=sfxn,
    standard_repeats=1
)
print(relax)

def pack(pose, posi, amino, scorefxn):

    # Select Mutate Position
    mut_posi = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    mut_posi.set_index(posi)
    #print(pyrosetta.rosetta.core.select.get_residues_from_subset(mut_posi.apply(pose)))

    # Select Neighbor Position
    nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    nbr_selector.set_focus_selector(mut_posi)
    nbr_selector.set_include_focus_in_subset(True)
    #print(pyrosetta.rosetta.core.select.get_residues_from_subset(nbr_selector.apply(pose)))

    # Select No Design Area
    not_design = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(mut_posi)
    #print(pyrosetta.rosetta.core.select.get_residues_from_subset(not_design.apply(pose)))

    # The task factory accepts all the task operations
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()

    # These are pretty standard
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    # Disable Packing
    prevent_repacking_rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
    prevent_subset_repacking = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repacking_rlt, nbr_selector, True )
    tf.push_back(prevent_subset_repacking)

    # Disable design
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(),not_design))

    # Enable design
    aa_to_design = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    aa_to_design.aas_to_keep(amino)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(aa_to_design, mut_posi))
    
    # Create Packer
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover()
    packer.task_factory(tf)

    #Perform The Move
    if not os.getenv("DEBUG"):
      packer.apply(pose)

print("Starting relaxes")
names = ["8QJ2", "7F58", "7F53"]

#Load the previously-relaxed pose.
relaxPose = pyrosetta.pose_from_pdb("data/relaxed.pdb") 

#Clone it
original = relaxPose.clone()
scorefxn = pyrosetta.get_score_function()
print(f"Relaxed Old Energy:", scorefxn(original))
pack(relaxPose, 130, 'A', scorefxn)
print(f"Relaxed New Energy:", scorefxn(relaxPose))

for name in names:
    #Load the previously-relaxed pose.
    relaxPose = pyrosetta.pose_from_file(
        filename=f"data/{name}.cif",
        read_fold_tree=False,
        type=pyrosetta.rosetta.core.import_pose.FileType.CIF_file
    ) # relaxed pdb from lab 6 or distributed cif? 
    
    #Clone it
    original = relaxPose.clone()
    scorefxn = pyrosetta.get_score_function()
    print(f"{name} Old Energy:", scorefxn(original))
    pack(relaxPose, 130, 'A', scorefxn)
    print(f"{name} New Energy:", scorefxn(relaxPose))

dms = pd.read_csv("data/mc4r_dms.tsv", sep="\t")

# remove nonsense
dms = dms[dms["aa"] != "*"]

# keep only canonical residues
valid_aas = list("ACDEFGHIKLMNPQRSTVWY")
dms = dms[dms["aa"].isin(valid_aas)]

def compute_ddg(pose, pdb_pos, mutant_aa):

  pose_mut = pose.clone()

  pose_index = pose_mut.pdb_info().pdb2pose(chain='A', res=int(pdb_pos))

  if pose_index == 0:
      return None

  wt_score = scorefxn(pose_mut)

  # mutate + repack neighbors
  pack(pose_mut, pose_index, mutant_aa, scorefxn)

  mut_score = scorefxn(pose_mut)

  return mut_score - wt_score  

for pose_file in ["relaxed.pdb", "7F53.cif", "7F58.cif", "8QJ2.cif"]:
  pose = pyrosetta.pose_from_file(f"data/{pose_file}")
  
  scorefxn = pyrosetta.get_score_function()
  
  results = []

  for _, row in dms.iterrows():

    pos = int(row["pos"])
    aa = row["aa"]

    ddg = compute_ddg(pose, pos, aa)

    if ddg is None:
        continue

    results.append({
        "pos": pos,
        "aa": aa,
        "pred_ddg": ddg
    })
    pass
  results_df = pd.DataFrame(results)

  results_df.to_csv(
    f"intermediate/{pose_file}_rosetta_ddg.tsv",
    sep="\t",
    index=False
  )
  
  print(f"Saved predictions for {pose_file}:", len(results_df))
  pass
