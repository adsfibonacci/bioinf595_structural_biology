import os
import time
import csv
import shutil
import pyrosetta

pyrosetta.init("-mute all")

# -----------------------------
# Directory setup
# -----------------------------
SRC = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.abspath(os.path.join(SRC, ".."))
DATA = os.path.join(ROOT, "data")
INTER = os.path.join(ROOT, "intermediates")
RELAX_DIR = os.path.join(INTER, "relaxes")

# Ensure base directories exist
os.makedirs(RELAX_DIR, exist_ok=True)

# -----------------------------
# CLEANUP SAFETY FIX
# Remove any accidental directories
# that were created with .pdb names
# -----------------------------
for item in os.listdir(RELAX_DIR):
    full_path = os.path.join(RELAX_DIR, item)
    if item.endswith(".pdb") and os.path.isdir(full_path):
        print(f"Removing broken directory: {full_path}")
        shutil.rmtree(full_path)

# -----------------------------
# Load structure
# -----------------------------
structure_fname = os.path.join(DATA, "MC4R.cif")
print("Loading:", structure_fname)

pose_original = pyrosetta.rosetta.core.import_pose.pose_from_file(
    filename=structure_fname,
    read_fold_tree=False,
    type=pyrosetta.rosetta.core.import_pose.FileType.CIF_file
)

pose_native = pose_original.clone()

# -----------------------------
# Score function
# -----------------------------
sfxn = pyrosetta.create_score_function(weights_tag="ref2015")
original_score = sfxn(pose_original)
print("Native score:", original_score)

# -----------------------------
# FastRelax setup
# -----------------------------
fast_relax = pyrosetta.rosetta.protocols.relax.FastRelax(
    scorefxn_in=sfxn,
    standard_repeats=1
)

fast_relax.constrain_relax_to_start_coords()
fast_relax.ramp_down_constraints(False)

# -----------------------------
# Sampling loop
# -----------------------------
nsamples = 5
metadata = []

for i in range(nsamples):
    print(f"\nRunning sample {i+1}/{nsamples}")

    pose_iter = pose_native.clone()

    start = time.time()
    fast_relax.apply(pose_iter)
    relax_time = time.time() - start

    relaxed_score = sfxn(pose_iter)
    rmsd = pyrosetta.rosetta.core.scoring.all_atom_rmsd(
        pose_native, pose_iter
    )
    delta_score = relaxed_score - original_score

    print(f"  Relaxed score: {relaxed_score:.2f}")
    print(f"  ΔScore: {delta_score:.2f}")
    print(f"  Heavy-atom RMSD: {rmsd:.2f} Å")
    print(f"  Relax time: {relax_time:.2f} s")

    # Correct file writing
    fname_out = os.path.join(RELAX_DIR, f"MC4R_relaxed_{i+1}.pdb")
    pose_iter.dump_pdb(fname_out)

    metadata.append([
        i + 1,
        original_score,
        relaxed_score,
        delta_score,
        rmsd,
        relax_time
    ])

# -----------------------------
# Save metadata
# -----------------------------
fname_metadata = os.path.join(RELAX_DIR, "relax_metadata.tsv")

with open(fname_metadata, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        "sample",
        "native_score",
        "relaxed_score",
        "delta_score",
        "RMSD",
        "relax_time"
    ])
    writer.writerows(metadata)

print("\nCompleted all relaxations.")
print("Metadata saved to:", fname_metadata)
