import argparse
import csv
import os
import time

# ── CLI args must be parsed BEFORE pyrosetta.init() ──────────────────────────
parser = argparse.ArgumentParser(description="BackRub ensemble sampler")
parser.add_argument("--native_pdb",  required=True,
                    help="Native / reference PDB for RMSD calculation")
parser.add_argument("--input_pdb",   required=True,
                    help="Starting conformation (relaxed PDB)")
parser.add_argument("--output_dir",  required=True,
                    help="Directory to write sampled PDBs and metadata")
parser.add_argument("--ntrials",     type=int,   default=5000,
                    help="BackRub MCMC steps per sample")
parser.add_argument("--mc_kt",       type=float, default=0.7,
                    help="MCMC temperature (kT units)")
parser.add_argument("--nsamples",    type=int,   default=10,
                    help="Number of independent samples to generate")
args = parser.parse_args()

# ── Init PyRosetta with BackRub options ──────────────────────────────────────
import pyrosetta 

pyrosetta.init(
    extra_options=(
        f"-mute all "
        f"-backrub:ntrials={args.ntrials} "
        f"-backrub:mc_kt={args.mc_kt}"
    )
)

# ── Paths ────────────────────────────────────────────────────────────────────
os.makedirs(args.output_dir, exist_ok=True)

tag = f"kt{args.mc_kt}_n{args.ntrials}"

# ── Load structures ──────────────────────────────────────────────────────────
print(f"Loading native : {args.native_pdb}")
pose_native = pyrosetta.pose_from_pdb(args.native_pdb)

print(f"Loading input  : {args.input_pdb}")
pose_input = pyrosetta.pose_from_pdb(args.input_pdb)

# ── Score function ───────────────────────────────────────────────────────────
sfxn = pyrosetta.create_score_function("ref2015")

input_score  = sfxn(pose_input)
native_score = sfxn(pose_native)
print(f"Input score  : {input_score:.2f}")
print(f"Native score : {native_score:.2f}")

# ── BackRub protocol ─────────────────────────────────────────────────────────
backrub_protocol = pyrosetta.rosetta.protocols.backrub.BackrubProtocol()

# ── Sampling loop ────────────────────────────────────────────────────────────
metadata = []

for i in range(args.nsamples):
    print(f"\n[{i+1}/{args.nsamples}]  kt={args.mc_kt}  ntrials={args.ntrials}")

    pose_iter = pose_input.clone() 

    t0 = time.time()
    backrub_protocol.apply(pose_iter)
    elapsed = time.time() - t0

    score = sfxn(pose_iter)

    # RMSD to native (reference experimental conformation)
    rmsd_to_native = pyrosetta.rosetta.core.scoring.all_atom_rmsd(
        pose_native, pose_iter
    )
    # RMSD to the relaxed input (how much BackRub moved it)
    rmsd_to_input = pyrosetta.rosetta.core.scoring.all_atom_rmsd(
        pose_input, pose_iter
    )

    delta_from_input  = score - input_score
    delta_from_native = score - native_score

    print(f"  Score            : {score:.2f}")
    print(f"  ΔScore (vs input): {delta_from_input:.2f}")
    print(f"  RMSD to native   : {rmsd_to_native:.3f} Å")
    print(f"  RMSD to input    : {rmsd_to_input:.3f} Å")
    print(f"  Time             : {elapsed:.1f} s")

    fname_out = os.path.join(
        args.output_dir, f"backrub_{tag}_sample{i+1:04d}.pdb"
    )
    pose_iter.dump_pdb(fname_out)

    metadata.append({
        "sample":           i + 1,
        "tag":              tag,
        "mc_kt":            args.mc_kt,
        "ntrials":          args.ntrials,
        "input_score":      input_score,
        "sample_score":     score,
        "delta_from_input": delta_from_input,
        "delta_from_native":delta_from_native,
        "rmsd_to_native":   rmsd_to_native,
        "rmsd_to_input":    rmsd_to_input,
        "time_s":           elapsed,
        "output_pdb":       fname_out,
    })

# ── Save metadata ────────────────────────────────────────────────────────────
fields = list(metadata[0].keys())
fname_meta = os.path.join(args.output_dir, f"backrub_metadata_{tag}.tsv")

with open(fname_meta, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(metadata)

    print(f"\nDone. Metadata → {fname_meta}")
