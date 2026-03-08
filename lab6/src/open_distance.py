"""
open_distance.py
----------------
Measures the Ca-Ca distance between GPCRdb positions:
  2x46 = Leu86  (MC4R residue 86 in uniprot/author numbering)
  6x37 = Leu247 (MC4R residue 247 in uniprot/author numbering)

Adapted from lab 2 biotite measurement scripts.

Usage (from project root):
    python src/open_distance.py
"""

import os
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import biotite.structure.io.pdb as bpdb
import biotite.structure as struc

# ── Residue numbers (author/uniprot numbering) ────────────────────────────────
RES_2x46 = 86   # Leu86
RES_6x37 = 247  # Leu247

os.makedirs("results", exist_ok=True)

# ── Diagnostic: print residue range of first PDB to verify numbering ─────────
first_pdb = sorted(glob.glob("intermediates/relaxes/MC4R_relaxed_*.pdb"))[0]
_f = bpdb.PDBFile.read(first_pdb)
_atoms = bpdb.get_structure(_f, model=1)
_ca = _atoms[_atoms.atom_name == "CA"]
_res = sorted(set(_ca.res_id))
print(f"Residue range in {os.path.basename(first_pdb)}: {_res[0]} – {_res[-1]}")
print(f"Looking for res {RES_2x46} and {RES_6x37}")
for r in [RES_2x46, RES_6x37]:
    hit = _atoms[(_atoms.res_id == r) & (_atoms.atom_name == "CA")]
    if len(hit):
        print(f"  res {r}: {hit.res_name[0]} CA — FOUND")
    else:
        print(f"  res {r}: NOT FOUND — check numbering!")
print()

# ── Helper ────────────────────────────────────────────────────────────────────
def measure_open_distance(pdb_path):
    f     = bpdb.PDBFile.read(pdb_path)
    atoms = bpdb.get_structure(f, model=1)
    ca1   = atoms[(atoms.res_id == RES_2x46) & (atoms.atom_name == "CA")]
    ca2   = atoms[(atoms.res_id == RES_6x37) & (atoms.atom_name == "CA")]
    if len(ca1) == 0 or len(ca2) == 0:
        return float("nan")
    return float(struc.distance(ca1.coord[0], ca2.coord[0]))

# ── Load metadata ─────────────────────────────────────────────────────────────
relax_meta = {}
with open("intermediates/relaxes/relax_metadata.tsv") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        relax_meta[int(row["sample"])] = {
            "score": float(row["relaxed_score"]),
            "rmsd":  float(row["RMSD"]),
        }

backrub_meta = {}
for tsv in sorted(glob.glob("intermediates/backrub/backrub_metadata_*.tsv")):
    with open(tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = os.path.basename(row["output_pdb"])
            backrub_meta[key] = {
                "tag":     row["tag"],
                "mc_kt":   float(row["mc_kt"]),
                "ntrials": int(row["ntrials"]),
                "score":   float(row["sample_score"]),
                "rmsd":    float(row["rmsd_to_native"]),
            }

# ── Measure all structures ────────────────────────────────────────────────────
records = []

# Relaxed
for pdb_path in sorted(glob.glob("intermediates/relaxes/MC4R_relaxed_*.pdb")):
    sample = int(os.path.basename(pdb_path).split("_")[-1].replace(".pdb", ""))
    meta   = relax_meta[sample]
    dist   = measure_open_distance(pdb_path)
    print(f"[relax {sample}]  dist={dist:.2f} A  rmsd={meta['rmsd']:.3f}")
    records.append({
        "source":        "FastRelax",
        "mc_kt":         float("nan"),
        "ntrials":       float("nan"),
        "score":         meta["score"],
        "rmsd_native":   meta["rmsd"],
        "open_distance": dist,
    })

# BackRub
for pdb_path in sorted(glob.glob("intermediates/backrub/backrub_kt*_sample*.pdb")):
    key  = os.path.basename(pdb_path)
    meta = backrub_meta.get(key)
    if meta is None:
        continue
    dist = measure_open_distance(pdb_path)
    records.append({
        "source":        "BackRub",
        "mc_kt":         meta["mc_kt"],
        "ntrials":       meta["ntrials"],
        "score":         meta["score"],
        "rmsd_native":   meta["rmsd"],
        "open_distance": dist,
    })

print(f"\nTotal structures measured: {len(records)}")
valid = [r for r in records if r["open_distance"] == r["open_distance"]]
print(f"Valid (non-NaN) distances: {len(valid)}")

# ── Save TSV ──────────────────────────────────────────────────────────────────
with open("results/open_distance_metadata.tsv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["source","mc_kt","ntrials","score","rmsd_native","open_distance"],
        delimiter="\t"
    )
    writer.writeheader()
    writer.writerows(records)
print("Saved: results/open_distance_metadata.tsv")

# ── Plot: open distance vs RMSD to native ─────────────────────────────────────
relax_recs   = [r for r in records if r["source"] == "FastRelax"]
backrub_recs = [r for r in records if r["source"] == "BackRub"]

kt_vals  = sorted(set(r["mc_kt"] for r in backrub_recs))
colors   = plt.cm.plasma(np.linspace(0.1, 0.85, len(kt_vals)))
kt_color = dict(zip(kt_vals, colors))

fig, ax = plt.subplots(figsize=(8, 5.5))

for kt in kt_vals:
    sub = [r for r in backrub_recs if r["mc_kt"] == kt]
    x = [r["open_distance"] for r in sub]
    y = [r["rmsd_native"]   for r in sub]
    ax.scatter(x, y, color=kt_color[kt], alpha=0.55, s=28,
               linewidths=0, label=f"BackRub kT={kt}")

if relax_recs:
    ax.scatter(
        [r["open_distance"] for r in relax_recs],
        [r["rmsd_native"]   for r in relax_recs],
        color="#E76F51", marker="D", s=85, zorder=5,
        edgecolors="black", linewidths=0.5, label="FastRelax"
    )

ax.axvline(11.9, color="grey", linestyle="--", linewidth=1, alpha=0.7,
           label="Active threshold ~11.9 Å")

ax.set_xlabel(
    f"TM2–TM6 Ca distance: Leu{RES_2x46} (2x46) – Leu{RES_6x37} (6x37)  (Å)",
    fontsize=11
)
ax.set_ylabel("Heavy-atom RMSD to native 8QJ2  (Å)", fontsize=11)
ax.set_title(
    "MC4R Ensemble: TM6 Opening Distance vs RMSD to Native\n"
    "Distance < 11.9 Å = active-like; > 11.9 Å = inactive/intermediate",
    fontsize=12
)
ax.legend(fontsize=8, loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/open_distance_vs_rmsd.pdf", bbox_inches="tight")
plt.savefig("results/open_distance_vs_rmsd.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: results/open_distance_vs_rmsd.pdf/png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Open distance summary ────────────────────────────────────────")
for label, recs in [("FastRelax", relax_recs)] + [
        (f"BackRub kT={kt}", [r for r in backrub_recs if r["mc_kt"] == kt])
        for kt in kt_vals]:
    d = np.array([r["open_distance"] for r in recs])
    d = d[~np.isnan(d)]
    if len(d):
        print(f"{label}: mean={d.mean():.2f}  sd={d.std():.2f}  [{d.min():.2f}, {d.max():.2f}] A")
    else:
        print(f"{label}: all NaN — residues not found in PDB")
