import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os

df = pd.read_csv("crossdock_summary_table.csv")

# FIX: extract receptor identity (remove ligand part)
df["receptor_base"] = df["receptor_design_ligand_id"].str.split("__").str[0]

# Build proper pivot
pivot = df.pivot_table(
    index="receptor_base",
    columns="ligand_id",
    values="boltz2_affinity_probability_binary",
    aggfunc="mean"
)

ligands = ["DAMGO", "morphine", "naltrexone", "nitazene"]

# Create 2x3 subplot grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes = axes.flatten()

for ax, (l1, l2) in zip(axes, itertools.combinations(ligands, 2)):

    sub = pivot[[l1, l2]].dropna()

    print(f"{l1} vs {l2}: {len(sub)} points")

    ax.scatter(sub[l1], sub[l2], alpha=0.7)

    ax.set_xlabel(l1)
    ax.set_ylabel(l2)
    ax.set_title(f"{l1} vs {l2}")

    # diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

# Clean layout
plt.tight_layout()

# Save ONE figure
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/crossdock_all_pairs.png", dpi=300)

plt.show()
