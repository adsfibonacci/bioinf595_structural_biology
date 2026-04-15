import os
import pandas as pd

BASE_DIR = "intermediate"

LIGAND_SMILES = {
    "DAMGO": "C[C@@H](NC([C@@H](N)CC1=CC=C(O)C=C1)=O)C(NCC(N([C@H](C(NCCO)=O)CC2=CC=CC=C2)C)=O)=O",
    "morphine": "O[C@@H](C=C[C@@]1([H])[C@H]2CC3=CC=C4O)[C@@]5([H])[C@]1(CCN2C)C3=C4O5",
    "naltrexone": "O[C@]12[C@]3(CCN(CC4CC4)[C@@H]2C5)[C@](OC6=C3C5=CC=C6O)([H])C(CC1)=O",
    "nitazene": "CCN(CC)CCN1C(CC2=CC=CC=C2)=NC3=C1C=CC([N+]([O-])=O)=C3",
}

template = """sequences:
  - protein:
      id: A
      sequence: {protein}
  - ligand:
      id: B
      smiles: '{smiles}'
properties:
  - affinity:
      binder: B
"""

ligands = list(LIGAND_SMILES.keys())

for receptor_ligand in ligands:

    csv_path = f"{BASE_DIR}/{receptor_ligand}/merged/final_ranked_designs/final_designs_metrics_30.csv"
    df = pd.read_csv(csv_path)

    out_dir = f"{BASE_DIR}/crossdock/{receptor_ligand}"
    os.makedirs(out_dir, exist_ok=True)

    for _, row in df.iterrows():

        design_id = row["final_rank"]
        protein_seq = row["designed_sequence"]

        receptor_id = f"{receptor_ligand}_rank{design_id}"

        for lig in ligands:

            smiles = LIGAND_SMILES[lig]

            yaml_text = template.format(
                protein=protein_seq,
                smiles=smiles
            )

            out_file = f"{out_dir}/{receptor_id}__{lig}.yaml"

            with open(out_file, "w") as f:
                f.write(yaml_text)

print("Done generating 480 YAML specs")
