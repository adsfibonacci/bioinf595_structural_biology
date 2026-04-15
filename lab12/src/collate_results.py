import os
import json
import pandas as pd
from glob import glob

BASE_DIR = "intermediate"
LIGANDS = ["DAMGO", "morphine", "naltrexone", "nitazene"]

def mean_affinity(data, prefix):
    vals = []
    for i in ["", "1", "2"]:
        key = f"{prefix}{i}"
        if key in data:
            vals.append(data[key])
    return sum(vals) / len(vals)

rows = []

for receptor_ligand in LIGANDS:

    csv_path = f"{BASE_DIR}/{receptor_ligand}/merged/final_ranked_designs/final_designs_metrics_30.csv"
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():

        design_rank = int(row["final_rank"])
        receptor_id = f"{receptor_ligand}_rank{design_rank}"

        receptor_seq = row["designed_sequence"]

        for lig in LIGANDS:

            complex_id = f"{receptor_id}__{lig}"

            json_path = (
                f"./boltz_results_{complex_id}/"
                f"predictions/{complex_id}/affinity_{complex_id}.json"
            )

            if not os.path.exists(json_path):
                print("Missing:", json_path)
                continue

            with open(json_path) as f:
                data = json.load(f)

            prob = mean_affinity(data, "affinity_probability_binary")
            val = mean_affinity(data, "affinity_pred_value")

            rows.append({
                "receptor_id": receptor_id,
                "ligand_id": lig,
                "receptor_design_ligand_id": complex_id,

                "receptor_design_rank": design_rank,
                "receptor_design_design_to_target_iptm": row["design_to_target_iptm"],
                "receptor_design_min_design_to_target_pae": row["min_design_to_target_pae"],

                "boltz2_affinity_probability_binary": prob,
                "boltz2_affinity_pred_value": val,
            })

out = pd.DataFrame(rows)
out.to_csv("crossdock_summary_table.csv", index=False)

print("DONE -> crossdock_summary_table.csv")
print("Rows:", len(out))
