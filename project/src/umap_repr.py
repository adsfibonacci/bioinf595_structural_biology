import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import os
import pathlib
import subprocess
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
from scipy.interpolate import griddata

# Import your existing utilities
from fragment_action import fold_3d, sanitize_smiles
from mcts import VinaScorer

# Silence RDKit
RDLogger.DisableLog('rdApp.*')

# --- CONFIGURATION ---
FDA_DRUGS = {
    "Amprenavir": "CC(C)CN(CC(C(CC1=CC=CC=C1)NC(=O)OC2CCOC2)O)S(=O)(=O)C3=CC=C(C=C3)N",
    "Darunavir": "CC(C)CN(CC(C(CC1=CC=CC=C1)NC(=O)OC2C3COCC3OC2)O)S(=O)(=O)C4=CC=C(C=C4)N",
    "Lopinavir": "CC1=C(C(=CC=C1)C)OCC(=O)NC(CC2=CC=CC=C2)C(CC(CC3=CC=CC=C3)NC(=O)C(C(C)C)N4CCCC4=O)O",
    "Ritonavir": "CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NC(CC2=CC=CC=C2)C(CC(CC3=CC=CC=C3)NC(=O)OCC4=CN=C(S4)C(C)C)O",
    "Tipranavir": "CCCC1=C(C(CC(C1)C2=CC(=CC=C2)S(=O)(=O)NC3=C(C=CC(=C3)C(F)(F)F)F)O)C4=C(O)C(=O)C(=C(C4)CC)C",
    "Atazanavir": "CC(C)(C)C(C(=O)NC(CC1=CC=CC=C1)C(CN(CC2=CC=CC=C2)NC(=O)C(C(C)(C)C)NC(=O)OC)O)NC(=O)OC",
    "Indinavir": "CC(C)(C)NC(=O)C1CN(CC(C1)CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)C(=O)NC4C5CCCCC5CC4O)O",
    "Nelfinavir": "CC1=C(C=CC=C1O)C(=O)NC(CS2=CC=CC=C2)C(CN3C4CCCCC4CC3C(=O)NC(C)C)O",
    "Saquinavir": "CC(C)(C)NC(=O)C1CC2CCCCC2CN1CC(C(CC3=CC=CC=C3)NC(=O)C(CC(N)=O)NC(=O)C4=NC5=CC=CC=C5C=C4)O"
}

# --- ANALYSIS CORE ---
def get_canonical(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

def run_comprehensive_analysis():
    df = pd.read_csv("results/mcts_round_140.csv").drop_duplicates(subset='SMILES')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    print("[*] Generating fingerprints and UMAP...")
    df['can_smiles'] = df['SMILES'].apply(get_canonical)
    fps = [np.array(gen.GetFingerprint(Chem.MolFromSmiles(s))) for s in df['SMILES']]
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, metric='euclidean')
    coords = reducer.fit_transform(np.array(fps))
    df['u1'], df['u2'] = coords[:, 0], coords[:, 1]
    df = df.dropna(subset=['u1', 'u2', 'Score'])
    # Prep Level Sets
    gx, gy = np.mgrid[df.u1.min():df.u1.max():100j, df.u2.min():df.u2.max():100j]
    gz = griddata((df.u1, df.u2), df.Score, (gx, gy), method='linear')

    # 1. Raw Plot
    plt.figure(figsize=(10, 8))
    plt.contour(gx, gy, gz, levels=10, colors='gray', alpha=0.2)
    sc = plt.scatter(df.u1, df.u2, c=df.Score, cmap='magma', s=15, alpha=0.6)
    plt.colorbar(sc, label='Vina Score')
    plt.title('Raw Chemical Space Map')
    plt.grid(True, alpha=0.2)
    plt.savefig("results/umap_1_raw.png", dpi=300)
    plt.close()

    # 2. FDA Labeled
    plt.figure(figsize=(10, 8))
    plt.contour(gx, gy, gz, levels=10, colors='gray', alpha=0.2)
    sc = plt.scatter(df.u1, df.u2, c=df.Score, cmap='magma', s=15, alpha=0.6)
    plt.colorbar(sc, label='Vina Score')
    for name, smi in FDA_DRUGS.items():
        row = df[df['can_smiles'] == get_canonical(smi)]
        if not row.empty:
            plt.scatter(row.u1, row.u2, marker='*', s=350, color='cyan', edgecolors='black', label=name, zorder=5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('FDA Reference Locations')
    plt.savefig("results/umap_2_fda_labeled.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Top Scorers
    plt.figure(figsize=(10, 8))
    plt.contour(gx, gy, gz, levels=10, colors='gray', alpha=0.2)
    sc = plt.scatter(df.u1, df.u2, c=df.Score, cmap='magma', s=15, alpha=0.6)
    plt.colorbar(sc, label='Vina Score')
    top_50 = df.sort_values("Score", ascending=False).head(50)
    # Overlay Top 50 with distinct edges to pop over the magma background
    plt.scatter(top_50.u1, top_50.u2, c=top_50.Score, cmap='spring', s=80, edgecolors='black', linewidth=1, zorder=5)
    plt.title('Top 50 Binding Affinity Leads')
    plt.savefig("results/umap_3_top_scorers.png", dpi=300)
    plt.close()

    # 4. Affinity Neighbors
    plt.figure(figsize=(10, 8))
    plt.contour(gx, gy, gz, levels=10, colors='gray', alpha=0.2)
    sc = plt.scatter(df.u1, df.u2, c=df.Score, cmap='magma', s=15, alpha=0.6)
    plt.colorbar(sc, label='Vina Score')
    print("\nTop 10 Similar Affinity Neighbors to FDA Drugs:")
    for name, smi in FDA_DRUGS.items():
        fda_score = df[df['can_smiles'] == get_canonical(smi)]['Score'].values
        if len(fda_score) > 0:
            target = fda_score[0]
            # Find 10 molecules with closest Score value
            df['score_diff'] = (df['Score'] - target).abs()
            neighbors = df.nsmallest(11, 'score_diff') # 11 because it includes the drug itself
            # Use white edges so they stand out clearly
            plt.scatter(neighbors.u1, neighbors.u2, s=60, edgecolors='white', linewidth=0.8, alpha=0.9, label=f"Near {name}", zorder=4)
            print(f"--- {name} (Score: {target:.2f}) ---")
            print(neighbors[['SMILES', 'Score']].head(5))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.title('Molecules with Scores Most Similar to FDA Drugs')
    plt.savefig("results/umap_4_affinity_similarity.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\n[*] All plots saved to results/ directory.")
    
    # --- ADDED LINE: Save top 50 to CSV ---
    top_50[['SMILES', 'Score']].to_csv("results/top_50_leads.csv", index=False)
    
    return top_50['SMILES'].tolist()

if __name__ == "__main__":
    leads = run_comprehensive_analysis()
    # Execute high-fidelity validation on Top 10 Leads
    scorer = VinaScorer(os.getcwd())
    _, detailed = scorer.evaluate_rollout(leads[:10], Q_proteins=5, run_id="final_val")
    pd.DataFrame(list(detailed.items()), columns=['SMILES', 'Val_Score']).to_csv("results/final_validated_leads.csv", index=False)
    print("[*] Re-docking complete. Check results/final_validated_leads.csv")
