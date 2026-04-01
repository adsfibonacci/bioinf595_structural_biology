import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import re
import os

import chemprop
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data.collate import collate_batch
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import AllChem, DataStructs

from molfeat.trans.fp import FPVecTransformer

import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from vae import VAE

MPNN_CKPT  = "./data/example_model_v2_regression_mol.ckpt"
SMILES_COL = "smiles"
N_CLUSTERS = 10

EXPERIMENTS = {
    "medium-mean": {
        "dataset": "data/medium_dataset.csv",
        "h_dim": 256,
        "l_dim": 64,
        "std_ckpt": "./lab10_vae/medium-mean-notp/checkpoints/epoch=22-step=10787.ckpt",
        "trip_ckpt": "./lab10_vae/medium-mean-tp/checkpoints/epoch=30-step=14539.ckpt",
    },
}

def is_valid_smiles(smi):
    if not isinstance(smi, str):
        return False
    if re.search(r'_null_', smi, re.IGNORECASE):
        return False
    mol = Chem.MolFromSmiles(smi)
    return mol is not None

def get_enc_dim(ckpt_path, dataset):
    mpnn = chemprop.models.MPNN.load_from_checkpoint(ckpt_path)
    mpnn.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    batch = next(iter(loader))

    with torch.no_grad():
        hidden = mpnn.encoding(batch.bmg, batch.V_d, batch.X_d, i=-1)
    return hidden.shape[-1]

def plot_embeddings_3(fps, mus_std, mus_trip, prefix):
    print(f"[{prefix}] Clustering fingerprints for reference labels...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    labels = kmeans.fit_predict(fps)

    print(f"[{prefix}] Running UMAP on input fingerprints...")
    reducer_fp = umap.UMAP(n_components=2, random_state=42)
    emb_fp = reducer_fp.fit_transform(fps)

    print(f"[{prefix}] Running UMAP on Standard VAE latent representations...")
    scaler_std = StandardScaler()
    emb_std = umap.UMAP(n_components=2, random_state=42).fit_transform(scaler_std.fit_transform(mus_std))

    print(f"[{prefix}] Running UMAP on Contrastive VAE latent representations...")
    scaler_trip = StandardScaler()
    emb_trip = umap.UMAP(n_components=2, random_state=42).fit_transform(scaler_trip.fit_transform(mus_trip))

    # Plotting 3 panels side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # 1. Fingerprints
    axes[0].scatter(emb_fp[:, 0], emb_fp[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    axes[0].set_title("Input Fingerprints (ECFP4)")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")

    # 2. Standard Latent Space
    axes[1].scatter(emb_std[:, 0], emb_std[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    axes[1].set_title("Standard VAE Latent Space")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")

    # 3. Contrastive Latent Space
    axes[2].scatter(emb_trip[:, 0], emb_trip[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    axes[2].set_title("Contrastive VAE (Triplet Loss) Latent Space")
    axes[2].set_xlabel("UMAP 1")
    axes[2].set_ylabel("UMAP 2")

    plt.suptitle(f"Embedding Analysis: {prefix}", fontsize=18)
    plt.tight_layout()
    
    os.makedirs("product", exist_ok=True)
    save_path = os.path.join("product", f"umap_{prefix}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[{prefix}] Saved {save_path}")

def load_model_from_ckpt(ckpt_path, enc_dim, h_dim, l_dim, fp_dim):
    model = VAE.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        enc_dim=enc_dim,
        h_dim=h_dim,
        l_dim=l_dim,
        fp_dim=fp_dim,
        ckpt_path=MPNN_CKPT,
        lr=1e-3, 
        strict=False
    )
    model.eval()
    model.cpu()
    return model

def main():
    torch.set_float32_matmul_precision("medium")
    fp_transformer = FPVecTransformer(kind="ecfp:4")

    for name, config in EXPERIMENTS.items():
        print(f"\n--- Analyzing Experiment: {name} ---")
        
        # 1. Load Dataset
        df = pd.read_csv(config["dataset"])
        df = df[df[SMILES_COL].apply(is_valid_smiles)].reset_index(drop=True)
        smiles_list = df[SMILES_COL].tolist()
        fps = fp_transformer(smiles_list)
        
        datapoints = [MoleculeDatapoint.from_smi(smi, y=fp.astype(np.float32)) for smi, fp in zip(smiles_list, fps)]
        dataset = MoleculeDataset(datapoints, SimpleMoleculeMolGraphFeaturizer())
        loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_batch, num_workers=4)

        enc_dim = get_enc_dim(MPNN_CKPT, dataset)
        fp_dim = dataset[0].y.shape[0]

        # 2. Load Both Models
        print(f"[{name}] Loading Standard VAE...")
        model_std = load_model_from_ckpt(config["std_ckpt"], enc_dim, config["h_dim"], config["l_dim"], fp_dim)
        
        print(f"[{name}] Loading Contrastive VAE...")
        model_trip = load_model_from_ckpt(config["trip_ckpt"], enc_dim, config["h_dim"], config["l_dim"], fp_dim)

        # 3. Extract Embeddings
        print(f"[{name}] Extracting representations...")
        fps_list, mus_std_list, mus_trip_list = [], [], []

        with torch.no_grad():
            for batch in loader:
                mu_std, _ = model_std.encoder(batch)
                mu_trip, _ = model_trip.encoder(batch)
                
                fps_list.append(batch.Y)
                mus_std_list.append(mu_std)
                mus_trip_list.append(mu_trip)

        fps_np = torch.cat(fps_list, dim=0).numpy()
        mus_std_np = torch.cat(mus_std_list, dim=0).numpy()
        mus_trip_np = torch.cat(mus_trip_list, dim=0).numpy()

        # 4. Generate the 3-panel UMAP
        plot_embeddings_3(fps_np, mus_std_np, mus_trip_np, name)

if __name__ == "__main__":
    main()
