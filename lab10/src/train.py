import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random
import re

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

import chemprop
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data.collate import collate_batch
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from molfeat.trans.fp import FPVecTransformer

import faiss

from sklearn.model_selection import train_test_split

from vae import VAE

def get_enc_dim(ckpt_path, dataset):
    mpnn = chemprop.models.MPNN.load_from_checkpoint(ckpt_path)
    mpnn.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    batch = next(iter(loader))

    with torch.no_grad():
        hidden = mpnn.encoding(batch.bmg, batch.V_d, batch.X_d, i=-1)
        pass

    return hidden.shape[-1]

def is_valid_smiles(smi):
    if not isinstance(smi, str):
        return False
    if re.search(r'_null_', smi, re.IGNORECASE):
        return False
    mol = Chem.MolFromSmiles(smi)
    return mol is not None

def clean_df(df, smiles_col):
    mask = df[smiles_col].apply(is_valid_smiles)
    df = df[mask].reset_index(drop=True)
    print(f"Kept {len(df)} rows after SMILES validation")
    return df

def build_dataset(df, smiles_col, fp_transformer):
    smiles_list = df[smiles_col].tolist()
    fps = fp_transformer(smiles_list)

    datapoints = []
    for smi, fp in zip(smiles_list, fps):
        dp = MoleculeDatapoint.from_smi(
            smi,
            y=fp.astype(np.float32)
        )
        datapoints.append(dp)
        pass

    featurizer = SimpleMoleculeMolGraphFeaturizer()
    dataset = MoleculeDataset(datapoints, featurizer)

    return dataset, fps

def build_triplets(fps, k=10):
    dim = fps.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(fps.astype(np.float32))

    D, I = index.search(fps.astype(np.float32), k)

    triplets = []
    N = len(fps)

    for i in range(N):
        neighbors = [j for j in I[i] if j != i]
        positive = random.choice(neighbors)

        negative = random.randint(0, N - 1)
        while negative == i:
            negative = random.randint(0, N - 1)
            pass

        triplets.append((i, positive, negative))
        pass

    return triplets

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, triplets):
        self.base = base_dataset
        self.triplets = triplets
        return

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]
        return {
            "anchor": self.base[a],
            "positive": self.base[p],
            "negative": self.base[n]
        }

    pass

def collate_fn(batch):
    anchors = [b["anchor"] for b in batch]
    positives = [b["positive"] for b in batch]
    negatives = [b["negative"] for b in batch]

    return {
        "anchor": collate_batch(anchors),
        "positive": collate_batch(positives),
        "negative": collate_batch(negatives)
    }

def main():
    torch.set_float32_matmul_precision("medium")

    df = pd.read_csv("data/medium_dataset.csv")
    df = clean_df(df, "smiles")

    train_df, temp = train_test_split(df, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42)

    smiles_col = "smiles"
    ckpt_path = "./data/example_model_v2_regression_mol.ckpt"

    fp_transformer = FPVecTransformer(kind="ecfp:4")

    train_dataset, train_fps = build_dataset(train_df, smiles_col, fp_transformer)
    val_dataset, val_fps = build_dataset(val_df, smiles_col, fp_transformer)
    test_dataset, test_fps = build_dataset(test_df, smiles_col, fp_transformer)

    train_triplets = build_triplets(train_fps)
    val_triplets = build_triplets(val_fps)
    test_triplets = build_triplets(test_fps)

    train_data = TripletDataset(train_dataset, train_triplets)
    val_data = TripletDataset(val_dataset, val_triplets)
    test_data = TripletDataset(test_dataset, test_triplets)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=15)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=15)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=15)

    enc_dim = get_enc_dim(ckpt_path, train_dataset)
    fp_dim = train_dataset[0].y.shape[0]

    model = VAE(
        enc_dim=enc_dim,
        h_dim=256,
        l_dim=64,
        fp_dim=fp_dim,
        ckpt_path=ckpt_path,
        lr=1e-3,
        lambda_triplet=1.0
    )

    early_stopping = EarlyStopping(
        monitor="val_kl",
        patience=10,
        mode="min"
    )

    logger = WandbLogger(project="lab10_vae", name="medium-mean-tp")

    trainer = L.Trainer(
        max_epochs=200,
        logger=logger,
        accelerator="auto",
        devices=1,
        callbacks=[early_stopping],
        log_every_n_steps=1
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    return

if __name__ == "__main__":
    main()
    pass
