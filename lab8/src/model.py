import numpy as np
import pandas as pd

import wandb
import argparse
import optuna
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from molfeat.trans.fp import FPVecTransformer


# =========================
# Model
# =========================
class FingerprintNN(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# =========================
# Dataset
# =========================
class FingerprintDataset(Dataset):

    def __init__(self, df, smiles_col: str, score_col: str, fingerprint_fn):
        self.df = df.reset_index(drop=True)
        self.smiles_col = smiles_col
        self.score_col = score_col
        self.fingerprint_fn = fingerprint_fn

    def __len__(self) -> int:        
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return row[self.smiles_col], row[self.score_col]

    def collate_fn(self, batch):
        smiles_list, scores_list = zip(*batch)

        fps = self.fingerprint_fn(list(smiles_list))
        fps = np.asarray(fps)  # fast tensor conversion

        fingerprints = torch.from_numpy(fps).float()
        scores = torch.tensor(scores_list, dtype=torch.float32)

        return fingerprints, scores


# =========================
# Training Step
# =========================
def step(model, data_loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    n_batches = 0

    for fingerprints, scores in data_loader:
        fingerprints = fingerprints.to(device)
        scores = scores.to(device)

        if train:
            optimizer.zero_grad()

        preds = model(fingerprints)
        loss = criterion(preds, scores)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches else 0.0


# =========================
# Utilities
# =========================
def _load_table(path: str) -> pd.DataFrame:
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def filter_featurizable(df, smiles_col, fingerprint_fn):
    valid_indices = []
    smiles = df[smiles_col].tolist()

    for i, smi in enumerate(tqdm.tqdm(smiles, desc="Filtering molecules")):
        try:
            fp = fingerprint_fn([smi])[0]
            if fp is not None and len(fp) > 0:
                valid_indices.append(i)
        except Exception:
            continue

    print(f"Kept {len(valid_indices):,} / {len(df):,} molecules")
    return df.iloc[valid_indices].reset_index(drop=True)


# =========================
# Train Mode
# =========================
def train_model(args):
    device = torch.device(args.device)

    wandb.init(project=args.wandb_project, tags=args.wandb_tags, config=vars(args))

    fp_transformer = FPVecTransformer(kind=args.fingerprint_type)
    fingerprint_fn = lambda smiles_list: np.asarray(fp_transformer(smiles_list))

    train_df = filter_featurizable(_load_table(args.train_path), args.smiles_col, fingerprint_fn)
    val_df   = filter_featurizable(_load_table(args.val_path), args.smiles_col, fingerprint_fn)
    test_df  = filter_featurizable(_load_table(args.test_path), args.smiles_col, fingerprint_fn)

    train_dataset = FingerprintDataset(train_df, args.smiles_col, args.score_col, fingerprint_fn)
    val_dataset   = FingerprintDataset(val_df, args.smiles_col, args.score_col, fingerprint_fn)
    test_dataset  = FingerprintDataset(test_df, args.smiles_col, args.score_col, fingerprint_fn)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  collate_fn=train_dataset.collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    sample_smiles = [train_df.iloc[0][args.smiles_col]]
    input_dim = fingerprint_fn(sample_smiles).shape[1]

    model = FingerprintNN(input_dim, args.hidden_dim, args.n_layers, args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, args.n_epochs + 1):
        train_loss = step(model, train_loader, optimizer, criterion, device, True)
        val_loss   = step(model, val_loader,   optimizer, criterion, device, False)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    test_loss = step(model, test_loader, optimizer, criterion, device, False)
    wandb.log({"test_loss": test_loss})
    wandb.finish()


# =========================
# Optuna Objective
# =========================
def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 512, 1024, 2048])
    n_layers   = trial.suggest_categorical("n_layers",   [1, 3, 5, 7])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 512, 1024, 2048])
    learning_rate = trial.suggest_categorical("learning_rate", [1, 3, 10, 30, 100, 300]) * 1e-4
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- W&B run per trial ----
    wandb.init(
        project="fpnn_optuna",
        name=f"trial_{trial.number}",
        config={
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "trial_number": trial.number,
        },
        reinit=True,
    )

    train_df = pd.read_csv("intermediates/small_train.csv")
    val_df   = pd.read_csv("intermediates/small_val.csv")

    fp_transformer = FPVecTransformer(kind="ecfp:4")
    fingerprint_fn = lambda smiles_list: np.asarray(fp_transformer(smiles_list))

    train_dataset = FingerprintDataset(train_df, "smiles", "score", fingerprint_fn)
    val_dataset   = FingerprintDataset(val_df,   "smiles", "score", fingerprint_fn)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=train_dataset.collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

    sample_smiles = [train_df.iloc[0]["smiles"]]
    input_dim = fingerprint_fn(sample_smiles).shape[1]

    model = FingerprintNN(input_dim, hidden_dim, n_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    n_epochs = 150
    for epoch in range(1, n_epochs + 1):
        train_loss = step(model, train_loader, optimizer, criterion, device, True)
        val_loss   = step(model, val_loader,   optimizer, criterion, device, False)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    wandb.log({"final_val_loss": val_loss})
    wandb.finish()

    return val_loss


# =========================
# Optuna Study
# =========================
def optuna_study(n_trials=50):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("\nBest trial:")
    print(study.best_trial.params)
    print("Best val loss:", study.best_value)


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FingerprintNN Trainer / Optuna Sweep")

    parser.add_argument("--mode", type=str, choices=["train", "optuna"], default="train")

    # Paths
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--val_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--smiles_col", type=str)
    parser.add_argument("--score_col", type=str)
    parser.add_argument("--fingerprint_type", type=str, default="ecfp:4")

    # Model
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--n_epochs", type=int)

    # W&B
    parser.add_argument("--wandb_project", type=str, default="fpnn_project")
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--n_trials", type=int, default=50)
    # Device
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(args)
    elif args.mode == "optuna":
        optuna_study(args.n_trials)
