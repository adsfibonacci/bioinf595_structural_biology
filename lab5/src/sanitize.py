import pandas as pd
import numpy as np

size = "large"

# Load data
dock_df = pd.read_csv(f"intermediates/irwin_{size}.csv")
umap_df = pd.read_csv(
    f"product/umap_{size}/umap_coordinates_{size}_maccs_cosine_10_0.25.csv"
)

# Merge
merged = pd.merge(
    umap_df,
    dock_df[["zincid", "score"]],
    on="zincid",
    how="inner"
)
print(merged.head())

# Replace with real predictions if you have them
np.random.seed(0)
merged["pred"] = merged["score"] + np.random.normal(0, 0.5, len(merged))

merged["residual"] = merged["score"] - merged["pred"]

# ============================================================
# 1️⃣ Docking score version
# ============================================================

interactive_score = merged[["UMAP_1", "UMAP_2", "score", "smiles"]].copy()
interactive_score = interactive_score.rename(columns={"score": "label"})
interactive_score.to_csv(f"product/umap_{size}/interactive_score_{size}.csv", index=False)

# ============================================================
# 2️⃣ Predicted score version
# ============================================================

interactive_pred = merged[["UMAP_1", "UMAP_2", "pred", "smiles"]].copy()
interactive_pred = interactive_pred.rename(columns={"pred": "label"})
interactive_pred.to_csv(f"product/umap_{size}/interactive_pred_{size}.csv", index=False)

# ============================================================
# 3️⃣ Residual version
# ============================================================

interactive_residual = merged[["UMAP_1", "UMAP_2", "residual", "smiles"]].copy()
interactive_residual = interactive_residual.rename(columns={"residual": "label"})
interactive_residual.to_csv(f"product/umap_{size}/interactive_residual_{size}.csv", index=False)

print("Saved:")
print(" - product/interactive_score_large.csv")
print(" - product/interactive_pred_large.csv")
print(" - product/interactive_residual_large.csv")
