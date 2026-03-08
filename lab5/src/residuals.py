#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf
import colorcet as cc

# ============================================================
# 1) LOAD DATA
# ============================================================

dock_df = pd.read_csv("intermediates/irwin_medium.csv")
umap_df = pd.read_csv(
    "product/umap_medium/umap_coordinates_medium_maccs_cosine_10_0.25.csv"
)

print("Dock columns:", dock_df.columns)
print("UMAP columns:", umap_df.columns)

# ============================================================
# 2) MERGE ON zincid
# ============================================================

merged = pd.merge(
    umap_df,
    dock_df[["zincid", "score"]],
    on="zincid",
    how="inner"
)

print("Merged shape:", merged.shape)

# ============================================================
# 3) ADD PREDICTIONS
# ============================================================

# 🔴 REPLACE THIS WITH YOUR REAL PREDICTIONS IF YOU HAVE THEM
np.random.seed(0)
merged["pred"] = merged["score"] + np.random.normal(0, 0.5, len(merged))

# ============================================================
# 4) COMPUTE RESIDUALS
# ============================================================

merged["residual"] = merged["score"] - merged["pred"]

print("Columns now:", merged.columns)

# ============================================================
# 5) PURE DATASHADER FUNCTION (ROBUST)
# ============================================================

def datashade_umap_by_value(
    df,
    value_col,
    cmap,
    width=1200,
    height=1200,
    output_fname="plot.png",
):

    print(f"Plotting {value_col} with {len(df)} points")

    # Compute bounds
    x_min, x_max = df["UMAP_1"].min(), df["UMAP_1"].max()
    y_min, y_max = df["UMAP_2"].min(), df["UMAP_2"].max()

    canvas = ds.Canvas(
        plot_width=width,
        plot_height=height,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
    )

    # Aggregate mean value per pixel
    agg = canvas.points(df, "UMAP_1", "UMAP_2", agg=ds.mean(value_col))

    # Shade
    img = tf.shade(
        agg,
        cmap=cmap,
        how="linear"
    )

    img = tf.dynspread(img, threshold=0.5, max_px=3)
    img = tf.set_background(img, "white")

    img.to_pil().save(output_fname)
    print(f"Saved {output_fname}")

# ============================================================
# 6) GENERATE PLOTS
# ============================================================

# Docking score
datashade_umap_by_value(
    merged,
    value_col="score",
    cmap=cc.fire,
    output_fname="product/umap_medium/umap_true_score_medium.png",
)

# Predicted score
datashade_umap_by_value(
    merged,
    value_col="pred",
    cmap=cc.fire,
    output_fname="product/umap_medium/umap_pred_score_medium.png",
)

# Residuals (diverging colormap!)
datashade_umap_by_value(
    merged,
    value_col="residual",
    cmap=cc.coolwarm,
    output_fname="product/umap_medium/umap_residual_medium.png",
)

print("Done.")
