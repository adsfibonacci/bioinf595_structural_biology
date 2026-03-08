#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import umap
from molfeat.trans.fp import FPVecTransformer
import datashader as ds
import holoviews as hv
import datashader.transfer_functions as tf
import colorcet as cc
from datashader.utils import export_image

import colorcet as cc
from holoviews.operation.datashader import rasterize

# ============================================================
# Utility: Clean SMILES
# ============================================================

def clean_smiles(df, smiles_col="smiles"):
    df = df.copy()
    df = df[df[smiles_col].notna()]
    df = df[df[smiles_col] != "_null_"]
    df = df[df[smiles_col].str.len() > 0]
    df = df.reset_index(drop=True)
    return df


# ============================================================
# Fingerprint Computation
# ============================================================

def compute_fingerprints(smiles_batch, featurizer):
    fps = featurizer(smiles_batch)
    if not isinstance(fps, np.ndarray):
        fps = np.array(fps)
    return fps


# ============================================================
# Incremental PCA Fit
# ============================================================

def fit_incremental_pca(df, smiles_col="smiles", config=None):
    featurizer = FPVecTransformer(kind=config["fp_kind"], dtype=float)
    pca_reducer = IncrementalPCA(n_components=config["pca_components"])

    n = len(df)
    for i in tqdm(range(0, n, config["batch_size"]), desc="Fitting PCA"):
        batch = df.iloc[i:i + config["batch_size"]]
        smiles_batch = batch[smiles_col].tolist()
        batch_features = compute_fingerprints(smiles_batch, featurizer)
        pca_reducer.partial_fit(batch_features)

    return pca_reducer, featurizer


# ============================================================
# PCA Transform in Batches
# ============================================================

def transform_pca_in_batches(df, pca_reducer, featurizer, smiles_col="smiles", config=None):
    n = len(df)
    embedding = None

    for i in tqdm(range(0, n, config["batch_size"]), desc="Projecting to PCA"):
        batch = df.iloc[i:i + config["batch_size"]]
        smiles_batch = batch[smiles_col].tolist()
        batch_features = compute_fingerprints(smiles_batch, featurizer)
        batch_pca = pca_reducer.transform(batch_features)

        if embedding is None:
            embedding = np.zeros((n, batch_pca.shape[1]), dtype=batch_pca.dtype)

        embedding[i:i + len(batch_pca)] = batch_pca

    return embedding


# ============================================================
# Fit UMAP
# ============================================================

def fit_umap_embedding(pca_embedding, config=None, random_state=42):
    umap_reducer = umap.UMAP(
        n_neighbors=config["n_neighbors"],
        min_dist=config["min_dist"],
        n_components=config["n_components"],
        metric=config["metric"],
        random_state=random_state,
        low_memory=True,
    )

    umap_embedding = umap_reducer.fit_transform(pca_embedding)
    return umap_reducer, umap_embedding


# ============================================================
# Full UMAP Pipeline
# ============================================================

def compute_umap_coordinates(df, smiles_col="smiles", id_col="substance_id", config=None):
    print("Cleaning SMILES...")
    df = clean_smiles(df, smiles_col)

    print("Step 1: Fitting Incremental PCA")
    pca_reducer, featurizer = fit_incremental_pca(df, smiles_col=smiles_col, config=config)

    print("Step 2: Projecting to PCA space")
    pca_embedding = transform_pca_in_batches(df, pca_reducer, featurizer, smiles_col=smiles_col, config=config)

    print("Step 3: Fitting UMAP")
    _, umap_embedding = fit_umap_embedding(pca_embedding, config=config)

    print("Step 4: Building output dataframe")
    output_df = df[[id_col, smiles_col]].copy()
    output_df["UMAP_1"] = umap_embedding[:, 0]
    output_df["UMAP_2"] = umap_embedding[:, 1]

    return output_df


# ============================================================
# Datashader Visualization with optional coloring
# ============================================================



def visualize_umap(
    df,
    x_col="UMAP_1",
    y_col="UMAP_2",
    width=1600,
    height=1400,
    output_fname="umap"
):
    """
    Faithful, large-scale UMAP visualization following
    Datashader best practices.
    """

    print(f"Visualizing {len(df)} points")

    # ---- Compute bounds ----
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()

    # ---- Aggregate to fixed pixel grid ----
    canvas = ds.Canvas(
        plot_width=width,
        plot_height=height,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
    )

    agg = canvas.points(df, x_col, y_col)

    # ---- Log transform to avoid underutilized dynamic range ----
    img = tf.shade(
        agg,
        cmap=cc.fire,     # perceptually uniform
        how="log"         # critical for large datasets
    )

    # ---- Improve visibility of sparse regions ----
    img = tf.dynspread(img, threshold=0.5, max_px=4)

    # ---- White background for publication ----
    img = tf.set_background(img, "white")

    # ---- Save image ----
    img.to_pil().convert("RGB").save(f"{output_fname}.png")

    print(f"Saved to {output_fname}.png")

hv.extension("bokeh")


def datashade_umap_by_value(
    df,
    value_col,
    x_col="UMAP_1",
    y_col="UMAP_2",
    cmap=cc.fire,
    width=900,
    height=900,
    output_fname="plot.png",
):

    print(f"Plotting {value_col} with {len(df)} points")

    points = hv.Points(
        df,
        kdims=[x_col, y_col],
        vdims=[value_col],
    )

    # Aggregate mean value per pixel
    agg = rasterize(
        points,
        width=width,
        height=height,
        aggregator=ds.mean(value_col),
    )

    shaded = tf.shade(
        agg.data,
        cmap=cmap,
        how="linear",
    )

    shaded = tf.dynspread(shaded, threshold=0.5, max_px=3)

    img = hv.RGB(shaded)

    plot = img.opts(
        width=width,
        height=height,
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        bgcolor="white",
        show_grid=True,
    )

    hv.save(plot, output_fname)
    print(f"Saved {output_fname}")    


def sweep():
    BASE = os.path.dirname(os.path.realpath(__file__))
    ROOT = os.path.abspath(os.path.join(BASE, ".."))

    INTER = os.path.join(ROOT, "intermediates")
    PROD = os.path.join(ROOT, "product")

    names = ['small', 'medium', 'large']
    sizes = [int(1e4), int(1e5), int(1e6)]
    idx = 0
    chosen_name = names[idx]
    chosen_size = sizes[idx]

    fp_types = ["ecfp-count", "maccs"]
    metric_types = ["cosine", "euclidean"]
    n_neighbor_types = [10, 50]
    min_dists = [0.1, 0.25, 0.5]

    HYPER = os.path.join(PROD, f"sweep_{chosen_name}")
    irwin_df = pd.read_csv(os.path.join(INTER, f"irwin_{chosen_name}.csv"))

    os.makedirs(INTER, exist_ok=True)
    os.makedirs(PROD, exist_ok=True)
    os.makedirs(HYPER, exist_ok=True)

    input_df = irwin_df.sample(n=chosen_size)

    for fp in fp_types:
        for metric in metric_types:
            for neighbor in n_neighbor_types:
                for dist in min_dists:
                    config = {
                        "fp_kind": fp,
                        "metric": metric,
                        "n_neighbors": neighbor,
                        "min_dist": dist,
                        "n_components": 2,
                        "batch_size": 4096,
                        "pca_components": 100,
                    }

                    umap_df = compute_umap_coordinates(
                        input_df,
                        smiles_col="smiles",
                        id_col="zincid",
                        config=config,
                    )

                    output_csv = os.path.join(HYPER, f"umap_coordinates_{chosen_name}_{fp}_{metric}_{neighbor}_{dist}.csv")
                    umap_df.to_csv(output_csv, index=False)
                    print(f"Saved coordinates to {output_csv}")

                    output_img = os.path.join(HYPER, f"umap_embedding_{chosen_name}_{fp}_{metric}_{neighbor}_{dist}.png")
                    visualize_umap(umap_df, output_fname=output_img)

    print("Done.")

def specific_values():
    BASE = os.path.dirname(os.path.realpath(__file__))
    ROOT = os.path.abspath(os.path.join(BASE, ".."))

    INTER = os.path.join(ROOT, "intermediates")
    PROD = os.path.join(ROOT, "product")

    names = ['small', 'medium', 'large']
    sizes = [int(1e4), int(1e5), int(1e6)]
    idx = 2
    chosen_name = names[idx]
    chosen_size = sizes[idx]

    fp_types = ["ecfp-count", "maccs"]
    metric_types = ["cosine", "euclidean"]
    n_neighbor_types = [10, 50]
    min_dists = [0.1, 0.25, 0.5]

    HYPER = os.path.join(PROD, f"umap_{chosen_name}")
    irwin_df = pd.read_csv(os.path.join(INTER, f"irwin_{chosen_name}.csv"))

    os.makedirs(INTER, exist_ok=True)
    os.makedirs(PROD, exist_ok=True)
    os.makedirs(HYPER, exist_ok=True)

    input_df = irwin_df.sample(n=chosen_size)

    fp, metric, neighbor, dist = "maccs", "cosine", 10, 0.25

    config = {
        "fp_kind": fp,
        "metric": metric,
        "n_neighbors": neighbor,
        "min_dist": dist,
        "n_components": 2,
        "batch_size": 4096,
        "pca_components": 100,
    }
    umap_df = compute_umap_coordinates(
        input_df,
        smiles_col="smiles",
        id_col="zincid",
        config=config,
    )
    
    output_csv = os.path.join(HYPER, f"umap_coordinates_{chosen_name}_{fp}_{metric}_{neighbor}_{dist}.csv")
    umap_df.to_csv(output_csv, index=False)
    print(f"Saved coordinates to {output_csv}")
    
    output_img = os.path.join(HYPER, f"umap_embedding_{chosen_name}_{fp}_{metric}_{neighbor}_{dist}.png")
    visualize_umap(umap_df, output_fname=output_img)
    
    


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    
    # True docking score
    datashade_umap_by_value(
        merged,
        value_col="score",
        cmap=cc.fire,
        output_fname="umap_true_score.png",
    )
    
    # Predicted docking score
    datashade_umap_by_value(
        merged,
        value_col="pred",
        cmap=cc.fire,
        output_fname="umap_pred_score.png",
    )
    
    # Residuals (use diverging colormap!)
    datashade_umap_by_value(
        merged,
        value_col="residual",
        cmap=cc.coolwarm,
        output_fname="umap_residuals.png",
    )
    
    
