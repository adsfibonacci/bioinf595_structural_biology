import sys
import time
import subprocess
print(sys.executable)
# packages = ["numpy", "pandas", "matplotlib", "pyright", "autogluon", "datamol", "molfeat"]
# subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

import numpy as np
import pandas as pd
import autogluon as ag
from autogluon.tabular import TabularDataset, TabularPredictor

import torch
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split

from datasets import load_dataset

import datamol as dm
from molfeat.calc import RDKitDescriptors2D
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


print("Hello, World!")

irwin_dataset = load_dataset("IrwinLab/LSD_NSP3_Mac1_Gahbauer_2022_Everted_ZINC22_screen", "docking_results")
irwin_df = irwin_dataset["train"].to_pandas()
irwin_df.to_csv("intermediates/irwin_ZINC2_screen.csv", index=False)
print("Saved to intermediates folder")

print("Loading full dataset")
irwin_df = pd.read_csv("intermediates/irwin_ZINC2_screen.csv")
names = ["small", "medium", "large"]
sizes = [int(1e3), int(1e4), int(1e5)]  

dfs = []

for size, name in zip(sizes, names):
    df = irwin_df.sample(n=size)
    dfs.append(df)
    df.to_csv(f"intermediates/irwin_{name}.csv", index=False)
    pass
print("Saved small, medium, and large sampled datasets")

names = ["small", "medium", "large"]

dfs = []
for name in names:
    dfs.append(pd.read_csv(f"intermediates/irwin_{name}.csv"))
    pass
print("Loaded small, medium, and large sampled datasets")

print("Log transforming scores")
mins = [ df["score"].min() for df in dfs ]

for i, df in enumerate(dfs):
    df["log_score_shift"] = np.log(df["score"] - mins[i] + 1)
    print(df["log_score_shift"].describe())
    pass

for df, name in zip(dfs, names):
    df.to_csv(f"intermediates/irwin_{name}_transformed.csv", index=False)
    pass

print("Log transformed all scores")

names = ['small', 'medium', 'large']
dfs = [ pd.read_csv(f"intermediates/irwin_{name}_transformed.csv") for name in names ]

print("Begin featurizing")
featurizer = FPVecTransformer(kind="ecfp-count", dtype=float)

smile = [ df['smiles'].tolist() for df in dfs ]

Xs = [
    [ featurizer(s).ravel() for s in df['smiles'] if s != "_null_"]
    for df in dfs
]
print("Finished featurs of smiles\nStarting response values")

ys = [
    df['log_score_shift'][df['smiles'] != "_null_"].values
    for df in dfs
]

print("Features computed for all datasets")
Xs = [np.array(X) for X in Xs]
ys = [np.array(y) for y in ys]

for i, (X, y) in enumerate(zip(Xs, ys)):
    print(X.shape)
    np.save(f"intermediates/x_data_{i}.npy", X)
    print(y.shape)
    np.save(f"intermediates/y_data_{i}.npy", y)      
    pass

names = ['small', 'medium', 'large']
dfs = [ pd.read_csv(f"intermediates/irwin_{name}_transformed.csv") for name in names ]
Xs = [ np.load(f"intermediates/x_data_{i}.npy") for i in range(len(dfs)) ]
ys = [ np.load(f"intermediates/y_data_{i}.npy") for i in range(len(dfs)) ]

time_limit = 1200

splits = []
label = 'target'

for X, y in zip(Xs, ys):
    X_train, X, y_train, y = train_test_split(X, y, train_size=0.6)
    X_test, X_val, y_test, y_val = train_test_split(X, y, train_size=0.5)

    train_df = pd.DataFrame(X_train)
    train_df[label] = y_train

    test_df = pd.DataFrame(X_test)
    test_df[label] = y_test

    val_df = pd.DataFrame(X_val)
    val_df[label] = y_val

    splits.append({"train": train_df, "test": test_df, "val": val_df})
    pass

train_df = splits[0]['train']
test_df = splits[0]['test']
val_df = splits[0]['val']

predictor = TabularPredictor(
        label=label,
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=f"product/ag_small"
    )
predictor.fit(
  train_data=train_df,
  tuning_data=test_df,
  use_bag_holdout=True,    
  presets="high_quality",
  time_limit=time_limit,
  num_bag_folds=5,
  num_stack_levels=1,
  ag_args_fit={'NN_TORCH': {'num_gpus': 1}},  # explicitly assign GPU
  dynamic_stacking=False,
  verbosity=0
)
print("Finished fitting")

lb = predictor.leaderboard(val_df, silent=True)
best_model = lb.iloc[0]
val, test = best_model['score_val'], best_model['score_test']
if val < test:
    print("High variance, implied overfitting")
elif val > test:
    print("High bias, implied underfitting")
else:
    print("Balanced")
    pass

y_val_true = val_df['target'].values
y_val_pred = predictor.predict(val_df)



plt.scatter(y_val_true, y_val_pred, alpha=0.5)
plt.plot([y_val_true.min(), y_val_true.max()],
         [y_val_true.min(), y_val_true.max()], 'r--')
plt.xlabel("True log_score_shift")
plt.ylabel("Predicted log_score_shift")
plt.title("Predicted vs True on Validation Set")
plt.savefig(f"product/small_plot.png")

calc = RDKitDescriptors2D()
featurizer = FPVecTransformer(kind="ecfp-count", dtype=float)
configs = [
    {"index": 0, "fp": calc, "preset": "high_quality", "bags":3},
    {"index": 1, "fp": featurizer, "preset": "high_quality", "bags":5},
    {"index": 1, "fp": calc, "preset": "high_quality", "bags":5},
    {"index": 2, "fp": calc, "preset": "high_quality", "bags":10}
]
Xs = []
ys = []
for v, config in enumerate(configs):
    df = dfs[config["index"]]
    # filter out invalid SMILES
    valid_mask = df['smiles'] != "_null_"
    smiles = df.loc[valid_mask, 'smiles'].tolist()
    # get the corresponding target values
    targets = df.loc[valid_mask, 'log_score_shift'].values
    
    # compute features in batch
    # Handle RDKitDescriptors2D differently from FPVecTransformer
    fp_instance = config["fp"]
    if isinstance(fp_instance, RDKitDescriptors2D):
        # Process each SMILES individually and stack results
        X = np.vstack([fp_instance(smi) for smi in smiles])
    else:
        # FPVecTransformer can handle lists directly
        X = fp_instance(smiles)
    
    print(f"Config {v}: Generated features shape {X.shape}, target shape {targets.shape}")
    
    # Split data: 60% train, 20% test, 20% val
    X_train, X_temp, y_train, y_temp = train_test_split(X, targets, train_size=0.6, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    train_df = pd.DataFrame(X_train)
    train_df[label] = y_train
    test_df = pd.DataFrame(X_test)
    test_df[label] = y_test
    val_df = pd.DataFrame(X_val)
    val_df[label] = y_val
    
    predictor = TabularPredictor(
        label=label,
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=f"product/ag_config_{config['index']}"          
    )
    predictor.fit(
        train_data=train_df,
        tuning_data=test_df,
        use_bag_holdout=True,
        presets=config["preset"],
        time_limit=time_limit,
        num_bag_folds=config["bags"],
        num_stack_levels=1,
        ag_args_fit={'NN_TORCH': {'num_gpus': 1}},
        dynamic_stacking=False,
        verbosity=0
    )
    lb = predictor.leaderboard(val_df, silent=True)
    best_model = lb.iloc[0]
    val_score, test_score = best_model['score_val'], best_model['score_test']
    if val_score < test_score:
        print("High variance, implied overfitting")
    elif val_score > test_score:
        print("High bias, implied underfitting")
    else:
        print("Balanced")
    
    y_val_true = val_df[label].values
    y_val_pred = predictor.predict(val_df)
    
    plt.scatter(y_val_true, y_val_pred, alpha=0.5)
    plt.plot([y_val_true.min(), y_val_true.max()],
             [y_val_true.min(), y_val_true.max()], 'r--')
    plt.xlabel("True log_score_shift")
    plt.ylabel("Predicted log_score_shift")
    plt.title("Predicted vs True on Validation Set")
    plt.savefig(f"product/plot_{config['index']}.png")
    plt.show()
