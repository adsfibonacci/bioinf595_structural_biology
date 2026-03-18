from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

ds = load_dataset("IrwinLab/LSD_NSP3_Mac1_Gahbauer_2022_Everted_ZINC22_screen", "docking_results")
df = ds["train"].to_pandas()
ds["train"].to_csv("data/full_dataset.csv")

print("Splitting datasets")
df_small = df.sample(n=10000)
df_medium = df.sample(n=100000)
df_large = df.sample(n=1000000)

df_small.to_csv("data/small_dataset.csv")
df_medium.to_csv("data/medium_dataset.csv")
df_large.to_csv("data/large_dataset.csv")

def splits(df, name):
    train, temp = train_test_split(df, test_size=0.4, random_state=42, shuffle=True)
    val, test = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True)
    train.to_csv(f"intermediates/{name}_train.csv", index=False)
    val.to_csv(f"intermediates/{name}_val.csv", index=False)
    test.to_csv(f"intermediates/{name}_test.csv", index=False)
    return

splits(df_small, "small")
splits(df_medium, "medium")
splits(df_large, "large")

