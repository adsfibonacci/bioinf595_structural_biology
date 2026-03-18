import pandas as pd
import os

os.makedirs("debug_data", exist_ok=True)

SEED = 42

# Load the full small datasets
train_df = pd.read_csv("intermediates/small_train.csv")
val_df   = pd.read_csv("intermediates/small_val.csv")
test_df  = pd.read_csv("intermediates/small_test.csv")

# Sample 1,000 rows each
train_sample = train_df.sample(n=min(1000, len(train_df)), random_state=SEED)
val_sample   = val_df.sample(n=min(1000, len(val_df)), random_state=SEED)
test_sample  = test_df.sample(n=min(1000, len(test_df)), random_state=SEED)

# Save for quick training runs
train_sample.to_csv("debug_data/train_1k.csv", index=False)
val_sample.to_csv("debug_data/val_1k.csv", index=False)
test_sample.to_csv("debug_data/test_1k.csv", index=False)
