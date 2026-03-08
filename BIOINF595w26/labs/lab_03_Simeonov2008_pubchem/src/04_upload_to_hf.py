
import datasets
import yaml
import pandas as pd

huggingface_repo = ... 
# e.g. maomlab/Simeonov2008
# https://huggingface.co/datasets/maomlab/Simeonov2008


# Note that the data should be uplaoded to hugging face using the datasets package
# rather than just copying to the huggingface because then it can be downloaded using
# the huggingface package
dataset = datasets.load_dataset(
    "csv",
    data_files = f"product/Simeonov2008_compounds_sanitized_20260128.tsv",
    keep_in_memory = True,
    sep = "\t")

print(f"Pushing to {huggingface_repo}")
dataset.push_to_hub(
    repo_id = huggingface_repo)


hf_dataset = datasets.load_dataset(huggingface_repo)
df = hf_dataset['train'].to_pandas()
