import sys
import subprocess
print(sys.executable)
packages = ["numpy", "pandas", "rdkit", "datasets", "huggingface_hub", "pyright"]
subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

import datasets
from datasets import Dataset, logging
from huggingface_hub import login, HfApi
logging.set_verbosity_error()

df = pd.read_csv("intermediate/active_compounds.tsv", sep="\t")
normalizer = rdMolStandardize.Normalizer()
reionizer = rdMolStandardize.Reionizer()
fragment_remover = rdMolStandardize.FragmentRemover()

def sanitize_smiles(smiles):
  try:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
      return None
    mol = normalizer.normalize(mol)
    mol = reionizer.reionize(mol)
    mol = fragment_remover.remove(mol)
    return Chem.MolToSmiles(mol, canonical=True)
  except Exception as e:
    print(f"Error sanitizing SMILES {smiles}: {e}")
    return None
  return
df['SMILES_sanitized'] = df['SMILES'].apply(sanitize_smiles)

changed_count = (df['SMILES'] != df['SMILES_sanitized']).sum()
total_count = len(df)

df = df.dropna(subset=['SMILES_sanitized'])

print(f"Total molecules: {total_count}")
print(f"SMILES changed after sanitization: {changed_count}")
print(f"Percentage changed: {changed_count/total_count*100:.2f}%")
df.to_csv("intermediate/active_compounds_sanitized.tsv", sep="\t", index=False)
df.to_csv("product/Simeonov2008_compounds_sanitized_2026128.tsv", sep="\t", index=False)
print("Sanitized SMILES saved to 'active_compounds_sanitized.tsv'")

huggingface_repo = "adsfibonacci/Simeonov2008"
dataset = datasets.load_dataset(
    "csv",
    data_files="product/Simeonov2008_compounds_sanitized_2026128.tsv",
    sep="\t",
    keep_in_memory=True
)
dataset.push_to_hub(
    repo_id = huggingface_repo)

hf_dataset = datasets.load_dataset(huggingface_repo)
df = hf_dataset['train'].to_pandas()
print("Completed dataset creation")
