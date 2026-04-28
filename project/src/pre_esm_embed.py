import os
import torch
import esm
from pathlib import Path
from Bio.PDB import PDBParser

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_ROOT = PROJECT_ROOT / "data/sanitized_mutated_apo_panels"
TARGET_ROOT = PROJECT_ROOT / "data/esm_embeddings"

# 1. Setup Model
print("Loading ESM-2 model onto GPU... (This takes a moment)")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the exact model DiffDock uses (ESM2 650M)
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()  # Set to evaluation mode
batch_converter = alphabet.get_batch_converter()

# 2. Find PDBs
pdb_files = list(SOURCE_ROOT.glob("**/*.pdb"))
print(f"Found {len(pdb_files)} proteins.")

parser = PDBParser(QUIET=True)
d3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H',
         'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
         'TYR':'Y','VAL':'V'}

# 3. Process Embeddings
with torch.no_grad():
    for pdb_path in pdb_files:
        rel_path = pdb_path.relative_to(SOURCE_ROOT)
        output_path = TARGET_ROOT / rel_path.with_suffix(".pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            print(f"[-] Skipping {pdb_path.name}")
            continue

        print(f"[+] Embedding: {rel_path}")

        # Extract sequence from PDB
        structure = parser.get_structure('protein', pdb_path)
        residues = structure.get_residues()
        sequence = "".join([d3to1.get(res.get_resname(), '') for res in residues if res.get_resname() in d3to1])

        if not sequence:
            print(f"[!] Warning: No sequence found for {pdb_path.name}")
            continue

        # Prepare data for ESM model
        data = [(pdb_path.stem, sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        # Run inference
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_reps = results["representations"][33]

        # Remove the <cls> and <eos> tokens (index 0 and end)
        seq_rep = token_reps[0, 1 : len(sequence) + 1].clone().cpu()

        # Save in the exact format DiffDock's loader expects
        output_dict = {
            "label": pdb_path.stem,
            "representations": {33: seq_rep}
        }
        torch.save(output_dict, output_path)

print("\n--- Batch ESM computation complete ---")
