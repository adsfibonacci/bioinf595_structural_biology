import os
import urllib.request
from rcsbapi.search import TextQuery


# -----------------------------
# FASTA extractor from PDB ATOM records
# -----------------------------
AA_MAP = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

def extract_sequence_from_pdb(file_path):
    seq = []
    seen = set()

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resn = line[17:20].strip()
                resi = line[22:26].strip()

                key = (resi, resn)
                if key in seen:
                    continue

                seen.add(key)

                if resn in AA_MAP:
                    seq.append(AA_MAP[resn])

    return "".join(seq)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def fetch_hiv_protease_complexes(inhibitors, max_results_per_drug=10):

    base_dir = "data/pdb_files"
    fasta_path = "data/dataset_sequences.fa"
    os.makedirs(base_dir, exist_ok=True)

    all_sequences = {}

    for drug in inhibitors:
        print(f"\n--- Searching for HIV-1 Protease bound to {drug} ---")

        query = TextQuery(f"HIV-1 protease {drug}")

        try:
            results = list(query())
            print(f"Found {len(results)} structures.")
        except Exception as e:
            print(f"Query failed for {drug}: {e}")
            continue

        drug_dir = os.path.join(base_dir, drug.lower())
        os.makedirs(drug_dir, exist_ok=True)

        for pdb_id in results[:max_results_per_drug]:
            pdb_id_lower = pdb_id.lower()

            file_path = os.path.join(drug_dir, f"{pdb_id_lower}.pdb")

            if not os.path.exists(file_path):
                try:
                    print(f"Downloading {pdb_id}...")
                    urllib.request.urlretrieve(
                        f"https://files.rcsb.org/download/{pdb_id_lower}.pdb",
                        file_path
                    )
                except Exception as e:
                    print(f"[{pdb_id}] download failed: {e}")
                    continue

            # -----------------------------
            # EXTRACT SEQUENCE
            # -----------------------------
            seq = extract_sequence_from_pdb(file_path)

            # -----------------------------
            # FILTER BY LENGTH
            # -----------------------------
            if not (70 <= len(seq) <= 120):
                print(f"[{pdb_id}] Rejected (length {len(seq)}). Deleting file.")
                os.remove(file_path)
                continue

            # keep sequence
            all_sequences[pdb_id] = seq
            print(f"[{pdb_id}] Accepted (length {len(seq)}).")

    # -----------------------------
    # WRITE FASTA
    # -----------------------------
    with open(fasta_path, "w") as f:
        for pdb_id, seq in all_sequences.items():
            f.write(f">{pdb_id}\n")
            f.write(seq + "\n")

    print(f"\nSaved {len(all_sequences)} sequences to {fasta_path}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":

    target_inhibitors = [
        "Ritonavir",
        "Indinavir",
        "Darunavir",
        "Saquinavir",
        "Tipranavir",
        "Nelfinavir",
        "Atazanavir",
        "Lopinavir",
        "Amprenavir"
    ]

    if __debug__:
        target_inhibitors = target_inhibitors[0:1]

    fetch_hiv_protease_complexes(target_inhibitors, max_results_per_drug=160)

    print("\nData collection complete.")
