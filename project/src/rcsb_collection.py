import os
import urllib.request
from rcsbapi.search import TextQuery

def fetch_hiv_protease_complexes(inhibitors, max_results_per_drug=10):
    """
    Queries the PDB for HIV-1 protease structures bound to specific inhibitors
    and downloads the corresponding .pdb files.
    """
    
    # Create a base directory for your raw data
    base_dir = "data/pdb_files"
    os.makedirs(base_dir, exist_ok=True)

    for drug in inhibitors:
        print(f"\n--- Searching for HIV-1 Protease bound to {drug} ---")
        
        # 1. Construct the Query
        search_term = f"HIV-1 protease {drug}"
        query = TextQuery(search_term)
        
        # Execute the query and convert the iterator to a list
        try:
            results = list(query())
            print(f"Found {len(results)} total matching structures in PDB.")
        except Exception as e:
            print(f"Query failed for {drug}: {e}")
            continue

        # 2. Set up the local directory for this specific drug
        drug_dir = os.path.join(base_dir, drug.lower())
        os.makedirs(drug_dir, exist_ok=True)

        # 3. Download the structures
        for pdb_id in results[:max_results_per_drug]:
            pdb_id_lower = pdb_id.lower()
            url = f"https://files.rcsb.org/download/{pdb_id_lower}.pdb"
            file_path = os.path.join(drug_dir, f"{pdb_id_lower}.pdb")
            
            if os.path.exists(file_path):
                print(f"[{pdb_id}] Already exists locally. Skipping.")
                continue

            try:
                print(f"Downloading {pdb_id}...")
                urllib.request.urlretrieve(url, file_path)
            except urllib.error.HTTPError as e:
                print(f"[{pdb_id}] Download failed. It may only be available as mmCIF. Error: {e}")
            except Exception as e:
                print(f"[{pdb_id}] Unexpected error: {e}")

if __name__ == "__main__":
    target_inhibitors = ["Ritonavir", "Indinavir", "Darunavir", "Saquinavir", "Tipranavir", "Nelfinavir", "Atazanavir", "Lopinavir", "Amprenavir"]
    fetch_hiv_protease_complexes(target_inhibitors, max_results_per_drug=160)
    print("\nData collection complete.")
