import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx
import biotite.structure as struc
import numpy as np
import pandas as pd

pdb_ids = ["7CR2", "7CR0"]
bcif_files = {}

for pdb_id in pdb_ids:
  bcif_file = rcsb.fetch(pdb_id, "bcif")
  bcif_files[pdb_id] = bcif_file
  print(f"{pdb_id} downloaded, saved at {bcif_file}")
  pass

for pdb_id, bcif_path in bcif_files.items():
  bcif_file = pdbx.BinaryCIFFile.read(bcif_path)
  atom_array = pdbx.get_structure(bcif_file, model=1, include_bonds=True)
  print(f"{pdb_id} structure loaded:")
  print(atom_array)
  pass

ser314_selection = (atom_array.res_id == 314) & (atom_array.res_name == "SER") & (atom_array.atom_name == "OG")
ser314_og_coords = atom_array[ser314_selection].coord

n = len(ser314_og_coords)

print("S314 OG coordinates:")
print(ser314_og_coords)

print("Distances between adjacent S314 OG atoms:")

for i in range(n):
    # Next atom in sequence, wrap around for last atom to first
    j = (i + 1) % n
    d = struc.distance(ser314_og_coords[i], ser314_og_coords[j])
    print(f"Distance between atom {i} and {j}: {d:.2f} A")
    pass

adjacent_coords = np.roll(ser314_og_coords, -1, axis=0)
distances = struc.distance(ser314_og_coords, adjacent_coords)
mean_distance = np.mean(distances)
print(f"Mean distance between adjacent S314 OG atoms: {mean_distance:.2f} A")

df = pd.read_csv("kcnq2_cryoem.csv", skiprows=1)

# Prepare a list to store results
results = []

# Iterate over each structure
for idx, row in df.iterrows():
    entry_id = row['Entry ID']
    print(f"Processing {entry_id}...")
    
    try:
        # Fetch PDB in bcif format
        bcif_path = rcsb.fetch(entry_id, "bcif")
        
        # Read the BinaryCIF file
        bcif_file = pdbx.BinaryCIFFile.read(bcif_path)
        
        # Get structure for model=1, include bonds
        atom_array = pdbx.get_structure(bcif_file, model=1, include_bonds=True)
        
        # Select S314 OG atoms (serine gate)
        ser314_selection = (atom_array.res_id == 314) & \
                           (atom_array.res_name == "SER") & \
                           (atom_array.atom_name == "OG")
        ser314_og_coords = atom_array[ser314_selection].coord
        
        # Skip if no OG atoms found
        if len(ser314_og_coords) < 2:
            print(f"Warning: less than 2 S314 OG atoms found in {entry_id}. Skipping.")
            continue
        
        # Compute mean distance between adjacent OG atoms (wrap-around)
        adjacent_coords = np.roll(ser314_og_coords, -1, axis=0)
        distances = struc.distance(ser314_og_coords, adjacent_coords)
        mean_distance = np.mean(distances)
        
        # Store result
        results.append({"Entry ID": entry_id, "Mean S314 OG Distance (Å)": mean_distance})
    
    except Exception as e:
        print(f"Error processing {entry_id}: {e}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Write to TSV
results_df.to_csv("mean_gate_distances.tsv", sep="\t", index=False)

print("Done! Results written to mean_gate_distances.tsv")
