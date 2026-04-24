import os
import glob
import multiprocessing

def sanitize_rosetta_pdb(input_pdb, output_dir_base="data/sanitized_mutated_apo_panels"):
    """
    Strips Rosetta-specific energy tables and non-standard lines from a PDB.
    Replicates the original subfolder structure in the new base directory.
    """
    # Extract the relative path parts to maintain the subfolder structure
    # Example: "data/mutated_apo_panels/1hpv/1hpv_WT.pdb" -> "1hpv/1hpv_WT.pdb"
    path_parts = os.path.normpath(input_pdb).split(os.sep)
    
    try:
        panel_idx = path_parts.index("mutated_apo_panels")
        sub_path = os.path.join(*path_parts[panel_idx+1:])
    except ValueError:
        # Fallback if the path doesn't match expected structure
        sub_path = os.path.basename(input_pdb)

    output_pdb = os.path.join(output_dir_base, sub_path)
    
    # Ensure the target sub-directory exists
    os.makedirs(os.path.dirname(output_pdb), exist_ok=True)
    
    # Checkpointing: Skip if already sanitized
    if os.path.exists(output_pdb):
        return f"[SKIP] {sub_path} already sanitized."

    valid_starts = ("ATOM", "HETATM", "TER", "END")
    
    try:
        with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
            for line in infile:
                if line.startswith(valid_starts):
                    outfile.write(line)
        return f"[DONE] {sub_path}"
    except Exception as e:
        return f"[ERROR] Failed {sub_path}: {e}"

if __name__ == "__main__":
    # Locate all PDB files recursively
    input_pattern = "data/mutated_apo_panels/**/*.pdb"
    if __debug__:
        input_pattern = "data/mutated_apo_panels/1fb7/*.pdb"
        pass
    
    pdb_files = glob.glob(input_pattern, recursive=True)
    
    if not pdb_files:
        print(f"[!] No PDB files found matching {input_pattern}")
    else:
        print(f"Found {len(pdb_files)} PDB files. Spinning up multiprocessing pool...\n")
        
        # Utilize CPU cores to parse text files rapidly
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(sanitize_rosetta_pdb, pdb_files)
            
        # Tally the results
        completed = sum(1 for r in results if "[DONE]" in r)
        skipped = sum(1 for r in results if "[SKIP]" in r)
        errors = sum(1 for r in results if "[ERROR]" in r)
        
        print("\n" + "="*50)
        print("SANITIZATION COMPLETE")
        print(f"Newly Processed: {completed}")
        print(f"Skipped:         {skipped}")
        print(f"Errors:          {errors}")
        print("="*50)
        print(f"\nAll clean PDBs are now available in data/sanitized_mutated_apo_panels/")
