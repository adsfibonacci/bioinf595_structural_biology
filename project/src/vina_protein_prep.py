import sys
import subprocess
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_ROOT = PROJECT_ROOT / "data/sanitized_mutated_apo_panels"
TARGET_ROOT = PROJECT_ROOT / "data/vinadock_panel"
LOG_FILE = TARGET_ROOT / "receptor_centers.csv"

# Ensure target root exists
TARGET_ROOT.mkdir(parents=True, exist_ok=True)

# Find all PDBs
pdb_files = list(SOURCE_ROOT.glob("**/*.pdb"))
print(f"Found {len(pdb_files)} proteins to prepare for Vina.")

parser = PDBParser(QUIET=True)

with open(LOG_FILE, "w") as log:
    # Write CSV header
    log.write("receptor_name,rel_path,center_x,center_y,center_z\n")
    
    for pdb_path in pdb_files:
        rel_path = pdb_path.relative_to(SOURCE_ROOT)
        output_path = TARGET_ROOT / rel_path.with_suffix(".pdbqt")
        
        # Mirror the subfolder structure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Calculate the Bounding Box Center
        try:
            struct = parser.get_structure('P', pdb_path)
            coords = np.array([atom.coord for atom in struct.get_atoms()])
            
            # Vina works best when the grid is centered on the extremeties of the protein
            min_c = coords.min(axis=0)
            max_c = coords.max(axis=0)
            center = (min_c + max_c) / 2.0
            
            # Log it
            log.write(f"{pdb_path.stem},{rel_path},{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}\n")
        except Exception as e:
            print(f"[!] Error calculating center for {pdb_path.name}: {e}")
            continue

        # 2. Convert to PDBQT
        if output_path.exists():
            print(f"[-] Skipping conversion (already exists): {rel_path.with_suffix('.pdbqt')}")
            continue
            
        print(f"[+] Processing and converting: {rel_path}")
        
        # Call OpenBabel
        # -xr: flags this as a rigid receptor
        # -p 7.4: adds polar hydrogens at physiological pH (required for Vina scoring)
        # --partialcharge gasteiger: calculates the Gasteiger charges Vina needs
        cmd = [
            "obabel", str(pdb_path), 
            "-O", str(output_path), 
            "-xr", "-p", "7.4", "--partialcharge", "gasteiger"
        ]
        
        # We suppress stderr because obabel spits out a lot of stereochemistry warnings
        subprocess.run(cmd, stderr=subprocess.DEVNULL)

print(f"\n--- Receptor preparation complete ---")
print(f"Centers logged to: {LOG_FILE}")
