import subprocess
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_ROOT = PROJECT_ROOT / "intermediate/testing_ligands"
TARGET_ROOT = PROJECT_ROOT / "intermediate/vinadock_ligands"

# Ensure target directory exists
TARGET_ROOT.mkdir(parents=True, exist_ok=True)

# Find all SDF files
sdf_files = list(SOURCE_ROOT.glob("**/*.sdf"))
print(f"Found {len(sdf_files)} ligands to prepare for Vina.")

for sdf_path in sdf_files:
    # Maintain any subfolder structure (if applicable)
    rel_path = sdf_path.relative_to(SOURCE_ROOT)
    output_path = TARGET_ROOT / rel_path.with_suffix(".pdbqt")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"[-] Skipping (already exists): {output_path.name}")
        continue
        
    print(f"[+] Converting: {rel_path}")
    
    # Call OpenBabel for Ligands
    # -p 7.4: Ensures proper protonation states for physiological pH
    # --partialcharge gasteiger: Adds the exact charge model Vina's scoring function expects
    cmd = [
        "obabel", str(sdf_path), 
        "-O", str(output_path), 
        "-p", "7.4", 
        "--partialcharge", "gasteiger"
    ]
    
    try:
        # Suppress stderr to keep the terminal clean from OpenBabel stereochemistry warnings
        subprocess.run(cmd, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Error converting {sdf_path.name}")

print(f"\n--- Ligand preparation complete ---")
print(f"Ready for Vina-GPU. Output saved to: {TARGET_ROOT}")
