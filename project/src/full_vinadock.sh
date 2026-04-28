#!/bin/bash
ulimit -s 32768

# --- 1. Configuration & Absolute Paths ---
PROJECT_ROOT=$(pwd)
VINA_DIR="$HOME/.aur/Vina-GPU-2.1/QuickVina2-GPU-2.1"
EXECUTABLE="./QuickVina2-GPU-2-1"

# Pointing directly to the full ligand directory now
LIGAND_DIR="$PROJECT_ROOT/intermediate/vinadock_ligands"
RECEPTOR_DIR="$PROJECT_ROOT/data/vinadock_panel"

# Updated output directories for the production run
RESULTS_DIR="$PROJECT_ROOT/intermediate/vina_gpu_docking_full"
CENTERS_CSV="$RECEPTOR_DIR/receptor_centers.csv"
SUMMARY_CSV="$RESULTS_DIR/full_affinity_matrix.csv"

# --- 2. Setup Environment ---
mkdir -p "$RESULTS_DIR"
echo "receptor_mutant,ligand_name,affinity_kcal_mol" > "$SUMMARY_CSV"

echo "Starting full docking pipeline..."
echo "Ligands directory: $LIGAND_DIR"
echo "Receptors listed in: $CENTERS_CSV"
echo "Results will be saved to: $RESULTS_DIR"

# --- 3. Iterate Over All Receptors ---
# tail -n +2 drops the header row of the CSV so we read the whole file
tail -n +2 "$CENTERS_CSV" | while IFS=',' read -r rec_name rel_path cx cy cz; do
    
    # Skip any trailing empty lines in the CSV
    [ -z "$rec_name" ] && continue

    # Clean up line endings (Windows carriage returns)
    cz=$(echo "$cz" | tr -d '\r')
    
    rel_pdbqt="${rel_path%.pdb}.pdbqt"
    rec_file="$RECEPTOR_DIR/$rel_pdbqt"
    out_dir="$RESULTS_DIR/$rec_name"
    
    mkdir -p "$out_dir"
    
    echo "======================================================"
    echo "DOCKING: $rec_name"
    echo "Center: X=$cx, Y=$cy, Z=$cz"
    echo "======================================================"
    
    # Move into the Vina-GPU directory
    cd "$VINA_DIR" || exit
    
    # Run QuickVina2-GPU
    $EXECUTABLE \
        --receptor "$rec_file" \
        --ligand_directory "$LIGAND_DIR" \
        --output_directory "$out_dir" \
        --center_x "$cx" --center_y "$cy" --center_z "$cz" \
        --size_x 25.0 --size_y 25.0 --size_z 25.0 \
        --thread 8000 \
        --opencl_binary_path "."
        
    # --- 4. Extract Affinity Scores ---
    cd "$PROJECT_ROOT" || exit
    
    for out_ligand in "$out_dir"/*_out.pdbqt; do
        # Safety check: skip if no output files were generated
        [ -e "$out_ligand" ] || continue
        
        ligand_name=$(basename "$out_ligand" _out.pdbqt)
        affinity=$(grep "REMARK VINA RESULT" "$out_ligand" | head -n 1 | awk '{print $4}')
        
        echo "$rec_name,$ligand_name,$affinity" >> "$SUMMARY_CSV"
    done
done

echo ""
echo "Full docking pipeline complete!"
echo "Check $SUMMARY_CSV for your aggregated affinity scores."
