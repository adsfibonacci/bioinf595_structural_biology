#!/bin/bash
ulimit -s unlimited
# --- 1. Configuration & Absolute Paths ---
PROJECT_ROOT=$(pwd)
VINA_DIR="$HOME/.aur/Vina-GPU-2.1/QuickVina2-GPU-2.1"
EXECUTABLE="./QuickVina2-GPU-2-1"

LIGAND_DIR="$PROJECT_ROOT/intermediate/vinadock_ligands"
TEST_LIGAND_DIR="$PROJECT_ROOT/intermediate/vinadock_ligands_test"
RECEPTOR_DIR="$PROJECT_ROOT/data/vinadock_panel"
RESULTS_DIR="$PROJECT_ROOT/results/vina_gpu_docking_test"
CENTERS_CSV="$RECEPTOR_DIR/receptor_centers.csv"
SUMMARY_CSV="$RESULTS_DIR/test_affinity_matrix.csv"

# --- 2. Setup Test Environment ---
mkdir -p "$RESULTS_DIR" "$TEST_LIGAND_DIR"

# Clean out the test directory in case of previous runs
rm -f "$TEST_LIGAND_DIR"/*.pdbqt

# Copy exactly the first 2 ligands into the test directory
echo "Setting up 2 ligands for testing..."
ls -1 "$LIGAND_DIR"/*.pdbqt | head -n 2 | xargs -I {} cp {} "$TEST_LIGAND_DIR/"

echo "receptor_mutant,ligand_name,affinity_kcal_mol" > "$SUMMARY_CSV"

# --- 3. Iterate Over First 2 Receptors ---
# We use head -n 3 to get the header + first 2 receptors, then tail -n +2 to drop the header
head -n 3 "$CENTERS_CSV" | tail -n +2 | while IFS=',' read -r rec_name rel_path cx cy cz; do
    
    # Clean up line endings
    cz=$(echo "$cz" | tr -d '\r')
    
    rel_pdbqt="${rel_path%.pdb}.pdbqt"
    rec_file="$RECEPTOR_DIR/$rel_pdbqt"
    out_dir="$RESULTS_DIR/$rec_name"
    
    mkdir -p "$out_dir"
    
    echo "======================================================"
    echo "TEST DOCKING: $rec_name"
    echo "Center: X=$cx, Y=$cy, Z=$cz"
    echo "======================================================"
    
    # Move into the Vina-GPU directory
    cd "$VINA_DIR"
    
    # Run QuickVina-W-GPU pointed at the TEST_LIGAND_DIR
    $EXECUTABLE \
        --receptor "$rec_file" \
        --ligand_directory "$TEST_LIGAND_DIR" \
        --output_directory "$out_dir" \
        --center_x "$cx" --center_y "$cy" --center_z "$cz" \
        --size_x 25.0 --size_y 25.0 --size_z 25.0 \
        --thread 1000 \
        --opencl_binary_path "."
        
    # --- 4. Extract Affinity Scores ---
    cd "$PROJECT_ROOT"
    
    for out_ligand in "$out_dir"/*_out.pdbqt; do
        [ -e "$out_ligand" ] || continue
        
        ligand_name=$(basename "$out_ligand" _out.pdbqt)
        affinity=$(grep "REMARK VINA RESULT" "$out_ligand" | head -n 1 | awk '{print $4}')
        
        echo "$rec_name,$ligand_name,$affinity" >> "$SUMMARY_CSV"
    done
done

echo ""
echo "Test complete!"
echo "Check $SUMMARY_CSV to ensure affinities were extracted properly."
