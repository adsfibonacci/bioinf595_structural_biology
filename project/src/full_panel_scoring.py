import os
import pandas as pd
import pathlib
import subprocess
from tqdm import tqdm
from rdkit import Chem
from mcts import VinaScorer
from fragment_action import fold_3d

def prepare_reference_ligands(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    output_path = pathlib.Path(output_dir).absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    
    prepared = {}
    for _, row in df.iterrows():
        name = str(row['ligand']).lower()
        m3d = fold_3d(row['smiles'])
        if m3d:
            pdbqt = output_path / f"{name}.pdbqt"
            sdf = output_path / f"{name}.sdf"
            writer = Chem.SDWriter(str(sdf))
            writer.write(m3d)
            writer.close()
            subprocess.run(["obabel", str(sdf), "-O", str(pdbqt), "-p", "7.4"], stderr=subprocess.DEVNULL)
            prepared[name] = str(pdbqt)
    return prepared

def main():
    project_root = pathlib.Path(os.getcwd()).absolute()
    scorer = VinaScorer(project_root=project_root)
    vina_bin_dir = pathlib.Path(scorer.vina_dir).absolute()
    
    # 1. Prep
    ref_paths = prepare_reference_ligands("data/reference_ligands.csv", "intermediate/reference_pdbqts")
    
    # 2. Setup Loop
    results = []
    # Convert receptors to list of dicts for easy iteration
    receptors = scorer.receptors_df.to_dict('records')
    
    # Use TQDM progress bar
    pbar = tqdm(total=len(receptors) * len(ref_paths), desc="Docking Reference Panel")
    
    for rec in receptors:
        rec_name = rec['receptor_name']
        rel_path = str(rec['rel_path']).replace('.pdb', '.pdbqt')
        rec_pdbqt = (project_root / "data/vinadock_panel" / rel_path).absolute()

        for lig_name, lig_pdbqt in ref_paths.items():
            out_dir = project_root / f"intermediate/reference_docking/{rec_name}/{lig_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "out.pdbqt"

            cmd = [
                "./QuickVina2-GPU-2-1",
                "--receptor", str(rec_pdbqt),
                "--ligand", str(lig_pdbqt),
                "--out", str(out_file),
                "--center_x", str(rec['center_x']),
                "--center_y", str(rec['center_y']),
                "--center_z", str(rec['center_z']).strip(),
                "--size_x", "25.0", "--size_y", "25.0", "--size_z", "25.0",
                "--thread", "1024",
                "--opencl_binary_path", str(vina_bin_dir)
            ]

            try:
                # Capture stderr to see exactly what is failing if it does
                res = subprocess.run(cmd, cwd=str(vina_bin_dir), capture_output=True, text=True)
                
                if res.returncode == 0 and out_file.exists():
                    with open(out_file, 'r') as f:
                        for line in f:
                            if "REMARK VINA RESULT" in line:
                                affinity = float(line.split()[3])
                                results.append({"mutant": rec_name, "ligand": lig_name, "affinity": affinity})
                                break
                else:
                    # Log the specific error to a file instead of flooding the screen
                    with open("docking_errors.log", "a") as log:
                        log.write(f"Error {rec_name}/{lig_name}: {res.stderr}\n")
            
            except Exception as e:
                pass
            
            pbar.update(1)

    # 3. Save
    pd.DataFrame(results).to_csv("data/reference_lookup_scores.csv", index=False)
    pbar.close()

if __name__ == "__main__":
    main()
