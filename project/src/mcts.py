import os
import sys
import math
import json
import torch
import random
import pickle
import pathlib
import subprocess
import pandas as pd
from datetime import datetime

# Import 3D folding and mutation helpers
from fragment_action import fold_3d, sanitize_smiles

# =========================================================
# 1. SUBPROCESS WORKER (GPU ISOLATION)
# =========================================================

def call_synformer_worker(smiles, K, L, model_path="dependencies/synformer/data/trained_weights/sf_ed_default.ckpt"):
    print(os.getcwd())
    """
    Spawns a clean process to run Synformer. 
    Returns a jagged array (list of lists) of SMILES sequences.
    """
    worker_script = str(pathlib.Path(__file__).parent / "synformer_worker.py")
    command = [
        sys.executable, 
        worker_script,
        "--smiles", smiles, 
        "--k", str(K), 
        "--l", str(L),
        "--model_path", model_path
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"      [!] Synformer worker crashed:\n{result.stderr}", file=sys.stderr)
        return []
        
    try:
        # Robust parsing: find the line containing the JSON array
        lines = result.stdout.strip().split('\n')
        for line in reversed(lines):
            if (line.startswith('[[') and line.endswith(']]')) or line == "[]":
                return json.loads(line)
        return []
    except (json.JSONDecodeError, IndexError):
        print(f"      [!] Failed to parse worker JSON. Raw output length: {len(result.stdout)}", file=sys.stderr)
        return []

# =========================================================
# 2. VINA DOCKING EVALUATION
# =========================================================

class VinaScorer:
    def __init__(self, project_root):
        self.project_root = pathlib.Path(project_root)
        self.vina_dir = pathlib.Path(os.path.expanduser("~/.aur/Vina-GPU-2.1/QuickVina2-GPU-2.1"))
        self.receptor_dir = self.project_root / "data/vinadock_panel"
        self.centers_csv = self.receptor_dir / "receptor_centers.csv"
        
        if not self.centers_csv.exists():
            raise FileNotFoundError(f"Receptor centers not found at {self.centers_csv}")
        self.receptors_df = pd.read_csv(self.centers_csv)
    
    def evaluate_rollout(self, smiles_list, Q_proteins, run_id):
        if not smiles_list:
            return 0.0, {}

        run_dir = self.project_root / f"intermediate/mcts_rollouts/run_{run_id}"
        sdf_dir = run_dir / "ligands_sdf"
        ligand_dir = run_dir / "ligands_pdbqt"    
        
        sdf_dir.mkdir(parents=True, exist_ok=True)
        ligand_dir.mkdir(parents=True, exist_ok=True)
        
        valid_ligands = {}
        from rdkit import Chem

        for i, smi in enumerate(smiles_list):
            clean_smi = sanitize_smiles(smi)
            if not clean_smi: continue
                
            m3d = fold_3d(clean_smi)
            if m3d:
                title = f"lig_{i}"
                m3d.SetProp("_Name", title)
                sdf_path = sdf_dir / f"{title}.sdf"
                writer = Chem.SDWriter(str(sdf_path))
                writer.write(m3d)
                writer.close()
                
                pdbqt_path = ligand_dir / f"{title}.pdbqt"
                cmd = ["obabel", str(sdf_path), "-O", str(pdbqt_path), "-p", "7.4", "--partialcharge", "gasteiger"]
                
                try:
                    subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)
                    valid_ligands[title] = (i, smi, pdbqt_path)
                except:
                    continue
                
        if not valid_ligands:
            return 0.0, {smi: 0.0 for smi in smiles_list}

        selected_receptors = self.receptors_df.sample(n=min(Q_proteins, len(self.receptors_df)))
        ligand_scores = {smi: [] for smi in smiles_list}
        
        for _, rec in selected_receptors.iterrows():
            rec_name = rec['receptor_name']
            rel_path = str(rec['rel_path']).replace('.pdb', '.pdbqt')
            rec_file = self.receptor_dir / rel_path
            out_dir = run_dir / "docking_out" / rec_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                "./QuickVina2-GPU-2-1",
                "--receptor", str(rec_file),
                "--ligand_directory", str(ligand_dir),
                "--output_directory", str(out_dir),
                "--center_x", str(rec['center_x']),
                "--center_y", str(rec['center_y']),
                "--center_z", str(rec['center_z']).strip(),
                "--size_x", "25.0", "--size_y", "25.0", "--size_z", "25.0",
                "--thread", "1024",
                "--opencl_binary_path", "."
            ]
            
            try:
                subprocess.run(cmd, cwd=str(self.vina_dir), check=True, capture_output=True)
                for out_ligand in out_dir.glob("*_out.pdbqt"):
                    original_title = out_ligand.stem.replace("_out", "")
                    if original_title in valid_ligands:
                        smi = valid_ligands[original_title][1]
                        with open(out_ligand, 'r') as f:
                            for line in f:
                                if "REMARK VINA RESULT" in line:
                                    affinity = float(line.split()[3])
                                    ligand_scores[smi].append(abs(affinity)) 
                                    break
            except Exception:
                continue

        smi_to_score = {}
        total_sum = 0.0
        count = 0
        for smi in smiles_list:
            scores = ligand_scores.get(smi, [])
            avg = sum(scores) / len(scores) if scores else 0.0
            smi_to_score[smi] = avg
            total_sum += sum(scores)
            count += len(scores)

        return (total_sum / count if count > 0 else 0.0), smi_to_score

# =========================================================
# 3. MCTS CORE LOGIC
# =========================================================

class MCTSNode:
    def __init__(self, smiles, parent=None):
        self.smiles = smiles
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value_sum = 0.0     
        self.vina_score = None   

    @property
    def q_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    def is_fully_expanded(self):
        return len(self.children) > 0

def calculate_uct(node, c=1.414):
    if node.visits == 0:
        return float('inf')
    return node.q_value + c * math.sqrt(math.log(node.parent.visits) / node.visits)

class MCTS_Pipeline:
    def __init__(self, scorer):
        self.scorer = scorer
        torch.set_grad_enabled(False)

    def select(self, node):
        while node.is_fully_expanded():
            shuffled_children = random.sample(node.children, len(node.children))
            node = max(shuffled_children, key=lambda n: calculate_uct(n))
        return node

    def expand(self, node, K=30, Q_proteins=3):
        walks = call_synformer_worker(node.smiles, K=K, L=1)
        newly_created_nodes = []
        seen_smiles = {c.smiles for c in node.children}
        seen_smiles.add(node.smiles)
        
        for walk in walks:
            if not walk: continue
            p = sanitize_smiles(walk[0])
            if p and p not in seen_smiles:
                child = MCTSNode(smiles=p, parent=node)
                node.children.append(child)
                newly_created_nodes.append(child)
                seen_smiles.add(p)

        if newly_created_nodes:
            smiles_to_dock = [c.smiles for c in newly_created_nodes]
            _, smi_to_score = self.scorer.evaluate_rollout(smiles_to_dock, Q_proteins, run_id="expand")
            for child in newly_created_nodes:
                child.vina_score = smi_to_score.get(child.smiles, 0.0)
        
        return random.choice(node.children) if node.children else node

    def rollout(self, start_node, N_walks, L_steps, Q_proteins, run_id):      
        walks = call_synformer_worker(start_node.smiles, K=N_walks, L=L_steps)
        terminal_nodes = []
        all_rollout_smiles = set()
        
        for walk in walks:
            for smi in walk:
                all_rollout_smiles.add(smi)

        unique_smiles = list(all_rollout_smiles)
        _, smi_to_score = self.scorer.evaluate_rollout(unique_smiles, Q_proteins, run_id)

        for walk in walks:
            curr_node = start_node
            for step_smi in walk:
                existing_child = next((c for c in curr_node.children if c.smiles == step_smi), None)
                if not existing_child:
                    child_node = MCTSNode(smiles=step_smi, parent=curr_node)
                    child_node.vina_score = smi_to_score.get(step_smi, 0.0)
                    curr_node.children.append(child_node)
                    curr_node = child_node
                else:
                    curr_node = existing_child
            terminal_nodes.append(curr_node)

        avg_reward = sum([n.vina_score for n in terminal_nodes if n.vina_score]) / len(terminal_nodes) if terminal_nodes else 0.0
        return terminal_nodes, avg_reward

    def search(self, root_node, iterations, N_walks, L_steps, Q_proteins):
        for i in range(iterations):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Iteration {i+1}/{iterations}")
            leaf = self.select(root_node)
            
            if leaf.visits > 0 or leaf == root_node:
                leaf = self.expand(leaf, K=N_walks, Q_proteins=Q_proteins)
            
            terminal_nodes, reward = self.rollout(leaf, N_walks, L_steps, Q_proteins, run_id=i)
            
            if not terminal_nodes:
                print("Backprop on leaf nodes")
                self.backpropagate(leaf, leaf.vina_score or 0.0)
            else:
                if len(terminal_nodes) > 0:
                    print(f"Backprop on {len(terminal_nodes)} terminal nodes")
                for t_node in terminal_nodes:                    
                    self.backpropagate(t_node, reward)
                    
            save_mcts_tree(root_node, f"intermediate/round_{i+143}.pkl")
        return root_node

    def backpropagate(self, node, reward):
        curr = node
        while curr is not None:
            curr.visits += 1
            curr.value_sum += reward
            curr = curr.parent

# =========================================================
# EXPORT & RUN
# =========================================================

def save_mcts_tree(root_node, filepath="intermediate/mcts_tree.pkl"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(root_node, f)

def load_mcts_tree(filepath="intermediate/mcts_tree.pkl"):
    """Loads a previously saved MCTS tree from a pickle file."""
    try:
        with open(filepath, 'rb') as f:
            print(f"[*] Successfully loaded tree from {filepath}")
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        print(f"[!] No valid checkpoint found at {filepath}. Starting fresh.")
        return None

def export_tree_to_csv(root_node, filename="results/mcts_final_production.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    rows = []
    seen = set()
    def traverse(node, depth=0):
        if node.smiles not in seen:
            rows.append({
                "SMILES": node.smiles,
                "Visits": node.visits,
                "Score": node.vina_score,
                "Q_Value": node.q_value,
                "Depth": depth
            })
            seen.add(node.smiles)
        for c in node.children: traverse(c, depth+1)
    traverse(root_node)
    pd.DataFrame(rows).sort_values("Score", ascending=False).to_csv(filename, index=False)

if __name__ == "__main__":
    # RESTORED FULL SMILES LIST
    smiles_init = [
        "CC(C)CN(CC(C(CC1=CC=CC=C1)NC(=O)OC2CCOC2)O)S(=O)(=O)C3=CC=C(C=C3)N", # Amprenavir
        "CC(C)CN(CC(C(CC1=CC=CC=C1)NC(=O)OC2C3COCC3OC2)O)S(=O)(=O)C4=CC=C(C=C4)N", # Darunavir
        "CC1=C(C(=CC=C1)C)OCC(=O)NC(CC2=CC=CC=C2)C(CC(CC3=CC=CC=C3)NC(=O)C(C(C)C)N4CCCC4=O)O", # Lopinavir
        "CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NC(CC2=CC=CC=C2)C(CC(CC3=CC=CC=C3)NC(=O)OCC4=CN=C(S4)C(C)C)O", # Ritonavir
        "CCCC1=C(C(CC(C1)C2=CC(=CC=C2)S(=O)(=O)NC3=C(C=CC(=C3)C(F)(F)F)F)O)C4=C(O)C(=O)C(=C(C4)CC)C", # Tipranavir
        "CC(C)(C)C(C(=O)NC(CC1=CC=CC=C1)C(CN(CC2=CC=CC=C2)NC(=O)C(C(C)(C)C)NC(=O)OC)O)NC(=O)OC", # Atazanavir
        "CC(C)(C)NC(=O)C1CN(CC(C1)CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)C(=O)NC4C5CCCCC5CC4O)O", # Indinavir
        "CC1=C(C=CC=C1O)C(=O)NC(CS2=CC=CC=C2)C(CN3C4CCCCC4CC3C(=O)NC(C)C)O", # Nelfinavir
        "CC(C)(C)NC(=O)C1CC2CCCCC2CN1CC(C(CC3=CC=CC=C3)NC(=O)C(CC(N)=O)NC(=O)C4=NC5=CC=CC=C5C=C4)O" # Saquinavir
    ]
    
    scorer = VinaScorer(project_root=os.getcwd())
    pipeline = MCTS_Pipeline(scorer)

    root = MCTSNode(smiles="C")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluating initial ligands...")
    _, initial_scores = scorer.evaluate_rollout(smiles_init, Q_proteins=15, run_id="init")
    
    for s in smiles_init:
        child = MCTSNode(smiles=s, parent=root)
        child.vina_score = initial_scores.get(s, 0.0)
        root.children.append(child)

    # Production run parameters
    root = pipeline.search(root, iterations=140, N_walks=30, L_steps=7, Q_proteins=10)
    
    save_mcts_tree(root, "intermediate/mcts_production_tree.pkl")
    export_tree_to_csv(root, "results/mcts_final_production.csv")
    print("MCTS Search Complete. Results saved in 'results/' directory.")
