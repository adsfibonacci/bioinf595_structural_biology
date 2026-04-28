import sys
import json
import argparse
import random
import contextlib
from fragment_action import SynformerFastEngine, mutate_molecule, sanitize_smiles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--l", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    # Redirect ALL initialization noise to stderr
    with contextlib.redirect_stdout(sys.stderr):
        try:
            engine = SynformerFastEngine(args.model_path)
        except Exception as e:
            print(f"[Worker Error] Engine init failed: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Ensure start smiles is clean
    start_smi = sanitize_smiles(args.smiles)
    if not start_smi:
        sys.stdout.write("[]\n")
        return

    active_walks = {i: start_smi for i in range(args.k)}
    walk_history = {i: [] for i in range(args.k)}

    for _ in range(1, args.l + 1):
        if not active_walks: 
            break
        
        # Mutation step
        m_map = {}
        for k, smi in active_walks.items():
            mut_smi, _ = mutate_molecule(smi)
            clean_mut = sanitize_smiles(mut_smi)
            if clean_mut:
                m_map[k] = clean_mut

        if not m_map:
            break

        unique_muts = list(set(m_map.values()))
        
        # Batch generation from Synformer
        with contextlib.redirect_stdout(sys.stderr):
            projections = engine.predict_batch(unique_muts)
        
        new_active = {}
        for k, msmi in m_map.items():
            possible_projs = projections.get(msmi, [])
            # Filter and sanitize projections
            valid_projs = [sanitize_smiles(p) for p in possible_projs if sanitize_smiles(p)]
            
            if valid_projs:
                chosen_smi = random.choice(valid_projs)
                new_active[k] = chosen_smi
                walk_history[k].append(chosen_smi)
                
        active_walks = new_active

    # Format the array: Only return walks that actually generated steps
    jagged_array = [walk_history[k] for k in range(args.k) if walk_history[k]]
    
    # Final clean JSON output
    sys.stdout.write(json.dumps(jagged_array) + '\n')
    sys.stdout.flush()

if __name__ == "__main__":
    main()
