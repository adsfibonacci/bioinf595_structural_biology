import os
import random
import csv

# ====================================================================
# [!] UPDATE THESE IMPORTS BASED ON THE ACTUAL SYNGA REPO STRUCTURE
# You will need to look inside `src/` to find what they named their 
# Library/Chemspace loader, their Tree object, and their Mutator.
# ====================================================================
from src.library import ChemSpace        # The class that loads building blocks & reactions
from src.tree import SynthesisTree       # The object representing a molecule's synthesis route
from src.mutations import mutate_tree    # The function that applies Grow/Shrink/Change
# ====================================================================

def synga_random_walk(chemspace_path, starting_tree=None, num_steps=10):
    """
    Performs a continuous random walk by mutating a Synthesis Tree step-by-step.
    """
    seed = random.randint(1000, 9999)
    random.seed(seed)
    
    output_dir = os.path.join("testing", f"synga_walk_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "synga_trajectory.csv")
    csv_data = [["Step", "Action", "SMILES"]]
    
    print(f"\n[START] SynGA Random Walk | Seed: {seed}")
    print("-" * 60)
    
    # 1. Load the building blocks and reactions
    print("Loading ChemSpace library...")
    library = ChemSpace(chemspace_path) 
    
    # 2. Initialize the starting state
    if starting_tree is None:
        # If no tree is provided, pick a random building block to start
        current_tree = SynthesisTree.random_initialization(library)
    else:
        current_tree = starting_tree
        
    current_smiles = current_tree.get_smiles() # Adjust method name as needed
    
    print("Step 0: Scoring Base Tree...")
    print(f"  -> Base Molecule: {current_smiles}")
    csv_data.append(["0", "Starting Tree", current_smiles])
    
    # 3. Perform the Walk
    for step in range(1, num_steps + 1):
        print(f"\nStep {step}: Mutating Tree...")
        
        # Apply a valid chemical reaction/mutation to the tree
        new_tree = mutate_tree(current_tree, library)
        
        # If the mutation fails (e.g., incompatible building blocks), we reject it
        if new_tree is None or new_tree.get_smiles() == current_smiles:
            print("  -> [REJECTED]: Mutation produced invalid chemistry. Retrying next step.")
            continue
            
        action_detail = new_tree.last_mutation_type # e.g., "Grew via Amide Coupling"
        current_smiles = new_tree.get_smiles()
        
        print(f"  -> {action_detail}")
        print(f"  -> Current Molecule: {current_smiles}")
        
        csv_data.append([str(step), action_detail, current_smiles])
        
        # Advance the walk to the newly synthesized molecule
        current_tree = new_tree
        
    # Save results
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
        
    print("-" * 60)
    print(f"Walk complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    # Point this to the library you initialized via data/libs/setup
    lib_path = "data/libs/chemspace"
    
    if os.path.exists(lib_path):
        synga_random_walk(chemspace_path=lib_path, num_steps=10)
    else:
        print("[!] Please run the data setup script to build the ChemSpace first.")
