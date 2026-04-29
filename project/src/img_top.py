import os
from rdkit import Chem
from rdkit.Chem import Draw

# Create the target directory
output_dir = "results/images"
os.makedirs(output_dir, exist_ok=True)

# Your provided SMILES list
smiles_input = """
C=C1CC[C@H](C(C)=C(CC(C)CC)c2cc(S(=O)(=O)N(C)c3cc(C(F)(F)F)ccc3N)ccc2F)CC1OC1(C(=O)Nc2ccc(F)c(S(=O)(=O)NC)c2)CC(OCC)=C1C
Cc1c(F)c(F)c(NS(=O)(=O)c2ccc(C3(O)CCC(=O)C(=Cc4cc(Cl)cc(Cl)c4)C3)c(C)c2Cl)c(F)c1F
Cc1c(C(F)(F)F)cc(C(F)(F)F)c(C)c1NS(=O)(=O)c1cccc(C=C2C(=O)CC3CC(C)CC23)c1
Cc1cc(Cl)c(C2(O)CCC(=O)C(=Cc3cccc(O)c3)C2)cc1S(=O)(=O)Nc1c(O)c(Cl)cc(C)c1F
NC1=CC(=O)C(=Cc2ccc(F)c(S(=O)(=O)Nc3cc(F)c(F)cc3NC3=CC(=O)CCC3)c2)CC1
Cc1c(C2(O)CCC(=O)C(=Cc3cccc(Cl)c3)C2)ccc(S(=O)(=O)Nc2c(F)c(F)c(C)c(F)c2F)c1C
CCOC1=C(C)C(O[C@H]2C[C@@H](C(C)=Cc3cc(S(=O)(=O)Nc4cc(C(F)(F)F)ccc4F)ccc3F)CC(C(C)CC)=C2C)(C(=O)Nc2cccc(S(=O)(=O)Nc3ccccc3F)c2)C1
Cc1ccc(C=C2CC3(C)CC(=O)CC(C)(C2)C3)cc1S(=O)(=O)Nc1c(C)c(C(F)(F)F)cc(C(F)(F)F)c1C
Cc1cc(C2(O)CCC(=O)C(=Cc3cccc(C=O)c3)C2)ccc1S(=O)(=O)Nc1c(F)c(F)c(C)c(F)c1F
Cc1cccc(N(c2cc(C)c(F)cc2F)c2cccc(NC3=CC(=O)C(=Cc4ccc(F)c(F)c4)CC3)c2C)c1
Cc1ccccc1C1(Oc2ccc(C(F)(F)F)cc2Nc2cccc(C3(O)CCCCCCCC3)c2O)CCCCC1
O=S(=O)(F)c1cccc(S(=O)(=O)NC2CCC3(CC2)CCC3Nc2c(F)cccc2C(F)(F)F)c1
Cc1cc(Cl)c(C2CCC(=O)C(=Cc3cccc(O)c3)C2)cc1S(=O)(=O)Nc1c(O)c(F)cc(C)c1F
Cc1cc(Cl)c(C2(O)CC(=Cc3cccc(O)c3)C(=O)C2C)cc1S(=O)(=O)Nc1c(O)c(F)cc(C)c1F
O=C1CCC(=Cc2cccc(S(=O)(=O)Nc3cc(F)c(O)c(F)c3F)c2)CC1=Cc1cccc(O)c1
Cc1cc(N)c(NS(=O)(=O)c2ccc(C=C3C(F)=C(F)C(=O)C(F)=C3F)cc2C)cc1C
Cc1ccc(C2(O)CCC3=C(C2)C(=Cc2cccc(Cl)c2)C(=O)CC3)cc1S(=O)(=O)Nc1c(O)c(Cl)cc(C)c1F
Cc1c(F)ccc(NS(=O)(=O)c2ccc(F)c(C=C3C(=O)CC(C)CC3C)c2)c1F
Cc1c(C(F)(F)F)cc(C(F)(F)F)c(C)c1NS(=O)(=O)c1cccc(C(C)C=C2CC3(C)CC(=O)CC(C)(C2)C3)c1
Cc1ccc(C=C2CC(O)(c3ccc(F)c(S(=O)(=O)Nc4cc(F)cc(F)c4F)c3F)C(C)C2=O)cc1C(O)c1cccc(Cl)c1
Cc1cc(O)c(C=C2CC(O)(c3ccc(C)c(S(=O)(=O)Nc4c(F)cc(F)c(O)c4F)c3)CCC2=O)cc1C
Cc1cc(S(=O)(=O)NS(=O)(=O)c2cc(C=C3CC(O)=C(Br)C3=O)ccc2Cl)c(C)c(C)c1C
Cc1ccc(S(=O)(=O)Nc2ccc(C)c(C3(O)CCCCCCCCC3)c2C)c(Cl)c1Cl
Cc1ccc(C2(O)CCC(=O)CC2=Cc2cccc(Br)c2)cc1S(=O)(=O)Nc1c(F)cc(F)cc1F
O=C1C=C(Nc2cc(C=C3CCC(N(C4=CC(=O)CCC4)c4cc(F)c(F)c(F)c4F)=CC3=O)ccc2Cl)CCC1
Cc1ccc(-n2cnc3c(Nc4c(F)cc(NC5C=C(C6(O)C=C(Cl)CC6)CC5)cc4F)ncnc32)cc1C
CC1=CC(O)(c2cc(Cl)cc(S(=O)(=O)Nc3c(F)cc(F)cc3F)c2)CC(C)(C)C1=O
Cc1c(F)cc(NS(=O)(=O)c2cccc(C=C3C(=O)CC(C)CC3C)c2)cc1F
CC1=C(C)C(=O)C(=Cc2cccc(S(=O)(=O)Nc3c(F)c(F)c(C)c(F)c3F)c2)CC1
Nc1ccc(C(F)(F)C(F)(F)F)cc1NS(=O)(=O)c1cccc(C2(O)CCCCCCCCC2)c1F
Nc1cc(C(F)(F)F)ccc1Nc1cc(C(F)(F)F)cc(C2(O)CCCCCCCCC2)c1
Cc1ccc(C2(O)CCC(=O)C(=Cc3cccc(O)c3)C2)cc1S(=O)(=O)Nc1c(F)c(F)c(C)c(F)c1F
Cc1cc(S(=O)(=O)Nc2c(O)c(F)cc(C)c2F)c(F)cc1C1(O)CCC(=O)C1=Cc1cccc(O)c1
Cc1cc(F)c(O)c(NS(=O)(=O)c2c(C)cc(C)c(C3(O)CCC(=O)C(=Cc4cccc(O)c4)C3)c2C)c1F
CC1=CC(=O)C(C)=C(C)C1=Cc1cccc(S(=O)(=O)Nc2cc(F)c(C)c(Br)c2C)c1
CC1=C(NC2=CC(=O)CC2)C(=Cc2ccc(F)c(S(=O)(=O)Nc3cc(F)c(F)cc3F)c2)C1=O
CC1=CC(=O)C(=Cc2cccc(S(=O)(=O)Nc3cc(F)c(F)c(F)c3)c2)CC1
Cc1ccc(C=C2CCC(O)(c3cc(F)cc(Nc4cc(F)cc(N)c4F)c3C)C(C)C2=O)cc1C(O)c1cccc(O)c1
Nc1ccc(C(F)(F)F)cc1NS(=O)(=O)c1c(F)ccc(C2(O)CCCCCCCCC2)c1F
Cc1c(C)c(S(=O)(=O)Nc2cc(C(F)(F)F)ccc2F)c(C)c(C)c1C1(O)CCCC(C)CC1
Cc1c(C(F)(F)F)cc(C(F)(F)F)c(C)c1NS(=O)(=O)c1cccc(C=C2CCCC(C)(C)C2)c1
CC1=C(C)C(=O)C(=Cc2cccc(S(=O)(=O)Nc3c(F)c(F)c(C)c(F)c3F)c2)C1
Cc1ccc(C2(O)CCC(=O)C(=Cc3cc(O)cc(O)c3)C2)cc1S(=O)(=O)Nc1c(F)c(F)c(F)c(F)c1F
Cc1ccc(C=C2C(=O)C(C)C2(O)c2ccc(F)c(NS(=O)(=O)c3c(C)cc(F)cc3C)c2F)cc1C(O)c1cccc(O)c1
Cc1cc(C)c(C=C2CCC2(O)c2ccc(C)c(S(=O)(=O)Nc3c(F)cc(F)c(O)c3F)c2)cc1C
CCC1=CC(=O)C(=Cc2cccc(S(=O)(=O)Nc3cc(F)c(F)c(F)c3)c2)CC1
Nc1cc(F)c(F)cc1NS(=O)(=O)c1cc(C=C2CCC(NC3=CC(=O)CCC3)=CC2=O)ccc1Cl
Cc1ccc(C=C2C(=O)C(C)C2(O)c2ccc(F)c(NS(=O)(=O)c3c(C)cc(F)cc3C)c2F)cc1C(O)c1cccc(C=O)c1
CCc1ccc(F)c(NS(=O)(=O)c2cc(C3(O)CCC(=O)C(=Cc4cc(O)cc(O)c4)C3)c(Cl)cc2C)c1F
Cc1c(NS(=O)(=O)c2cccc(CC=C3CC4(C)CC(=O)CC(C)(C3)C4)c2)cc(C2CC2)cc1C(F)(F)F
"""

# Process lines and draw
smiles_list = [line.strip() for line in smiles_input.strip().split("\n")]

for i, smi in enumerate(smiles_list):
    # Attempt to sanitize or fix common truncated copy issues
    # specifically brackets around stereocenters
    mol = Chem.MolFromSmiles(smi)
    
    if not mol:
        # Simple fix: if it failed and has '@' without '[', it's definitely broken
        if "@" in smi and "[" not in smi:
             # This is a very crude fix, usually manually fixing the csv is better
             # but for your rank_0 (index 0), the SMILES was definitely truncated.
             print(f"Skipping index {i}: SMILES syntax error (likely truncated brackets).")
             continue
        else:
            print(f"Skipping index {i}: RDKit could not parse the string.")
            continue

    # Create high-quality drawing
    filename = os.path.join(output_dir, f"rank_{i}.png")
    Draw.MolToFile(mol, filename, size=(600, 600), legend=f"Rank {i}")

print(f"[*] Processing complete. Check {output_dir}/ for images.")
