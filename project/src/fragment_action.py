import os
import gc
import sys
import torch
import random
import pickle
import pathlib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from datetime import datetime
from omegaconf import OmegaConf

# Mute RDKit warnings
RDLogger.DisableLog('rdApp.*')

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.mol import Molecule
from synformer.models.synformer import Synformer

def sanitize_smiles(smi):
    """Handles disconnected fragments and ensures RDKit can parse the molecule."""
    if not smi or not isinstance(smi, str):
        return None
    try:
        # If multiple fragments exist (separated by .), keep only the largest one
        if "." in smi:
            fragments = smi.split(".")
            smi = max(fragments, key=len)
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        # Return canonical SMILES
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return None

class SynformerFastEngine:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        ckpt = torch.load(model_path, map_location="cpu")
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        
        synformer_root = pathlib.Path("dependencies/synformer")
        fp_path = synformer_root / config.chem.fpindex
        rxn_path = synformer_root / config.chem.rxn_matrix
        
        self.fpindex = pickle.load(open(fp_path, "rb"))
        self.rxn_matrix = pickle.load(open(rxn_path, "rb"))

        self.model = Synformer(config.model).to(self.device)
        state_dict = ckpt["state_dict"]
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def featurize(self, smiles, repeat=15):
        mol = Molecule(smiles)
        atoms, bonds = mol.featurize_simple()
        atoms = atoms[None].repeat(repeat, 1).to(self.device)
        bonds = bonds[None].repeat(repeat, 1, 1).to(self.device)
        atom_padding_mask = torch.zeros([repeat, atoms.size(1)], dtype=torch.bool, device=self.device)
        smiles_t = mol.tokenize_csmiles()[None].repeat(repeat, 1).to(self.device)
        
        return {
            "atoms": atoms,
            "bonds": bonds,
            "atom_padding_mask": atom_padding_mask,
            "smiles": smiles_t,
        }

    def predict_batch(self, smiles_list):
        results_map = {smi: [] for smi in smiles_list}
        for smi in smiles_list:
            try:
                feat = self.featurize(smi, repeat=10)
                with torch.inference_mode():
                    result = self.model.generate_without_stack(
                        feat,
                        rxn_matrix=self.rxn_matrix,
                        fpindex=self.fpindex,
                        temperature_token=1.0,
                        temperature_reactant=0.1,
                        temperature_reaction=1.0,
                    )
                
                stacks = result.build()
                for stack in stacks:
                    if stack.get_stack_depth() == 1:
                        analog = stack.get_one_top()
                        results_map[smi].append(analog.smiles)
                
                # Cleanup to save VRAM
                del feat, result, stacks
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"[Engine Error] {smi[:15]}: {e}", file=sys.stderr)
                
        return results_map

def mutate_molecule(smiles):
    """Performs small RDKit mutations."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return smiles, "STALL"
    
    try:
        rw_mol = Chem.RWMol(mol)
        num_atoms = rw_mol.GetNumAtoms()
        action = random.choice(["DELETE", "CHANGE", "ADD"])
        target_idx = random.randint(0, num_atoms - 1)
        
        if action == "DELETE" and num_atoms > 10:
            atom = rw_mol.GetAtomWithIdx(target_idx)
            if atom.GetDegree() == 1:
                rw_mol.RemoveAtom(target_idx)
                return Chem.MolToSmiles(rw_mol), "DELETE"
        elif action == "CHANGE":
            atom = rw_mol.GetAtomWithIdx(target_idx)
            if atom.GetSymbol() == 'C' and not atom.IsInRing():
                new_el = random.choice(['O', 'N'])
                atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_el))
                return Chem.MolToSmiles(rw_mol), f"CHG_{new_el}"
        elif action == "ADD":
            atom = rw_mol.GetAtomWithIdx(target_idx)
            if atom.GetSymbol() in ['C', 'N'] and atom.GetTotalNumHs() > 0:
                new_idx = rw_mol.AddAtom(Chem.Atom(6)) 
                rw_mol.AddBond(target_idx, new_idx, Chem.BondType.SINGLE)
                return Chem.MolToSmiles(rw_mol), "ADD_C"
    except:
        pass
            
    return smiles, "STALL" 

def fold_3d(smiles):
    """Generates 3D conformation for docking."""
    smi = sanitize_smiles(smiles)
    if not smi: return None
    
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.maxIterations = 100
        
        if AllChem.EmbedMolecule(mol, params) == -1: 
            if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
                return None
        return mol
    except:
        return None
