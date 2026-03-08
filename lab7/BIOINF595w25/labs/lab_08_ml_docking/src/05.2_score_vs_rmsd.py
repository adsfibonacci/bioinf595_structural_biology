

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolAlign


def rmsd_to_best(path):
    scores = pd.read_csv(f"{path}/smina_scores.tsv", sep = "\t")    
    scores_best = scores.sort_values('affinity').drop_duplicates(['output_path'])
    rmsd_results = []
    for index_best, row_best in scores_best.iterrows():
        ligand_best = list(Chem.SDMolSupplier(row_best['ligand_fname']))[0]
        for index_ligand, row_ligand in scores[ scores['output_path'] == row_best['output_path'] ].iterrows():
            ligand = list(Chem.SDMolSupplier(row_ligand['ligand_fname']))[0]
            rmsd = rdMolAlign.AlignMol(ligand, ligand_best)
            rmsd_results.append({
                'output_path' : row_best['output_path'],
                'receptor_fname' : row_best['receptor_fname'],
                'pose_id_best' : row_best['pose_id'],
                'ligand_fname_best' : row_best['ligand_fname'],
                'affinity_best' : row_best['affinity'],
                'pose_id' : row_ligand['pose_id'],
                'ligand_fname' : row_ligand['ligand_fname'],
                'affinity' : row_ligand['affinity'],
                'rmsd' : rmsd
            })
    return pd.DataFrame(rmsd_results)

rmsd_results = rmsd_to_best("intermediate_data/approved_template_1HXW")
rmsd_results.to_csv("intermediate_data/approved_template_1HXW/rmsd.tsv", sep = "\t", index = False)

