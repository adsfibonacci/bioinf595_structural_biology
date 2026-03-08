

import pandas as pd
from src.filter_mol2 import filter_mol2

D4_in_vitro = pd.read_csv("data/D4_in_vitro_data.csv")

filter_mol2(
    "data/top500k_D4_poses.mol2",
    D4_in_vitro['zincid'].tolist(),
    "intermediate_data/D4_in_vitro.mol2")
    
