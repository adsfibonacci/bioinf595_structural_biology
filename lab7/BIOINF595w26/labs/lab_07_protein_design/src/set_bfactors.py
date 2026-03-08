

#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdbx as pdbx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Set B-factors in a CIF file from a TSV score table using biotite."
    )

    parser.add_argument("--input_cif", help="Input mmCIF structure file")
    parser.add_argument("--scores_tsv", help="Input TSV file with residue scores")
    parser.add_argument("--output_cif", help="Output mmCIF file")
    parser.add_argument(
        "--model",
        help="Set the model in multimodel cif files (default=1)",
        type=int,
        default=1)

    parser.add_argument("--chain_id_col", required=True,
                        help="Column name for chain ID")
    parser.add_argument("--res_id_col", required=True,
                        help="Column name for residue number")
    parser.add_argument("--ins_code_col", default=None,
                        help="Column name for insertion code (optional)")

    parser.add_argument("--score_col", required=True,
                        help="Column name for score")

    parser.add_argument("--default", type=float, default=0,
                        help="Default B-factor for residues not in table")

    return parser.parse_args()


def build_score_dict(df, chain_id_col, res_id_col, ins_code_col, score_col):
    score_dict = {}

    for _, row in df.iterrows():
        chain_id = str(row[chain_id_col]).strip()
        res_id = int(row[res_id_col])
        if ins_code_col:
            ins_code = str(row[ins_code_col]).strip()
            if pd.isna(row[ins_code_col]):
                ins_code = ""
        else:
            ins_code = ""

        score = float(row[score_col])


            
        key = (chain_id, res_id, ins_code)
        score_dict[key] = score

    return score_dict


def main():
    args = parse_args()

    # Read TSV
    df = pd.read_csv(args.scores_tsv, sep="\t")

    score_dict = build_score_dict(
        df,
        args.chain_id_col,
        args.res_id_col,
        args.ins_code_col,
        args.score_col
    )

    min_score = df[args.score_col].min()
    max_score = df[args.score_col].max()
    
    # Load structure
    cif_file = pdbx.CIFFile.read(args.input_cif)
    atom_array = pdbx.get_structure(cif_file, model=1)
    
    # Ensure insertion code annotation exists
    if "ins_code" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("ins_code", np.array([""] * atom_array.array_length()))

    if "b_factor" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("b_factor", np.array([args.default] * atom_array.array_length()))

    # Iterate over residues
    residue_starts = struc.get_residue_starts(atom_array)

    for start in residue_starts:
        chain_id = atom_array.chain_id[start]
        res_id = atom_array.res_id[start]
        ins_code = atom_array.ins_code[start]

        key = (chain_id, res_id, ins_code)

        if key in score_dict:
            score = score_dict[key]

            # Apply to all atoms in residue
            mask = (
                (atom_array.chain_id == chain_id) &
                (atom_array.res_id == res_id) &
                (atom_array.ins_code == ins_code)
            )

            atom_array.b_factor[mask] = score

    # Write output CIF
    out_cif = pdbx.CIFFile()
    pdbx.set_structure(out_cif, atom_array)
    out_cif.write(args.output_cif)

    print(f"Wrote modified CIF to {args.output_cif}")
    print(f"To visualize in pymol `spectrum b, blue_white_red, minimum={min_score}, maximum={max_score}`")


if __name__ == "__main__":
    main()
