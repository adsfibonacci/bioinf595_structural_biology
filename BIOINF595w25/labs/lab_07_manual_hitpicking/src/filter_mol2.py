import os

def filter_mol2(input_mol2_file_path, hit_identifiers, output_mol2_file_path):
    # Input validation
    if not isinstance(hit_identifiers, (list, set)):
        raise ValueError("hit_identifiers must be a list or set.")
    output_f = open(output_mol2_file_path, 'w')
    line_index=0
    start_line=None
    with open(input_mol2_file_path, 'r') as f:
        found_tripos = False
        mol_lines = []
        name = None
        while not f.tell() == os.fstat(f.fileno()).st_size:
            line = f.readline()
            if line.startswith("##########                 Name:"):
                if name is not None and name in hit_identifiers:
                    print(f"Found hit: {name}")
                    for mol_line in mol_lines:
                        output_f.write(mol_line)
                name = line.split()[2]
                mol_lines = []
            mol_lines.append(line)
        output_f.close()
