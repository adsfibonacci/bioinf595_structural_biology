#!/bin/bash

find data/pdb_files -type f -name "*.pdb" | while read -r file; do
    # Filter out common water residue names (HOH, WAT, H2O, DOD) to a temp file
    grep -E -v "^HETATM.*(HOH|WAT|H2O|DOD)" "$file" > "${file}.tmp"
    
    # Overwrite the original file with the cleaned temp file
    mv "${file}.tmp" "$file"
    
    echo "Removed waters from: $file"
done
