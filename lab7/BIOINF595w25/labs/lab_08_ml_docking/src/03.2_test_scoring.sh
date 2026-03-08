


OUTPUT_PATH=intermediate_data/1HXW_A-84538
~/turbo_bioinf595/opt/bin/smina \
    --score_only -r ${OUTPUT_PATH}/prot_0.pdb \
    -l ${OUTPUT_PATH}/lig_0.sdf
# look for the line starting with 'Affinity:'
