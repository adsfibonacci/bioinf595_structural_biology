import os

protein_seq = "YELPEDPRWELPRDRLVLGKPLGEGAFGQVVLAEAIGLDKDKPNRVTKVAVKMLKSDATEKDLSDLISEMEMMKMIGKHKNIINLLGACTQDGPLYVIVEYASKGNLREYLQARRPEQLSSKDLVSCAYQVARGMEYLASKKCIHRDLAARNVLVTEDNVMKIADFGLARDHIDYYKKTTNGRLPVKWMAPEALFDRIYTHQSDVWSFGVLLWEIFTLGGSPYPGVPVEELFKLLKEGHRMDKPSNCTNELYMMMRDCWHAVPSQRPTFKQLVEDLDRIVALTS"

# Full ligand table (deduplicated ZINC72412269)
ligands = {
    "ZINC15726433": "CCNC(=O)[C@H]1CC[N@H+](Cc2nc(c3ccc(OC)c(OC)c3)no2)CC1",
    "ZINC32891679": "C[C@@H](c1cccc(NC(=O)c2ccccc2)c1)N(C)C(=O)CCn1c(=O)oc2ccccc21",
    "ZINC44308404": "CCCNC(=O)C[N@H+]1CC[C@H](NC(=O)Nc2ccc3oc(C4CC4)nc3c2)CC1",
    "ZINC90883402": "C[C@H]([NH2+]Cc1ccc(-c2ccc3c(c2)C[C@@H](C)O3)cc1)c1cnn(C)c1",
    "ZINC33310196": "CCCNC(=O)Nc1ccc(NC(=O)N2CCC(C(=O)NCC3CC3)CC2)cc1",
    "ZINC23338973": "COc1cccc(-c2cccc(N3CCC([NH2+]CC[C@H]4CC(=O)N(C)C4)CC3)c2)c1",
    "ZINC08428037": "COc1ccc(NC(=O)c2ccc(C[N@H+](C)Cc3ccccc3F)cc2)cc1",
    "ZINC12447200": "CCn1ccnc1CNc1ccc(-c2nc(-c3ccccc3F)no2)c[nH+]1",
    "ZINC64719881": "O=C(CCCn1c(=O)oc2ccccc21)NCCC1=c2ccccc2=[NH+]C1",
    "ZINC48355563": "O=C([O-])c1ccn(-c2ccc(NS(=O)(=O)c3ccc(N4CCCC4=O)cc3)cc2)n1",
    "ZINC73306518": "C[NH+](C)CC(=O)Nc1ccc(NC(=O)c2ccc(-n3ccnc3)nc2)cc1",
    "ZINC89773358": "C[NH+](C)CC(=O)Nc1ccc(NC(=O)c2ccc(-n3ccnn3)cc2)cc1",
    "ZINC72412269": "COc1ccc(OC[C@H]2CC[N@H+](C[C@]3(O)CCC[NH2+]C3)CC2)cc1",
    "ZINC01351668": "O=C(Nc1ccc(C[NH+]2CCCCC2)cc1)c1cc2ccccc2oc1=O",
    "ZINC29323279": "O=C(NCc1ccc(N2CCCC2)[nH+]c1)NCc1ccc2c(c1)OCCO2"
}

os.makedirs("boltz_yaml_fixed", exist_ok=True)

# EXACT template you confirmed works
template = """sequences:
- protein:
    id: A
    sequence: {protein}
- ligand:
    id: B
    smiles: '{smiles}'
properties:
- affinity:
    binder: B
"""

for lig, smi in ligands.items():
    yaml_text = template.format(
        protein=protein_seq,
        smiles=smi
    )

    out_file = f"boltz_yaml_fixed/boltz_{lig}.yaml"
    with open(out_file, "w") as f:
        f.write(yaml_text)

    print(f"Wrote {out_file}")

print("\nDone: all YAML files generated.")
