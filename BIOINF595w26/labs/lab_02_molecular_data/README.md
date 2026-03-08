
# Chemical and Molecular Data

This Lab aims at understanding the representation and manipulation of chemical and molecular data

Learning objectives
 * Loading and analyzing 3D molecular structure data in PyMOL
 * Parsing 3D molecular data using python and the biotite library
 
Reference material:

Li, X., Zhang, Q., Guo, P. et al.
Molecular basis for ligand activation of the human KCNQ2 channel.
Cell Res 31, 52–61 (2021). https://doi.org/10.1038/s41422-020-00410-8

Zhao, et al., Cell Reports, 2026, 10.1016/j.celrep.2025.116771
Structure basis for the activation of KCNQ2 by endogenous and exogenous ligands

## Explore KCNQ2 structures in visually

### 1 Load KCNQ2 structures from the PDB into PyMOL
1) Go to the protein databank (PDB) at https://www.rcsb.org/ and search for KCNQ2
2) Navigate to 7CR0, and 7CR2 and download each structure in `PDBx/mmCIF Format` and store in the `data/` directory
3) Open the two structures in PyMOL
4) In the right panel next to 7CR2:
     * 'A'ctions => Align => All to this (*/CA) 

### 2 Measure the gating residues in the apo structure (7CR0)
1) Toggle sequence view in PyMOL by clicking the `SEQ` button in the right hand panel.
2) Show the gating residues as sticks: `show sticks, (resi 314) or (resi 318)`
3) select the S314 residues in each chain, for the 'A'ction for `(sele)`,
   Find => Polar Contacts => Within Selection
4) From the application toolbar at the very top of the screen,
   select Wizard => Measurement. Then measure the distance between the terminal
   oxygen in the S314 that form H-bonds. Note the atom name.
4) Label key residues
   * From the application toolbar => Wizard => Label
   * From the wizard menu => Mode => {onelettercode}{resi}
   * Label key residues in the site
5) Save a good view as a scene
   * From the application menu at the top of the screen, Scenes => Save new Scene


### 3 Measure the geometry of retigabine binding in 7CR2
1) Select one of the retigabine either by finding of the four copies in the
   structure, or in the sequence view at the top
2) While RTG is selected, on the right panel, next to `(sele)`
     * 'A'ctions => Orient, and rotate structure to view the ring of RTG is downards
	 * Adjust mist by moving two fingers on the trackpad up or down together to focus attention on foreground
	 * 'C'olor => by element => pick a warm color
	 * 'A'ctions => hydrogens => remove nonpolar
	 * 'A'ctions => Find => Polar Contacts => To other atoms in object
	 * 'A'ctions => Modify => Expand => 6A by residue
     * 'S'how => sticks (note this will also show sticks in 7CR0 if is is aligned with it
3) Look at residue W236 how does it rotate from the apo structure to the RTG bound structure
4) Label key residues
   * From the application toolbar => Wizard => Label
   * From the wizard menu => Mode => {onelettercode}{resi}
   * Label key residues in the site
4) Save a good view as a scene
   * From the application menu at the top of the screen, Scenes => Save new Scene

### 4 Render views
1) Set up options to make high-quality renderings by pasting this in the box next to "PyMOL>" at the bottom 

```
bg white
set light_count,8
set spec_count,5
set shininess, 10
set specular, 0.25
set ambient,0
set direct,0
set reflect,1.5
set ray_shadow_decay_factor, 0.1
set ray_shadow_decay_range, 2
unset depth_cue
```

2) For each scene i:
   * Click the scene on the right window to orient the scene well
   * In the "PyMOL>" bar, run `ray 2000, 2000; png scene_<i>.png` where i is e.g. 1, 2, etc.
   * Look at the result and either adjust the scene or the render parameters re-render as needed to make it look good.
   
## Explore KCNQ2 structures programatically

### 1 Install biotite
From the command line, activate the course conda environment and install biotite
(https://www.biotite-python.org/latest/) using conda:

    conda activate bioinf595w26
	conda install -c conda-forge biotite
   
### 2 Load cif structures in python
1) Use the `biotite.database.rcsb` module to fetch `7cr2` and `7cr0` in "bcif" format
2) Use the `biotite.structure.io.pdbx` module to
   * read the `BinaryCIFFile`
   * get the structure for `model=1` and `include_bonds=True`

#### 3 Use Biotite to measure the distances of the gate residues
1) Use numpy selection syntax to select the terminal oxygens for the
   serine gate residues (S314)
2) use the `biotite.structure` module to measure the distances between adjacent
   terminal oxygens that form the gate
3) use numpy to compute the mean distance of adjacent terminal oxygens
4) Use the rcsb.org search to find all CryoEM structures for KCNQ2
   (should be 20 of them as of 1/26) and down load it as a .csv file
5) Use pandas to read in this table and iterate over each structure to
   compute the mean distance for each structure and write it to a .tsv file 
 

## What to turn in 
Please submit:
  * rendered pictures of the S314 gate and retagabine binding site suitable for a manuscript figure
     - They should be well composed so the key features are visible
	 - The key features should be visible from the background by using mist, hiding detail, use of color etc.
	 - Label the relevant residues and show hydrogen bonds
  * table of average distances for the S314 across all publically available CryoEM structures of KCNQ2
  * short answers to the following questions:

1) What are the range of average distances you see for the S314 gate? How does this relate to the function of protein?
2) Is retigabine (RTG) an agonist or an antagonist?
3) What is a structural re-arrangement that has to happen in the retigabine binding site to allow it to bind?






