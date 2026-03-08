
# Manual Hit Picking for D4 Screen

In virtual screening, after molecular modeling and machine learning have narrowed down compounds, it can be useful to go through top hits and manually evaluated their quality. In this lab, we will manually evaluate a subset of those experimentally tested in

    Lyu, J., Wang, S., Balius, T.E. et al. Ultra-large library docking for discovering new chemotypes. Nature 566, 224–229 (2019). https://doi.org/10.1038/s41586-019-0917-9

retrieved from https://lsd.docking.org/targets/D4 following steps from

    Bender, B.J., Gahbauer, S., Luttens, A. et al. A practical guide to large-scale docking. Nat Protoc 16, 4799–4832 (2021). https://doi.org/10.1038/s41596-021-00597-z

### Learning Objectives
* Practical skills for loading and viewing docked compounds in PyMol and Chimera
* Deeper understanding of the forces and patterns in protein/small molecule interactions and those prioritized by large-scale docking
* Test your ability to discriminate actives from inactives compared to ML models and your peers

## Steps

### Download UCSF Chimera (Not ChimeraX)
* https://www.cgl.ucsf.edu/chimera/download.html
* Note: This may take a few minutes to download

### In Pymol
* Look at prepared docking structure (rec.pdb) and reference ligand (xtal-lig.pdb)
* Open all rec.pdb and xtal-lig.pdb
* run `fetch 5WIU` from the pymol prompt
* All -> "S" -> Show Lines to see the lines
* Open intermediate_data/D4_in_vitro_hits.mol2
  * Click through some of the hits to see their variation

## In Chimera
* Open rec.pdb and xtal-lig.pdb
* Center view
  * hold control and double click atom on the xtal-lig -> Set pivot
  * Hold option and click and drag to pan to center
  * Favorites -> Side View
  * Select front and back planes and adjust to be closer
  * Surface Clipping …
    * Uncheck cap surfaces to clip planes
* Prepare showing receptor
  * Select -> Chain A
  * Actions -> Atoms/Bonds -> Wire
  * Actions -> Surface -> Show
  * Actions -> Surface ->transparency -> 70%
  * Actions -> Surface -> Color -> by heteroatom

## Hit picking In Chimera
This is to open a multi-molecule .mol2 file and select which ones
are worth persuing, by partitioning them in to
'viable', 'deleted', 'purged'

* Tools -> Surface/Binding Analysis -> ViewDock
  * Open intermediate_data/D4_in_vitro_hits.mol2
  * Dock 3.5.x single
* Hide non-polar hydrogens
  * From the main window
  * Select -> Chemistry -> IDATM Type -> HC
  * Actions -> Atoms/Bonds -> Hide
* Show Total Energy
  * From ViewDock panel
  * Column -> Show -> Total Energy
* Find HBonds
  * From the ViewDock panel
  * HBonds -> Add Count to entire receptor
  * OK (don't need to click apply)
* Prepare for hit-picking
  * In ViewDock
  * Compounds -> List Deleted (so both Viable and Deleted are checked)
  * Select all compounds
  * (Bottom part of panel) Change Compound State -> Deleted
* Go through each compound
  * If you think the compound will bind
  * (Bottom part of panel) Change Compound State -> Viable
* Finalize Hit-picking
  * Compounds -> uncheck List Viable
  * Select All Compounds
  * (Bottom part of panel) Change Compound State -> Purged
  * File -> Rewrite …
  * Select new Mol2 file that will only have Viable compounds



##  Questions

* In hit-picking what were three of the factors you used pick compounds?
* Using the Binder column in data/D4_in_vitro_data.csv, compare with the compounds you selected
you will have to write code to extract the zincid from the .mol2 file that was generated from the picked hits
  * What is your hit-rate?
