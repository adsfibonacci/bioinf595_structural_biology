
# Machine Learning Docking

The goal of this lab is to use machine learning docking to dock
known active and inactive compounds into a target. Here we will use
NeuralPLexer model,


    Qiao Z, Nie W, Vahdat A, Miller III TF, Anandkumar A.
    State-specific protein-ligand complex structure prediction with a multi-scale deep
    generative model. Nature Machine Intelligence, 2024. https://doi.org/10.1038/s42256-024-00792-z.
    https://github.com/zrqiao/NeuralPLexer


We will revisit the HIV-1 Protease as the target and dock approved drugs
and a selection of inactive ligands curated from ChEMBL. NeuralPLexer
does "co-folding" in that it predicts the protein structure along with
the docked ligand conformation.

To score bound ligand conformations, we will use SMINA, a version of AutdockVina for virtual screening

    Lessons Learned in Empirical Scoring with smina from the CSAR 2011 Benchmarking Exercise
    David Ryan Koes, Matthew P. Baumgartner, Carlos J. Camacho
    J. Chem. Inf. Model. 2013, 53, 8, 1893–1904

## Lab steps

1) Install NeuralPLexer and dependencies
      This is a little tricky so follow the instructions carefully!
      src/01_install_packages.sh

2) Download and install targets and ligands and the models
      02.1_download_target_and_ligands.sh

   Using the instructions from Lab 1 download approved ligands
   and inactive ligands from ChEMBL for target CHEMBL243.
   Pre-downloaded versions are here, which you can use
      data/1HXW/CHEMBL243_approved.tsv
      data/1HXW/CHEMBL243_inactives.tsv

3) Test that NeuralPLexer and smina are working on your system
   03.1_test_neuralplex_docking.sh
   03.2_test_scoring.sh

4) Dock the approved and inactive compounds into HIV-1 protease
   04.1_dock_with_neuralplexer.R
     Consider docking
       * with 1HXW where only the sequence is used from the pdb
         (exclude the --use-template --input-template <path>) flags. 
       * with 1HXW that is used as a template
       * with 1HHP that is used as a template
     => so 6 different docking runs:
           (approved, inactive) * (template_1HXW, template_1HHP, and no_template)

5) Score docked poses using the SMINA molecular mechanics focefield
     04.2_score_predictions.R

6) Analyze the resulting scores to see if any of template/no-template
     NeuralPLexer docking runs are able to discriminate the actives
     from inactives. Plot the results and use a T-test to test
     for significance.

Note: that some of the results have been computed as an example
but the rest will need to be computed.


## Writeup

   Summarize the docking that you did and test if the NeuralPLexer
   docking poses combined with SMINA docking scores are able to
   explain the experimentally characterized activities.
