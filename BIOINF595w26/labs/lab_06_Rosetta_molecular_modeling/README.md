
# Molecular Modeling

The goal of this lab is to get familiar with Rosetta based molecular modeling

In this lab you will explore preparing manipulating structures of the
MC4R GPCR. GPCRs are important regulators of a range of biological
functions, common drug targets.  The MC4R receptor is regulates
appetite, feeding, metabolism, and skin pigmentation. It's endogenous
ligand is the a-MSH peptide.

The specific learning objectives are

  * Setting up and use PyRosetta, a python based front-end to Rosetta
  * Prepare a structure for Rosetta Modeling using constrained FastRelax
  * Generate structural diversity using Backrub MCMC
  * Analyze structure prediction diversity using Pnear and structural features

### Refernces

The documentation for Rosetta is a little scattered, but some resources are,

  * (Rosetta Commons on GitHub)[https://github.com/RosettaCommons/rosetta]
  * (PyRosetta API)[https://graylab.jhu.edu/PyRosetta.documentation/]
  * (PyRosetta Notebooks)[https://rosettacommons.github.io/PyRosetta.notebooks/]
  * (Rosetta Documentation)[https://docs.rosettacommons.org/docs/latest/Home]
  

Alternate States of Proteins Revealed by Detailed Energy Landscape Mapping
Michael D. Tyka, Daniel A. Keedy, Ingemar André, Frank DiMaio, Yifan Song,
David C. Richardson, Jane S. Richardson, David Baker
JMB Volume 405, Issue 2, 14 January 2011, Pages 607-618

  * Robust protocol for finging locally optimal conformations

Backrub-Like Backbone Simulation Recapitulates Natural Protein
Conformational Variability and Improves Mutant Side-Chain Prediction
Colin A. Smith, Tanja Kortemme
JMB Volume 380, Issue 4, 18 July 2008, Pages 742-756

  * Used to generate backbone conformational ensembles using MCMC

A Pareto-Optimal Refinement Method for Protein Design Scaffolds
Lucas Gregorio Nivón, Rocco Moretti, David Baker 
PLOS One, (2013), https://doi.org/10.1371/journal.pone.0059004

  * Explores constraining relaxation of native structures

Accurate de novo design of hyperstable constrained peptides.
Bhardwaj, G., Mulligan, V., Bahl, C. et al. Nature 538, 329–335
(2016). https://doi.org/10.1038/nature19791

  * Where Pnear was defined as a measure of score-vs-rmsd folding funnel
  * Example [python code](https://github.com/RosettaCommons/main/blob/master/tests/benchmark/util/quality_measures.py#L268%22) to implement it
  * A [vignette](https://maomlab.github.io/BayesPharma/articles/apply_sigmoid_model_Pnear.html) exploring it as a quantitative measure
  
  
Aranda-García, D., Stepniewski, T.M., Torrens-Fontanals, M. et al. Large scale investigation of GPCR molecular dynamics data uncovers allosteric sites and lateral gateways. Nat Commun 16, 2020 (2025). https://doi.org/10.1038/s41467-025-57034-y  
  
  * Study of general GPCR dynamics
  
  
## Lab Steps

### 1 Install PyRosetta

Download and install pyrosetta using pip

    pip install pyrosetta-installer
    python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'


### 1 Download an clean the experimental structure
  1) Download the `8QJ2` structure of MC4R from the PDB
  2) Inspect the structure in PyMOL and record key details
  
     * What experimental technique was used to determine it (i.e. X-ray,
	   CryoEM, NMR)
	 * What is the overall resolution in Angstroms of the structure
	 * What are the main components of the structure. What state is the
	   structure in (e.g. active/inactive or open/closed, etc.)
	 * What is the reference for the publication that describes it?  
  
  2) Using PyMOL extract just the GPCR chain as its own `.cif` file.
     The structure has other chains, but for now we'll just model the GPCR.

### 2 Use constrained `FastRelax` to prepare an MCR4 experimental structure
Relaxing a structure is important for releving small clashes that Rosetta
think exist in the structure while not moving it too much from the native,
experimentally determined conformation.

  1) Initialize pyrosetta with `pyrosetta.init()`

  2) Load the native `.cif` as a pose

    pose = pyrosetta.rosetta.core.import_pose.pose_from_file(
        filename = structure_fname,
        read_fold_tree = False,
        type = pyrosetta.rosetta.core.import_pose.FileType.CIF_file)
 
   The `pose` variable in python is a reference to the underlying data
   that keeps the state of the molecular structure, and if you want to
   make a copy of it you can use the `pose_new = pose.clone()`.

  3) Define an energy function

    sfxn = pyrosetta.create_score_function(weights_tag = '...')
	
  For the `weights_tag` argument you can use `'ref2015'` for the
  standard weights function or `'beta'` for the latest [candidate
  energy function](https://github.com/RosettaCommons/rosetta/pull/548)

  4) Define the relax protocol, `FastRelax` is a standard that alternatively
     does gradient based minimization and monte-carlo markov chain
     repacking of the side chains.
	 
    fast_relax = pyrosetta.rosetta.protocols.relax.FastRelax(
       scorefxn_in = sfxn,
       standard_repeats = 1)
  
  To generally constrain the relaxed pose to the native pose, you can use

    fast_relax.constrain_relax_to_start_coords()
    fast_relax.ramp_down_constraints(False)

  You can apply the protocol with `fast_relax.apply(pose)`. Depending on
  the size of the structure relaxing can take 30 seconds to 10 minutes.
	 
  5) In PyRosetta, create a protocol that does
  
    score the initial native pose
	for i in range(nsamples):
	  clone the native pose
	  apply FastRelax to relax it
	  score the relaxed pose
	  record metadata
	    1) the difference in scores
		2) the heavy-atom RMSD between native and relaxed poses
		3) time it took to do the relax
	  save the pose to disk
    save the metadata to a .tsv file
		 
  Note: To compute the heavy atom RMSD, you can use this function  		 
    
     rmsd = pyrosetta.rosetta.core.scoring.all_atom_rmsd(pose1, pose2)
	 
  and to save a pose to disk, you can use `pose.dump_pdb(...)` where you
  pass the output path.
  
  6) Inspect the relaxed poses in PyMOL, by aligning them all with
     the `8QJ2`. Overall how well do they align, are there any major
	 differences, and if so what may be causing this?

### 3 Generate a diverse ensemble using the `BackRubProtocol`
BackRub is an inverse kinematics approach to make locally move portions of the
backbone without having lever arm effects that propogate to the rest of the
structure. To make BackRub a valid MCMC move, requires what is called detailed
balance, in that the probability of taking a step from X1 to X2 is the same as
taking a step from X2 to X1.

  1) Create the backrub protocol that can be applied to a pose 
  
    backrub_protocol = pyrosetta.rosetta.protocols.backrub.BackrubProtocol()
	...
	pose = ...
	backrub_protocol.apply(pose)
	...
	 
  2) Specify MCMC parameters. A minor inconvienence with
     BackRubProtocol is that it was originally designed using the
     Rosetta C++ API as a standalone application and not well adapted
     to PyRosetta. So to specify the key MCMC parameters of the
     temperature defined by the acceptance rate and the number of
     steps to take, it expects them to be specified on the commandline
     which can be done in PyRosetta when calling the `init` function
     before creating the `backrub_protocol` instance.
  
    pyrosetta.init(
	  extra_options = f"-backrub:ntrials={ntrials} -backrub:mc_kt={mc_kt}")

  WARNING: PyRosetta cannot be re-initialized more than once per session. So,
  I recommend creating a python script called `src/sample_backrub.py` that
  you can call from the command line like this
  
    python src/sample_backrub.py \
	  --native_pdb "$native_pdb" \
	  --input_pdb "$relaxed_pdb" \
	  --output_pdb "$output_pdb" \
	  --ntrials "$ntrials" \
	  --mc_kt "$mc_kt" \
	  --nsamples "$nsamples"
     
  and it will take in a relaxed pdb, generate 10 samples where where
  each sample applies the `BackRubProtocol` and reports similar
  metrics as for the relax stage above. Note, record the RMSD to the
  input native conformation **and** the input conformation. Having it as
  a standalone python program allows the init function be able to be
  called one for each time the MCMC parameters need to be specified.
  
  3) Generate samples for varying levels of `mc_kt`, `ntrials`, and
     `nsamples`. For a production run typical parameters would be
     `mc_kt=0.7`, `ntrials=10000`, and `nsamples=10000`. But this is quite
	 computationally expensive, so for this lab consider testing in the range
	 
	input = [one of the relaxed structures]
	mc_kt = [0.4-1.5]
    ntrials = [1000-10000]
	nsamples = 10
  
  Start with small numbers for `ntrials` and `nsamples` to make sure the code
  is working, and if you have time, scale up to run on greatlakes to have them
  run in parallel. 
  
  4) Inspect some of the backrub samples in PyMOL to confirm the protocol is
     actually running.
  

### 4 Analyze the Score-vs-RMSD plot of the BackRub sampling protocol

  1) Generate a scatter plot (e.g. using `ggplot2` in R) of the different
     relaxed and BackRub generated samples. Describe the units of the plot axes.
	 Use the plot to answer the questions
	 
	 * Do you see more diversity during the relax stage or the backrub stage?
	 * Do higher backrub `ntrials` or `mc_kt` values produce more diversity?
	 * Do you see folding funnel to the native conformation? in that Rosetta
	   scores conformationst that have close to 0 rmsd the lowest?
	   
   2) Quantify the the overall folding funnel across different backrub
      sampling strategies using Pnear metric. You can use the
      [Pnear](https://github.com/RosettaCommons/main/blob/master/tests/benchmark/util/quality_measures.py#L268%22)
      python function. Or from the [BayesPharma](https://maomlab.github.io/BayesPharma/reference/Pnear.html).
  
### 5 Analyze the functional states of the Rosetta modeled conformations
From figure 2 of (Aranda-García, et al., 2025), they define the open, intermediate,
and closed states based on the distance between the C-alpha atoms of positions
2x46 and 6x37. These generic GPCR positions can be mapped to MC4R using
the residue tables from the [gpcrdb.org](https://gpcrdb.org/residue/residuetable).

   1) Adapt the biotite measurement scripts from lab 2 to quantify the
      open distance geometry of each of the sampled
	  conformations.
	  
   2) Generate a scatter plot of open distance geometry vs. RMSD. 
   
   
## What to turn in
Answer the questions in each of the steps above, in a brief pdf write up.
Include the scripts as separate files or well formatted into the writeup.

