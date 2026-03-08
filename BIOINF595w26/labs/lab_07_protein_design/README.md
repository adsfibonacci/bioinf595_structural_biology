# Protein Design

The goal  of this lab is to get familiar with more complex Rosetta-based protocols
and protein design.

This lab will model the change in free-energy due to mutations (ddG) in the MC4R
GPCR and compare against high-throughput deep-mutational scanning data collected in

    Conor J Howard, Nathan S Abell, Beatriz A Osuna, Eric M Jones, Leon Y
    Chan, Henry Chan, Dean R Artis, Jonathan B Asfaha, Joshua S Bloom, Aaron R
    Cooper, Andrew Liao, Eden Mahdavi, Nabil Mohammed, Alan L Su, Giselle A
    Uribe, Sriram Kosuri, Diane E Dickel, Nathan B Lubock (2025)
    High-resolution deep mutational scanning of the melanocortin-4
    receptor enables target characterization for drug discovery eLife
    13:RP104725. (https://doi.org/10.7554/eLife.104725.3)
  
The lab will cover the following concepts

  * Analyzing Deep Mutational Scanning data
  * Computing ddG using Rosetta Scripts

## Lab Steps

### 1 Gather MC4R DMS data from Howard2025

Look and data presented in figure 1 from (Howard, et al., 2025) and
briefly summarize what it presents. The [raw
data](https://github.com/octantbio/mc4r-dms/raw/refs/heads/main/paper/mc4r-dms.tsv)
is available from their [github repo for the
study](https://github.com/octantbio/mc4r-dms).
   
Download the data and check that it makes sense by plotting it with ggplot2.

Reproduce the heatmaps in fig 2a:

  * Use `ggplot2::geom_tile()`
  * X axis: `pos`, Y axis: `aa`
  * color: `statistic` use a color scale that is similar to what they use in the paper
  * Use `facet_grid` to create a grid of plots, where the columns are `pathway` and the rows
    are compound dose (e.g. `"aMSH high"`. If you use `dplyr::mutate()` to create a new column
	called `compound_dose`, convert it to a factor to make sure the order of the panels make sense.

Create a [volcan](https://en.wikipedia.org/wiki/Volcano_plot_(statistics)) plot to relate effect size to signifiance
  * use `ggplot2::geom_point()`
  * X axis: `log2FoldChange`, Y axis: `-log10(p.adj)`
  * Same facets as in the previous plot.

	  
Visualize the best ddG of mutation at each position on a MC4R structure in PyMOL

  * Download an MC4R cif file from the pdb e.g. `8QJ2`, that we used last week.
  * Prepare an input file with columns `[chain_id, res_id, score]` and sse the `src/set_befactors.py` to
    set the b-factors the structure.
  * Visualize the b-factors on the structure in PyMOL with `spectrum b, blue_white_red`
  * Render a view of structure using `ray`.
	

### 2 Predict ddG of mutations using pyrosetta
Use pyrosetta to make each mutation and measure and report the change in energy. 

  * Starting from from the lowest energy relaxed structure of MC4R from Lab 6
  * For each residue/amin acid combination that is not a nonsense mutation (i.e. '*')
  * Follow Step 2 in the [Point Mutant Scan](https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb) pyrosetta notebook tutorial to compute score difference before and after packing.
  * NOTE: To convert from PBD residue numbering (i.e. the pos column in Howard2025) to `pose_index` numbering (the 1-based index of the residue in the Rosetta pose) you can use `pose_index = pose.pdb_info().pdb2pose(chain = 'A', res = pos)`.
  
    Also note that positions that are in the protein that have experimental ddG values but aren't in the structure, will give a `pose_index` zero. For this analysis we'll skip making predictions for these residues, as it would require e.g. building missing loops.
	
### 3 Plot and analyze the predicted and experimental ddG values

  * Create a scatter plot of the `-log2FoldChange` from Howard2025 vs. Rosetta predicted ddG value. Since there are outliers in for the Rosetta predicted ddG, to stabilize the variance you can plot it on on the `sqrt()` or `log10(x + 1)` transformed scale. Make a `facet_grid` with `pathway` for the columns, `compound_dose` for the columns. Fit a linear trendline for each in each panel.
  * Compute the spearman rank correlation and p-value for each group (pathway, compound, dose). Here we're using the more robust rank correlation beacuse there may be outliers.
  * Is there a significant correlation that is going in the expected direction?


### 4 Briefly discuss strengths and limitations of the modeling strategy

  * How does predicted stability in the specific conformational state relate the functional outcome of the experiment?
  * Are there critical components of the real system are not being modeled? e.g. complexities described in [Kleinau, et al., 2020](https://www.mdpi.com/1422-0067/21/16/5728).
  * Are limitations in the energy function? E.g. would using an improved energy function like the `beta_jan25` [Haddox, et al., 2025](https://www.biorxiv.org/content/10.64898/2025.12.12.691241v1.abstract) help?
  * Are the limitations in the sampling that could be improved e.g. in [Barlow, et al., 2018](https://pubs.acs.org/doi/10.1021/acs.jpcb.7b11367)?
