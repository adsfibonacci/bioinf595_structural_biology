# Analysis workflow to curate autofluorescence data from (Simeonov 2008)

## Goal of the lab:
Curate data from PubChem and create a hugging face dataset ready for machine learning. The dataset comes from

  Simeonov A, Jadhav A, Thomas CJ, Wang Y, Huang R, Southall NT, Shinn P, Smith J, Austin CP, Auld DS, Inglese J.
  Fluorescence spectroscopic profiling of compound libraries.
  J Med Chem. 2008 Apr 24;51(8):2363-71. DOI: 10.1021/jm701301m. Epub 2008 Mar 26. PMID: 18363325.

Abstract:

  Chromo/fluorophoric properties often accompany the heterocyclic
  scaffolds and impurities that comprise libraries used for
  high-throughput screening (HTS). These properties affect assay outputs
  obtained with optical detection, thus complicating analysis and
  leading to false positives and negatives. Here, we report the
  fluorescence profile of more than 70,000 samples across spectral
  regions commonly utilized in HTS. The quantitative HTS paradigm was
  utilized to test each sample at seven or more concentrations over a
  4-log range in 1,536-well format. Raw fluorescence was compared with
  fluorophore standards to compute a normalized response as a function
  of concentration and spectral region. More than 5% of library members
  were brighter than the equivalent of 10 nM 4-methyl umbelliferone, a
  common UV-active probe. Red-shifting the spectral window by as little
  as 100 nm was accompanied by a dramatic decrease in
  autofluorescence. Native compound fluorescence, fluorescent
  impurities, novel fluorescent compounds, and the utilization of
  fluorescence profiling data are discussed.


Key learning objectives:
  1) Developing a data analysis workflow
  2) Practice working with large-scale data from PubChem
  3) Creation of HuggingFace datasets


## 1 Load Active Compounds from Simeonov2008

### Inspect the data in pubchem on the web
The assay IDs for this study are 587, 588, 590, 591, 592, 593, 594

Go to https://pubchem.ncbi.nlm.nih.gov/ and search for AID=587 as an example of the above assay ids
   * Advanced Search
   * Filter by BioAssay (AID)
   
Inspect the dataset
   * What is the overall point of the study, and what is this specific dataset measuring?
   * How many were screened, and how many were inactive, inconclusive, or active?
   * What is a CID and InChiI key of one of the active molecules?
     (well use this molecule as a sanitity check after we've parsed it)
	 
### 2 Load the active compounds using the API

In R use the get_aid_compounds functions to download the active compounds for each AID
  * look at https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest#section=URL-based-API for details
    about the PubChem "power user gateway" (PUG) API.
  * You can use the split-apply-combine pattern from lecture 2.
  * use dplyr to count the number of compounds that are active in multiple datasets
  * Save the data as a .tsv file with columns
      AID, CID, SMILES, InChiI, InChIKey
	  
### 3 Sanitize the SMILES using MolVS or RDKit
  * Following what I showed in Lecture 3, use MolVS (https://molvs.readthedocs.io/en/latest/)
    or RDKit (https://github.com/rdkit/rdkit/blob/master/Docs/Notebooks/MolStandardize.ipynb) to
	sanitize the SMILES for each molecule
  * How many of the SMILES changed upon sanitization?
  * Save the results as a .tsv with the sanitized smiles as an additional column SMILES_sanitized

### 4 Upload the dataset to huggingface
  * HuggingFace is a platform for sharing data, models, and spaces.
    See https://docs.google.com/document/d/16E6D06eT18Qb7eDpnTjW6MQXhuq55DkTdTU2YWA7XaI/edit?usp=sharing
	for a detaile guide on creating a dataset
  * Create a personal respository called <username>/Simeonov2008
  * Use src/upload_to_hf.py and push_to_hub to load the dataset with santizied smiles 
  * Create a dataset card that contains
     * Description of the ovrall dataset including proper citations
	 * A quick-start guide for how to load it into python
	 * Brief details about the specific assays and data
  
  
  
 # What to turn in
 Please submit:
   * Brief answers to the exploratory data questions above
   * A link to your huggingface repo
      - I will check that it can be downloaded using the datasets python package
	    (which will be the case if you used push_to_hub)
	  - If the dataset card is brief, informative, and clear
   * The code that you wrote
