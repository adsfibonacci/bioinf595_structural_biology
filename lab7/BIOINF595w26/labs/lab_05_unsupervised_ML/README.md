# Generate and explore embeddings

Building on the lab from last week, in this lab you will create and analyze
stochastic neighbor embeddings


## Learning objectives

 * How to create, tune, and visualize UMAP embeddings



## Steps

### 1 Create a workflow for creating UMAP embeddings

Write python functions to fit the umap

    a) the input dataset should have columns [substance_id, smiles]
	   the output dataset should have columns [substance_id, smiles, UMAP_1, UMAP_2]
	   Where UMAP_1 and UMAP_2 are the coordinates after transforming each molecule
	   to a molecular fingerprint, then jointly projecting to a moderate number of dimensions
	   (e.g. 100), then using UMAP to project down to lower dimesions (e.g. 2).

    b) To limit the amount of memory required, incrementally compute the PCA
	   transformation and then project each compounds in batches. While this requires
	   computing the fingerprints for each compound twice, the total amount of memory
	   required is much less.

	
	   pca_reducer = sklearn.decomposition.IncrementalPCA(...)
	   
	   for i in range(0, n_substances, batch_size):
	       batch_smiles = ...
		   batch_features = ...
		   pca_reducer.partial_fit(batch_features)
		   
	   for i in range(0, n_substances, batch_size):
	       batch_smiles = ...
		   batch_features = ...
		   pca_embedding_batch = pca_reducer.transform(batch_features)

       pca_embedding = ...
	   
           
	c) Then use umap from the umap-learn package to fit the umap embedding
	
	   umap_reducer = umap.UMAP(...)
	   umap_embedding = umap_reducer.fit_transform(pca_embedding)
	   
	   
2 Write functions to visualize the UMAP

    a) Use the holoviewsdatashader package to visualize embeddings without overplotting
	   https://datashader.org/user_guide/Plotting_Pitfalls.html
	
       canvas = datashader.Canvas(
         plot_width=width,
         plot_height=height).points(umap_embedding, umap_field_1, umap_field_2)
	   canvas = holoviews.operation.datashader.rasterize(canvas)
       canvas = datashader.transfer_functions.shade(
	     canvas, cmap=datashader.colors.viridis)
       canvas.to_pil().convert('RGB').save(output_fname)

### 2 Apply the work flow to explore embeddings for make-on-demand chemical space

   a) Select the same dataset from lsd.docking.org you used for the last lab,
      download it from huggingface and subsample small=10_000, medium=100_000,
	  and large=1_000_000 substance subsets. We'll use the model you trained in the
	  last lab in the next step.
	  
   b) For the small=10_000 substance subset, do a hyper-parameter search over at least two
      of the following parameters generating ~10 embeddings
	  
      * fingerprint_type (e.g. ECFP)
	  * metric (https://umap-learn.readthedocs.io/en/latest/parameters.html#metric)
	  * a and b parameters (which are related to min_dist, and num_neighbors)
	  * PCA pre-processing dimension

   c) Select a set of parameters that you think looks good and on the medium and large
      subsets.
	  
   d) Describe the parameters you adjusted and what you looked for to make your selection,
      especially in terms of attraction/repulsion dynamics described in
	  (Damrich, et al., 2024, DOI: 10.1101/2024.04.26.590867))

### 3 Compare the distribution of docking scores and model predictions

   a) Use datashader to plot the docking score, the predicted docking scores
      and the residuals (i.e. score - pred) for the large dataset.
	  
   b) Are the errors localized?


### 4 Explore the embeddings using the interactive app

   a) Install and open the chemical_embedding app located in src/chemical_embedding
      on the github page for the lab. If you have issues with package conflicts,
	  you can try running it through the docker image.
	  
   b) Use the BoxEdit tool to select specific clusters and describe
      how the chemicals are chemically similar?
	  
   c) What are some of the chemical features of chemicals that have good score?
      And, good score and bad predictions?
	  
	  
## What to turn in 

    * Show the a selection of the embeddings based on the hyperparameter choices
	* Answer the questions, and the chemicals identified exploring the map.
	* Show the code you used to do the embeddings.


