
# Supervised Machine Learning

Training supervised machine learning to predict outcomes of a virtual screen

We will leverage data from https://lsd.docking.org/

   Hall BW, Tummino TA, Tang K, Mailhot O, Castanon M, Irwin JJ, Shoichet BK.
   A Database for Large-Scale Docking and Experimental Results. J Chem Inf Model.
   2025 May 12; 65(9):4458-4467, DOI: http://dx.doi.org/10.1021/acs.jcim.5c00394

featurization from DataMol / molfeat

    datamol-io/molfeat: 0.9.4 (2023)
    https://zenodo.org/records/8373019
    

and the [AutoGluon](https://auto.gluon.ai/) platform for supervised prediction.

## Learning objectives

Practice a workflow for loading data, training and evaluating machine
learning models, and assessing the results. 

**NOTE**: These datasets are quite large so part of the exercise here is to learn how to work
with real world large datasets. 

  * Subsample the data
  * Train simple models first as baseline before moving to more complex/computationally expensive models
  * Use appropriate computational resources. You can use your laptop or the UMich greatlakes cluster.
    If you do use greatlakes, don't run production code on the head node!

##  Analysis

As the size of chemical libraries increases it becomes challenging to screen them. One strategy is to use
retrieval augmented docking (RAD) described here,

    Hall, B. W.; Keiser, M. J. Retrieval Augmented Docking Using
	Hierarchical Navigable Small Worlds. J. Chem. Inf. Model. 2024,
	64 (19), 7398– 7408,  DOI: 10.1021/acs.jcim.4c00683

The idea is to dock a subset of the library, train a cheap ML predictor, and then continue use the
predictor to prioritize screening a portion of the remainder of the library. In this lab, we will
explore training ML models for RAD.

### Set up

Install packages

 * autogluon
 * datamol, molfeat
 
 
### 1 Load data from HuggingFace

 * The LSD dataset are available from https://huggingface.co/IrwinLab
 * Download the one of the datasets from the LSD using the HuggingFace `datasets` library
 * Since the HuggingFace datasets don't have good dataset cards, find the references from https:lsd.docking.org/
   and briefly give the refernce for the paper and describe what the data represents.

### 2 Prepare data for machine learning

 * From the dataset subsample small, medium, and large subsets to make modeling easier,
   and start with the small subset

In virtual screening we care most about discriminating the top set of molecules, as we expect most to be non-binders
 * Transform the score to the log scale by subtracting off the smallest value, and using the `log(x + 1)` transform.

Compute features for each molecule
 * Pick a feature representation from datamol and compute features using `molfeat`
 * Split the data using `train_test_split` from sklearn twice to get train: 60%, validation: 20%, test = 20% 
 * Inspect the data to make sure it has been loaded appropriately

### 3 Train an ensemble model using AutoGluon

 * Check the AutoGluon documentation to understand how the data should be passes in
 * Use the `presets` and `time_limit` functions to

### 4 Evaluate the model

 * Call `.leaderboard()` on the validation split. In AutoGluon high scores are better.
   Compare the `score_val` for the cross validation score on the training data vs. the
   `score_test` for the validation data. In terms of the bias vs. variance trade-off,
   do you see evidence of overfitting?
   
 * Call `.evaluate()` on the validation data. Describe the fit of the model by describing one of the metric,
   how it relates to the task, whether smaller or larger numbers are better, and what completely random
   predictor would give for the metric.
   
### 5 Retrain the model to test sensitivity
For retrieval augmented docking task, in this lab we care about these objectives
  * training compute requirements (speed, memory)
  * prediction accuracy
  * inference compute requirements (speed, memory)
  
In the next lab well explore the prediction diversity.

Retrain 3-5 versions of the AutoGluon model adjusting either the dataset size, fingerprint types,
or training parameters. Generate one or more plots and describe how these choices affect the task goals.

# What to turn in:

* Answers to the questions in steps 1-4.
* Plot and description of the analysis for question 5
* Code to do the analysis

