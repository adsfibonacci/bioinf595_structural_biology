
import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from matplotlib import pyplot as plt

import h2o
from h2o.automl import H2OAutoML

with open("parameters.yaml") as parameters_file:
    parameters = yaml.safe_load(parameters_file)


# Start the H2O cluster (locally)
h2o.init()


data_train = pq.read_table(
    "intermediate_data/data_train.parquet").to_pandas()
data_validation = pq.read_table(
    "intermediate_data/data_validation.parquet").to_pandas()
data_test = pq.read_table(
    "intermediate_data/data_test.parquet").to_pandas()


# Convert pandas DataFrames to H2OFrames
h2o_data_train = h2o.H2OFrame(data_train)
h2o_data_validation = h2o.H2OFrame(data_validation)
h2o_data_test = h2o.H2OFrame(data_test)

features = [col for col in h2o_data_train.columns if col not in ["SMILES", "AID", "PUBCHEM_CID"]]
target = 'Y'

h2o_data_train[target] = h2o_data_train[target].asfactor()
h2o_data_validation[target] = h2o_data_validation[target].asfactor()
h2o_data_test[target] = h2o_data_test[target].asfactor()

for AID in parameters['target_AIDs']:
    print(f"Fitting model for AID: {AID}")
    h2o_data_train_target = h2o_data_train[h2o_data_train['AID'] == AID]
    h2o_data_validation_target = h2o_data_validation[h2o_data_validation['AID'] == AID]
    h2o_data_test_target = h2o_data_test[h2o_data_test['AID'] == AID]
    aml = H2OAutoML(max_models=1, max_runtime_secs=300, seed=42)
    aml.train(x=features, y=target, training_frame=h2o_data_train_target)
    shaps_data=h2o_data_test_target[features]
    aml.leader.shap_summary_plot(shaps_data)

    plt.savefig(
        f"product/shap_model_summary_{AID}_{parameters['date_code']}.pdf",
        format="pdf",
        bbox_inches="tight")
