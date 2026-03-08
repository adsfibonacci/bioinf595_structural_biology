
# example with HIV-1 Protease
MODEL_PATH=${HOME}/turbo_bioinf595/opt/NeuralPLexer/neuralplexermodels_downstream_datasets_predictions/models
DATA_PATH=data/1HXW
OUTPUT_PATH=intermediate_data/1HXW_A-84538_test

mkdir -p ${OUTPUT_PATH}
neuralplexer-inference \
    --task=batched_structure_sampling \
    --input-receptor ${DATA_PATH}/protein.pdb \
    --input-ligand ${DATA_PATH}/ligand.pdb \
    --use-template  --input-template ${DATA_PATH}/protein.pdb \
    --out-path ${OUTPUT_PATH} \
    --model-checkpoint ${MODEL_PATH}/complex_structure_prediction.ckpt \
    --n-samples 16 \
    --chunk-size 4 \
    --num-steps=40 \
    --cuda \
    --sampler=langevin_simulated_annealing \
    --separate-pdb

