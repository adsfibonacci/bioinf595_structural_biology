


model_checkpoint <- "${HOME}/turbo_bioinf595/opt/NeuralPLexer/neuralplexermodels_downstream_datasets_predictions/models/complex_structure_prediction.ckpt"

source("src/dock_neuralplexer.R")

# APPROVED, should be 6 ligands
data <- readr::read_tsv(
    "data/1HXW/CHEMBL243_approved.tsv",
    show_col_types = FALSE) |>
    dplyr::distinct(Smiles, .keep_all = TRUE)

# dock in to holo receptor
data |>
    dplyr::rowwise() |>
    dplyr::do({
        ligand <- .
        substance_id <- ligand$`Molecule ChEMBL ID`[1]
        dock_neuralplexer(
            input_ligand = ligand$Smiles[1],
            input_receptor = "data/1HXW/protein.pdb",
            output_path = paste0("intermediate_data/approved_template_1HXW/", substance_id),
            model_checkpoint = model_checkpoint,
            input_template = "data/1HXW/protein.pdb",
            verbose = TRUE)
        data.frame()
    })

# dock into apo receptor
data |>
    dplyr::rowwise() |>
    dplyr::do({
        ligand <- .
        substance_id <- ligand$`Molecule ChEMBL ID`[1]
        dock_neuralplexer(
            input_ligand = ligand$Smiles[1],
            input_receptor = "data/1HHP/protein.pdb",
            output_path = paste0("intermediate_data/approved_template_1HHP/", substance_id),
            model_checkpoint = model_checkpoint,
            input_template = "data/1HHP/protein.pdb",            
            verbose = TRUE)
        data.frame()
    })

# dock into no template
data |>
    dplyr::rowwise() |>
    dplyr::do({
        ligand <- .
        substance_id <- ligand$`Molecule ChEMBL ID`[1]
        dock_neuralplexer(
            input_ligand = ligand$Smiles[1],
            input_receptor = "data/1HXW/protein.pdb",
            output_path = paste0("intermediate_data/approved_no_template/", substance_id),
            model_checkpoint = model_checkpoint,
            verbose = TRUE)
        data.frame()
    })


#############
# inactives #
#############
data <- readr::read_tsv(
    "data/1HXW/CHEMBL243_inactive.tsv",
    show_col_types = FALSE) |>
    dplyr::distinct(Smiles, .keep_all = TRUE) |>
    dplyr::filter(
        `Standard Relation` == "'>'",
        `#RO5 Violations` %in% c("None", "1"))


# dock in to holo receptor
data |>
    dplyr::rowwise() |>
    dplyr::do({
        ligand <- .
        substance_id <- ligand$`Molecule ChEMBL ID`[1]
        dock_neuralplexer(
            input_ligand = ligand$Smiles[1],
            input_receptor = "data/1HXW/protein.pdb",
            output_path = paste0("intermediate_data/inactive_template_1HXW/", substance_id),
            model_checkpoint = model_checkpoint,
            input_template = "data/1HXW/protein.pdb",
            verbose = TRUE)
        data.frame()
    })

# dock into apo receptor
data |>
    dplyr::rowwise() |>
    dplyr::do({
        ligand <- .
        substance_id <- ligand$`Molecule ChEMBL ID`[1]
        dock_neuralplexer(
            input_ligand = ligand$Smiles[1],
            input_receptor = "data/1HHP/protein.pdb",
            output_path = paste0("intermediate_data/inactive_template_1HHP/", substance_id),
            model_checkpoint = model_checkpoint,
            input_template = "data/1HHP/protein.pdb",            
            verbose = TRUE)
        data.frame()
    })

# dock into no template
data |>
    dplyr::rowwise() |>
    dplyr::do({
        ligand <- .
        substance_id <- ligand$`Molecule ChEMBL ID`[1]
        dock_neuralplexer(
            input_ligand = ligand$Smiles[1],
            input_receptor = "data/1HXW/protein.pdb",
            output_path = paste0("intermediate_data/inactive_no_template/", substance_id),
            model_checkpoint = model_checkpoint,
            verbose = TRUE)
        data.frame()
    })
