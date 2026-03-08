
dock_neuralplexer <- function(
    input_ligand,
    input_receptor,
    output_path,
    model_checkpoint,
    input_template = NULL,
    application_path = "neuralplexer-inference --task=batched_structure_sampling ",
    n_samples = 16,
    chunk_size = 4,
    num_steps = 40,
    discard_sdf_coords = TRUE,
    overwrite = FALSE,
    verbose = FALSE) {


    if (!file.exists(input_receptor)) {
        stop("Input receptor path '", input_receptor, "' does not exist\n")
    }

    if (dir.exists(output_path)) {
        if (overwrite) {
            if (verbose) {
                cat(
                    "Output path '", output_path, "' exists and ovewrite is requested \n",
                    sep = "")
            }
        } else {
            if (verbose) {
                cat(
                    "Output path '", output_path, "' exists and overwrite not requested, ",
                    "skipping ...\n", sep = "")
            }
            return(1)
        } # overwrite
    } else {
        if (verbose) {
            cat ("Creating output path '", output_path, "'\n", sep = "")
        }
        dir.create(output_path, recursive = TRUE)
    }
    cmd <- paste0(
        application_path, " ",
        "--input-receptor '", input_receptor, "' ",
        "--input-ligand '", input_ligand, "' ",
        ifelse(
            !is.null(input_template),
            paste0("--use-template  --input-template ", input_template, " "),
            ""),
        "--out-path ", output_path, " ",
        "--model-checkpoint ", model_checkpoint, " ",
        "--n-samples ", n_samples, " ",
        "--chunk-size ", chunk_size, " ",
        "--num-steps ", num_steps, " ",
        "--cuda ",
        ifelse(discard_sdf_coords, "--discard-sdf-coords ", ""),
        "--sampler=langevin_simulated_annealing ",
        "--separate-pdb")
    if (verbose) {
        cat(cmd, "\n", sep = "")
    }
    system(cmd)
    return(1)
}
