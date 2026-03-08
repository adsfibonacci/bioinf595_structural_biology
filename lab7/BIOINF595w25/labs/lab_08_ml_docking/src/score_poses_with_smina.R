

score_poses_with_smina <- function(
    base_path,
    smina_path = "${HOME}/turbo_bioinf595/opt/bin/smina") {
    data.frame(
        output_path = list.dirs(
            base_path,
            full.name = TRUE)) |>
        # skip base folder
        dplyr::filter(dplyr::row_number() > 1) |>
        dplyr::rowwise() |>
        dplyr::do({
            dock_run <- .
            output_path <- dock_run$output_path[1]
            cat("scoring poses for '", output_path, "'\n", sep = "")
            data.frame(
                ligand_fname = list.files(output_path, pattern = "[0-9]+.sdf")) |>
                dplyr::mutate(
                    pose_id = ligand_fname |> stringr::str_extract("lig_([0-9]+).sdf", group = 1),
                    receptor_fname = paste0("prot_", pose_id, ".pdb")) |>
                dplyr::rowwise() |>
                dplyr::do({
                    pose <- .
                    pose_id <- pose$pose_id[1]
                    ligand_fname <- paste0(output_path, "/", pose$ligand_fname[1])
                    receptor_fname <- paste0(output_path, "/", pose$receptor_fname[1])
                    cmd <- paste0(
                        smina_path, " ",
                        "--score_only ",
                        "-l ", ligand_fname, " ",
                        "-r ", receptor_fname, " | ",
                        "grep \"Affinity: \"")
                    cat(cmd, "\n", sep = "")
                    affinity_line <- system(cmd, intern = TRUE)
                    affinity <- affinity_line |>
                        stringr::str_extract("Affinity: ([\\-0-9.]+)", group = 1) |>
                        as.numeric()
                    if (is.na(affinity)) {
                        cat("Affinity is NA, inspect what is happening:")
                        browser()
                    }
                    data.frame(
                        output_path = output_path,
                        pose_id = pose_id,
                        ligand_fname = ligand_fname,
                        receptor_fname = receptor_fname,
                        affinity = affinity)
                })
        })
}
