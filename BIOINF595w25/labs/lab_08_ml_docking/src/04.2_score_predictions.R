
source("src/score_poses_with_smina.R")

smina_scores <- score_poses_with_smina("intermediate_data/approved_template_1HXW")
smina_scores |>
    readr::write_tsv("intermediate_data/approved_template_1HXW/smina_scores.tsv")

smina_scores <- score_poses_with_smina("intermediate_data/approved_template_1HHP")
smina_scores |>
    readr::write_tsv("intermediate_data/approved_template_1HHP/smina_scores.tsv")

smina_scores <- score_poses_with_smina("intermediate_data/approved_no_template")
smina_scores |>
    readr::write_tsv("intermediate_data/approved_no_template/smina_scores.tsv")




smina_scores <- score_poses_with_smina("intermediate_data/inactive_template_1HXW")
smina_scores |>
    readr::write_tsv("intermediate_data/inactive_template_1HXW/smina_scores.tsv")

smina_scores <- score_poses_with_smina("intermediate_data/inactive_template_1HHP")
smina_scores |>
    readr::write_tsv("intermediate_data/inactive_template_1HHP/smina_scores.tsv")

smina_scores <- score_poses_with_smina("intermediate_data/inactive_no_template")
smina_scores |>
    readr::write_tsv("intermediate_data/inactive_no_template/smina_scores.tsv")
