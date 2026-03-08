


data <- readr::read_tsv(
    file = "intermediate_data/approved_template_1HXW/rmsd.tsv",
    show_col_types = FALSE) |>
    dplyr::mutate(
        ligand_id = output_path |> stringr::str_replace("intermediate_data/approved_template_1HXW/", ""))

plot <- ggplot2::ggplot() +
    ggplot2::geom_point(
        data = data,
        mapping = ggplot2::aes(
            x = rmsd,
            y = log10(affinity - affinity_best + 1)),
        alpha = 0.7,
        size = 0.7,
        shape = 16) +
    ggplot2::scale_y_continuous(
        "SMINA dG to Lowest Energy (kcal/mol) (lower is better)",
        breaks = log10(c(0, 1, 3, 10, 30, 100) + 1),
        labels = c("0", "1", "3", "10", "30", "100")) +
    ggplot2::facet_wrap(
        facets = dplyr::vars(ligand_id))

if (!dir.exists("product/approved_template_1HXW")) {
    dir.create("product/approved_template_1HXW")
}

ggplot2::ggsave(
    filename = "product/approved_template_1HXW/score_vs_rmsd_20250328.pdf",
    width = 5,
    height = 5,
    useDingbats = FALSE)


