

## for the yardstick package
# install.packages("tidymodels")

smina_scores <- data.frame(
    smina_scores_fname = list.files(
        "intermediate_data",
        pattern = "smina_scores.tsv",
        full.names = TRUE,
        recursive = TRUE)) |>
    dplyr::rowwise() |>
    dplyr::do({
        smina_scores_fname <- .$smina_scores_fname[1]
        smina_scores <- readr::read_tsv(
            smina_scores_fname,
            show_col_types = FALSE)
        smina_scores |>
            dplyr::mutate(
                fname = smina_scores_fname,
                .before = 1)
    })

smina_scores <- smina_scores |>
  dplyr::mutate(
    active = fname |> stringr::str_detect("approved") |> as.factor(),
    template = fname |>
      stringr::str_replace("intermediate_data/[^_]+_", "") |>
      stringr::str_replace("/smina_scores.tsv", ""))


scores <- smina_scores_best |>
  dplyr::group_by(template) |>
  dplyr::do({
    x <- .
    scores <- x |> yardstick::roc_auc(affinity, truth = active)

    scores_resample <- data.frame(replica_index = c(1:300)) |>
      dplyr::rowwise() |>
      dplyr::do({
        replica_index <- .$replica_index[1]
        x_resampled <- x |>
          dplyr::mutate(
            affinity = sample(affinity, size = dplyr::n(), replace = FALSE))
        auroc_resample = x_resampled |>
            yardstick::roc_auc(affinity, truth = active)
        data.frame(
          replica_index,
          auroc_resample = auroc_resample$.estimate[1]) 
      }) |>
      dplyr::ungroup()
    
    data.frame(
      auroc_resample = scores_resample$auroc_resample,
      auroc = scores$.estimate[1])
  }) |>
  dplyr::ungroup()


ggplot2::ggplot() +
  ggplot2::theme_bw() +
  ggplot2::geom_histogram(
    data = scores,
    mapping = ggplot2::aes(
      x = auroc_resample),
    bins = 30) +
  ggplot2::geom_vline(
    data = scores |> dplyr::distinct(template, .keep_all = TRUE),
    mapping = ggplot2::aes(
      xintercept = auroc)) +
  ggplot2::facet_wrap(facets = dplyr::vars(template)) +
  ggplot2::ggtitle(
    "AUROC Enrichment of Approved vs. Inactive",
    subtitle = "line is computed AUROC, density is permuted scores") +
  ggplot2::scale_x_continuous("(worse) <-- AUROC --> (better)")

ggplot2::ggsave(
  filename = "product/enrichment_by_template_20250327.pdf",
  width = 6,
  height = 3)
