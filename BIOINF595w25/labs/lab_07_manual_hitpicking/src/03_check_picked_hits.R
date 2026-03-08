


tested <- readr::read_csv("data/D4_in_vitro_data.csv")
picked <- data.frame(
    line = readr::read_lines("intermediate_data/D4_in_vitro.mol2")) |>
    dplyr::filter(
        line |> stringr::str_detect("##########                 Name:")) |>
    dplyr::mutate(
        zincid = line |> stringr::str_extract("ZINC.*$"))

# this should be empty if all the picked compounds were tested
picked |> dplyr::anti_join(
    tested,
    by = "zincid")
