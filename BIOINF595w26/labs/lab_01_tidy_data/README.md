# Lab 1: Tidy Data

The goal of this lab is to get familiar with processing tidy data and R.

In the lab you will load and process a few plate-based data formats, and
plot the data.


## 1 Setup environment
1) Install Rstudio
2) Install the following packages
  - tidyverse


## 2 Load a plate-reader based data file
This data is a growth curve for Candida auris under different drug treatments
data taken from (Metzner, 2023, 10.1128/aac.00503-23), figure 3E.

1) Load and process the data into a data.frame `dr_plate`
- Use `readr::read_tsv(...)` to extract the data between lines 35-92 and
  `dplyr::rename(...)`. to rename the first two columns to be `time_point` and
  `temperature`.
- Use `tidyr::pivot_longer(...)` to convert the data from wide to long format,
  with so the resulting `data.frame` has columns [`time_point`, `tempererature`,
  `well_id`, `growth`].
- Use `dplyr::mutate(...)` to add columns [`row`, `column`, `treatment`,
  `dose_uM` `log_dose`, `dose_label`]:
  * for `row` and `column` use `stringr::str_extract(...`) to get the letter and
    number parts. To convert the letter to the row index, try using e.g. ChatGPT
    or Gemini with a prompt like this: "I have a column in an R data.frame that
    are coded as upper-case letters and I would like to convert these into the
    numeric index of the letter, e.g. "A" -> 1, "B" -> 2, etc. for each row. Can
    you please give me example tidyverse code to do this?"
  * Use `dplyr::case_when()` to assign treatments based on the plate row:
    * rows 1, 8 are "DMSO""
    * rows 2-4 are "BAY 11-7082"
    * rows 5-7 are "BAY 11-7085"
  * The doses are in a 2-fold dilution series [100, 50, 25, 12.5, ...] for
    columns [1, 2, 3, 4, ...]
  * For `log10_dose`, recall 1 uM = 1e-6 M so, on the log scale `100 uM` => `-4`
  * For `dose_label`, make it a factor with levels e.g. `100 uM`, `50 uM` etc.
    with at 2 significant figures.
- Add the replica index for each row of each treatment
  * create a new `data.frame` with columns [`row`, `treatment`, `replica`]
    * Use `dplyr::distinct(...)` to filter the data to get distinct [`row`,
      `treatment`]
    * Use `dplyr::group_by(...)` to group by treatment
    * Use `dplyr::mutate(...)` and `dplyr::row_number()` to add the replica
      index for each row of each treatment`. Make the replica index a factor.
    * Use `dplyr::left_join(...)` to join this data.frame with the original
      `data.frame` using [`row`, `treatment`] as the join columns.
- Use `dplyr::select(...)` to select and re-order the columns so that.
  Check that the data types are correct.

```{r }
> dr_plate |> dplyr::glimpse()
Rows: 5,472
Columns: 7
$ row         <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, …
$ column      <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1…
$ treatment   <chr> "DMSO", "DMSO", "DMSO", "DMSO", "DMSO", "DMSO", "DMSO", "DMSO", "DMSO", "DM…
$ log10_dose  <dbl> -4.00000, -4.30103, -4.60206, -4.90309, -5.20412, -5.50515, -5.80618, -6.10…
$ dose_label  <fct> 100 uM, 50 uM, 25 uM, 12 uM, 6.2 uM, 3.1 uM, 1.6 uM, 0.78 uM, 0.39 uM, 0.2 …
$ temperature <dbl> 25.4, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4, 25.4, 25.…
$ growth      <dbl> 0.138, 0.126, 0.130, 0.122, 0.121, 0.117, 0.120, 0.140, 0.140, 0.143, 0.125…
```

- Use `save(...)` to save the file as an `.Rdata` file into the `intermediate/`
  directory.

## 3 Plot the growth curve data as facets
Generate a facet grid of plots that show in each panel how the growth of each
replica changes over time. The rows of the facet panels should be different
treatments and the columns should be different concentrations. 

- Add a `ggplot2::geom_line(...)` that maps
  * time_point -> x
  * growth -> y
  * replica -> color
- Create a grid of facets where
  * treatment -> row
  * dose_label -> column
- Use the following scale transformations
  * Convert HMS time to hours an have breaks at [0, 6, 12, 18, 24]
  * Convert growth to percentage (e.g. multiply by 100) with breaks [0, 50, 100]
  * Use discrete color scales for replica
- Use the following theme
  * `ggplot2::theme_bw()`
  * Turn the x-axis labels to be 90 degrees sideways
  * Use the following aesthetic element labels
    * x-axis => "Time Point (h)"
    * y-axis => "OD of Control (%)"
    * color => "Replica"
    * Title => "Drug Inhibition of Growth Kinetics"
- Save the plot as a pdf at a size where the facet labels are legible
  * Save into the `product/` with the extension use `_YYYMMDD.pdf`
  

## 3 Plot how the growth at different timepoints changes by concentration
Adapt the previous plot to show how if the 6, 12, 18, or 24 hour time-points
are used as end-point measurements, how the response at these end-points change
by dose. 

Generate a facet grid of plots that show in each panel how the growth of each
replica changes over concentration. The rows of the facet panels should be
different end-points (6, 12, 18, or 24) hours, and the the columns should be
different treatments.

- Use `dplyr::filter()` with the `%in%` operator to filter the dr_plate data to
  only have 4 time points
- Change the mapping for `geom_line()` so that  
  * log10_dose -> x
  * growth -> y
  * replica -> color
- Change the grid of facets where
  * time_point -> row
  * treatment -> column
- Change the scales as follows
  * Scale for x to have breaks at the unique log10_dose values and labels
    as the corresponding dose_label values.
- Give the plot appropriate title and labels for the axis and color scales
- Save the plot as a pdf at a size where the facet labels are legible
  * Save into the `product/` with the extension use `_YYYMMDD.pdf`
  
## 4 Briefly answer the following questions

1) Describe in words what these plots show. Are there effects due to the time 
   point and/or concentration?

2) By eye, are there any significant difference between the replicas?

3) If you had to pick a single timepoint rather than a full time-course, what
   timepoint would you choose?

4) In the paper the goal was to identify compounds that specifically block
   filamentation because that is thought to be important for virulence. In
   figure 2a it show how the filamentation score depends on the concentration
   of the drug. giving an IC50 of 15 uM for BAY 11-7082. These data show how
   overall growth depends on the concentration of the drugs. Is the effect on
   filamentation seperable from effect on growth? I.e. is there a concentration
   where there is an effect on one but not the other?
   
   Given the overall goal is to block virulence, is identifying compounds that
   selectively target filamentation important?
