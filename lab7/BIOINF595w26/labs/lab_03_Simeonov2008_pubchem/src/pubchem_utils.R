library(httr2)
library(purrr)
library(dplyr)

# These utils help retrieving the compounds for an assay using the
# the (complex) PubChem pug api.
#
# To use:
#
#   # an AID is the PubChem identifier for the assay
#   aid <- 587
#
#   cids <- get_aid_compounds(aid)
#   cids |> dplyr::glimpse()
#
#    Rows: 58,464
#    Columns: 4
#    $ CID      <int> 51, 108, 119, 138, 178, 199, 248, 249, 286, 289, 305, 339, 500, 525, 535, 546, 547, 564…
#    $ SMILES   <chr> "C(CC(=O)O)C(=O)C(=O)O", "C1=CC(=CN=C1)CC(=O)O", "C(CC(=O)O)CN", "C(CCN)CC(=O)O", "CC(=…
#    $ InChI    <chr> "InChI=1S/C5H6O5/c6-3(5(9)10)1-2-4(7)8/h1-2H2,(H,7,8)(H,9,10)", "InChI=1S/C7H7NO2/c9-7(…
#    $ InChIKey <chr> "KPGXRSRHYNQIFN-UHFFFAOYSA-N", "WGNUNYPERJMVRM-UHFFFAOYSA-N", "BTCSSZJGUNDROE-UHFFFAOYS…


####################
# HELPER FUNCTIONS #
####################

#' The logic for building up this url:
#' https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest#section=URL-based-API
#'
#'  <input specification> = <domain>/<namespace>/<identifiers>
#'  <domain> = ... assay ...
#'  assay domain <namespace> = aid ...
#'  <identifiers> = comma-separated list of positive integers (e.g. cid, sid, aid)
#'  <operation specifications> = ... cids ...
#'  <output specification> = ... JSON ...
#'
#' SIDS / CIDS / AIDS
#' option:
#'   list_return
#'     allowed values: grouped, flat, listkey
#'     Type of identifier list to return
#'   cids_type
#'     allowed values: all, active, inactive
#'
#' Request Headers
#'  JSON: Accept = 'application/json'
#'
aid_to_id_list <- function(
    aid,
    cids_type,
    verbose = FALSE) {

    req <- request("https://pubchem.ncbi.nlm.nih.gov/rest/pug") |>
        httr2::req_url_path_append("assay", "aid", aid, "cids", "JSON") |>
        httr2::req_url_query(
            cids_type = cids_type,
            list_return = "listkey") |>
        httr2::req_headers(Accept = "application/json")

    if (verbose) {
        req |> capture.output() |> cat("\n")
    }
    req <- req |> httr2::req_perform()

    res <- httr2::resp_body_json(req)
    list(
        listkey = res$IdentifierList$ListKey,
        size = res$IdentifierList$Size)
}


#' The logic for building up this url:
#' https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest#section=URL-based-API
#'
#'  <input specification> = <domain>/<namespace>/<identifiers>
#'  <domain> = ... compound ...
#'  compound domain <namespace> = ... listkey ...
#'  <identifiers> = comma separated list of positive ingegers (e.g. the returned listkey)
#'  compound domain <operation specification> = cids
#'  <output specification> = ... CSV ...
#'
#' Request Headers
#'  CSV: Accept = 'text/csv'
#'
id_list_to_cids <- function(
    id_list,
    verbose = FALSE) {

    req <- request("https://pubchem.ncbi.nlm.nih.gov/rest/pug") |>
        httr2::req_url_path_append("compound", "listkey", id_list$listkey, "cids", "JSON") |>
        httr2::req_headers(Accept = "application/json") |>
        httr2::req_retry(
            max_tries = 5,
            backoff = ~ 2 ^ .x,
            is_transient = function(resp) httr2::resp_status(resp) >= 500)
    
    if (verbose) {
        req |> capture.output() |> cat("\n")
    }
    req <- req |> httr2::req_perform()
    
    res <- req |> httr2::resp_body_json()
    data.frame(cid = res$IdentifierList$CID |> unlist())
}



#' The logic for building up this url:
#' https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest#section=URL-based-API
#'
#'  <input specification> = <domain>/<namespace>/<identifiers>
#'  <domain> = ... compound ...
#'  compound domain <namespace> = ... listkey ...
#'  <identifiers> = comma separated list of positive ingegers (e.g. the returned listkey)
#'  compound domain <operation specification> = <compound property>
#'  <compound property> = property / [comma-separated list of property tags]
#'  <output specification> = ... CSV ...
#'
#' Compound Property Tables
#'   SMILES
#'   InChI
#'   InChIKey
#'
id_list_to_properties <- function(
    id_list,
    properties,
    listkey_start,
    listkey_count,
    verbose = FALSE) {

    properties_string <- paste(properties, collapse = ",")
    
    req <- request("https://pubchem.ncbi.nlm.nih.gov/rest/pug") |>
        httr2::req_url_path_append(
            "compound", "listkey", id_list$listkey,
            "property", properties_string, "JSON") |>
        httr2::req_url_query(
            listkey_start = listkey_start,
            listkey_count = listkey_count) |>
        httr2::req_headers(Accept = "application/json") |>
        httr2::req_retry(
            max_tries = 5,
            backoff = ~ 2 ^ .x,
            is_transient = function(resp) httr2::resp_status(resp) >= 500)

    if (verbose) {
        req |> capture.output() |> cat("\n")
    }
    req <- req |> httr2::req_perform()

    res <- req |> httr2::resp_body_json()

    res$PropertyTable$Properties |>
        dplyr::bind_rows()

}

get_aid_compounds <- function(
  aid,
  cids_type = "all",
  properties = c("SMILES", "InChi", "InChiKey"),
  pagination_size = 10000,
  throttle = 0.2,
  verbose = FALSE) {

    id_list <- aid_to_id_list(
        aid = aid,
        cids_type = cids_type,
        verbose = verbose)

    tibble::tibble(
        listkey_start = seq(0, id_list$size - 1, by = pagination_size),
        listkey_count = pmin(pagination_size, id_list$size - listkey_start)) |>
        dplyr::rowwise() |>
        dplyr::do({
            pagination <- .
            id_list_to_properties(
                id_list = id_list,
                properties,
                listkey_start = pagination$listkey_start,
                listkey_count = pagination$listkey_count,
                verbose = verbose)
        }) |>
        dplyr::ungroup()
}
