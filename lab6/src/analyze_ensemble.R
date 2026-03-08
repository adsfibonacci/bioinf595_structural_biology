# analyze_ensemble.R
# Generates score-vs-RMSD plots and Pnear analysis from relax and backrub metadata.
#
# Run from project root:
#   Rscript src/analyze_ensemble.R
#
# Sweep parameters (from run_backrub_sweep.sh):
#   MC_KT_VALUES=(0.4 0.7 1.0 1.5 10)
#   NTRIALS_VALUES=(1000 5000 10000)
#   NSAMPLES=20

library(tidyverse)
library(ggplot2)

dir.create("results", showWarnings = FALSE)

# ── 1. Load relax data ────────────────────────────────────────────────────────
relax <- read_tsv("intermediates/relaxes/relax_metadata.tsv", show_col_types = FALSE) |>
  transmute(
    source      = "FastRelax",
    tag         = "FastRelax",
    mc_kt       = NA_real_,
    ntrials     = NA_integer_,
    score       = relaxed_score,
    rmsd_native = RMSD
  )

# ── 2. Load all BackRub metadata ──────────────────────────────────────────────
backrub_files <- list.files(
  "intermediates/backrub",
  pattern    = "backrub_metadata_.*\\.tsv",
  full.names = TRUE
)

backrub <- map_dfr(backrub_files, read_tsv, show_col_types = FALSE) |>
  transmute(
    source      = "BackRub",
    tag         = tag,
    mc_kt       = as.double(mc_kt),
    ntrials     = as.integer(ntrials),
    score       = sample_score,
    rmsd_native = rmsd_to_native
  )

all_data <- bind_rows(relax, backrub)

# ── 3. Score-vs-RMSD: all data ────────────────────────────────────────────────
# x: Heavy-atom RMSD to native (Å) — how far each conformation deviates from
#    the experimental crystal structure.
# y: Rosetta REF2015 score (REU, Rosetta Energy Units) — lower = more favorable.
# A folding funnel would appear as lowest scores clustered near RMSD = 0.

p1 <- ggplot(all_data, aes(x = rmsd_native, y = score, colour = source, shape = source)) +
  geom_point(alpha = 0.75, size = 2.5) +
  scale_colour_manual(values = c("FastRelax" = "#E76F51", "BackRub" = "#264653")) +
  scale_shape_manual(values  = c("FastRelax" = 17,        "BackRub" = 16)) +
  labs(
    title    = "MC4R Ensemble: Rosetta Score vs Heavy-atom RMSD to Native",
    subtitle = "x: RMSD (Å) to experimental structure  |  y: REF2015 energy (REU, lower = better)",
    x        = "Heavy-atom RMSD to native (Å)",
    y        = "Rosetta REF2015 score (REU)",
    colour   = "Protocol",
    shape    = "Protocol"
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "bottom")

ggsave("results/score_vs_rmsd.pdf", p1, width = 7, height = 5)
ggsave("results/score_vs_rmsd.png", p1, width = 7, height = 5, dpi = 200)
message("Saved: results/score_vs_rmsd.pdf/png")

# ── 4. Score-vs-RMSD: BackRub faceted by kT x ntrials ────────────────────────
# FastRelax is excluded from this plot — only BackRub conditions are shown.
p2 <- backrub |>
  mutate(
    mc_kt   = factor(mc_kt,   levels = sort(unique(mc_kt))),
    ntrials = factor(ntrials, levels = sort(unique(ntrials)))
  ) |>
  ggplot(aes(x = rmsd_native, y = score, colour = mc_kt)) +
  geom_point(alpha = 0.8, size = 2) +
  facet_grid(mc_kt ~ ntrials, labeller = label_both) +
  scale_colour_brewer(palette = "Dark2", name = "mc_kt") +
  labs(
    title = "BackRub Score vs RMSD faceted by sampling parameters",
    x     = "Heavy-atom RMSD to native (Å)",
    y     = "Rosetta REF2015 score (REU)"
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "none")

ggsave("results/score_vs_rmsd_facet.pdf", p2, width = 12, height = 12)
ggsave("results/score_vs_rmsd_facet.png", p2, width = 12, height = 12, dpi = 200)
message("Saved: results/score_vs_rmsd_facet.pdf/png")

# ── 5. Pnear ──────────────────────────────────────────────────────────────────
# Pnear = sum( exp(-RMSD^2/lambda^2) * exp(-score/kT) ) / sum( exp(-score/kT) )
#
# Summarises funnel quality as a scalar in [0, 1].
# Pnear -> 1: Boltzmann weight concentrated near native (good funnel).
# Pnear -> 0: energy function cannot discriminate native-like structures.
# lambda = 1.5 Å (what counts as "near native"); kT = 0.62 REU (room temperature).

pnear_fn <- function(scores, rmsds, lambda = 1.5, kT = 0.62) {
  shifted   <- scores - min(scores)          # numerical stability
  boltzmann <- exp(-shifted / kT)
  gaussian  <- exp(-(rmsds^2) / (lambda^2))
  sum(gaussian * boltzmann) / sum(boltzmann)
}

pnear_relax <- pnear_fn(relax$score, relax$rmsd_native)

pnear_backrub <- map_dfr(
    backrub_files,
    \(f) read_tsv(f, show_col_types = FALSE)
  ) |>
  group_by(mc_kt, ntrials) |>
  summarise(
    n          = n(),
    mean_score = mean(sample_score),
    sd_score   = sd(sample_score),
    mean_rmsd  = mean(rmsd_to_native),
    sd_rmsd    = sd(rmsd_to_native),
    rmsd_range = max(rmsd_to_native) - min(rmsd_to_native),
    Pnear      = pnear_fn(sample_score, rmsd_to_native),
    .groups    = "drop"
  ) |>
  mutate(condition = paste0("kT=", mc_kt, "\nn=", ntrials))

pnear_summary <- bind_rows(
  tibble(
    mc_kt = NA_real_, ntrials = NA_integer_, n = nrow(relax),
    mean_score = mean(relax$score), sd_score = sd(relax$score),
    mean_rmsd  = mean(relax$rmsd_native), sd_rmsd = sd(relax$rmsd_native),
    rmsd_range = diff(range(relax$rmsd_native)),
    Pnear = pnear_relax, condition = "FastRelax"
  ),
  pnear_backrub
) |>
  arrange(desc(Pnear))

write_tsv(pnear_summary, "results/pnear_summary.tsv")
message("Saved: results/pnear_summary.tsv")
print(pnear_summary |> select(condition, Pnear, mean_rmsd, sd_rmsd, mean_score))

# ── 6. Pnear bar plot ─────────────────────────────────────────────────────────
p3 <- pnear_summary |>
  mutate(
    condition = fct_reorder(condition, Pnear, .desc = TRUE),
    is_relax  = is.na(mc_kt)
  ) |>
  ggplot(aes(x = condition, y = Pnear, fill = is_relax)) +
  geom_col(width = 0.65, colour = "black", linewidth = 0.3) +
  geom_text(aes(label = round(Pnear, 3)), vjust = -0.4, size = 3) +
  geom_hline(yintercept = pnear_relax, linetype = "dashed",
             colour = "#E76F51", linewidth = 0.8) +
  annotate("text", x = 0.6, y = pnear_relax + 0.03,
           label = paste0("FastRelax Pnear = ", round(pnear_relax, 3)),
           colour = "#E76F51", hjust = 0, size = 3.5) +
  scale_fill_manual(values = c("TRUE" = "#E76F51", "FALSE" = "#264653"),
                    guide  = "none") +
  scale_y_continuous(limits = c(0, 1.12), expand = expansion(0)) +
  labs(
    title    = "Pnear funnel quality per sampling condition",
    subtitle = "lambda = 1.5 Ang, kT = 0.62 REU  |  Higher = more Boltzmann weight near native",
    x        = NULL,
    y        = "Pnear"
  ) +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(size = 8))

ggsave("results/pnear_barplot.pdf", p3, width = 12, height = 5)
ggsave("results/pnear_barplot.png", p3, width = 12, height = 5, dpi = 200)
message("Saved: results/pnear_barplot.pdf/png")
