#!/usr/bin/env Rscript --vanilla

library(decoupleR)
library(dorothea)
library(readr)
library(dplyr)
library(tibble)
library(optparse)

parser <- OptionParser(
  usage = "usage: %prog diff_expr_file output_file [options]",
  option_list = list(
    make_option(c("-t", "--tsv"), default=F, help="Assume file is TSV instead of CSV. Default = FALSE"),
    make_option(c("-w", "--weight_col"), default="t", help="Name of the column in the diff_expr file with the weight to be used for VIPER. Default = t (t-statistic)"),
    make_option(c("-i", "--id_col"), default="ID", help="Name of the column in the diff_expr file with the gene identifiers. Default = ID"),
    make_option(c("-m", "--minsize"), default=10, help="Min size allowed for regulons"),
    make_option(c("-s", "--source"), default="tf", help="Name of the source column in the PKN"),
    make_option(c("-c", "--confidence"), default="A,B,C", help="Levels of confidence to include, separated by commas, e.g. 'A,B,C'"),
    make_option(c("-v", "--verbose"), default=F, help="Verbosity (default False)"),
    make_option(c("-p", "--pval_threshold"), default=0.1, help="Filter out TFs with adj. p-val above the provided value. Default = 0.1"),
    make_option(c("-e", "--export_carnival"), default=F, help="Export a table with the results with two columns (id, value) only (for CARNIVAL). Default = F")
  ),
  add_help_option = T,
  prog = "Differential expression data processing (Building Block)",
  formatter = IndentedHelpFormatter
)

arguments <- parse_args(parser, positional_arguments = T)
verbose <- arguments$options$verbose


if (arguments$options$tsv) {
  df <- read_tsv(arguments$args[1])  
} else {
  df <- read_csv(arguments$args[1])
}

DE_matrix <- df %>% 
  dplyr::select(arguments$options$id_col, arguments$options$weight_col) %>% 
  dplyr::filter(!is.na(arguments$options$weight_col)) %>% 
  column_to_rownames(var = arguments$options$id_col) %>%
  as.matrix()

dorothea::dorothea_hs
data(dorothea_hs, package = "dorothea")
regulons <- dorothea_hs %>%
  dplyr::filter(confidence %in% unlist(strsplit(arguments$options$confidence,","))) %>%
  dplyr::mutate(likelihood = 1)

#network <- intersect_regulons(DE_matrix, regulons, tf, target, minsize = 5)
#network$likelihood <- 1

tf_activity <- decoupleR::run_viper(
  DE_matrix, 
  regulons, 
  .source=arguments$options$source,
  minsize = arguments$options$minsize,
  verbose = verbose,
  eset.filter = FALSE
) %>% 
dplyr::mutate(adj_p = p.adjust(p_value, method = "BH")) %>%
filter(adj_p <= arguments$options$pval_threshold)


if (arguments$options$export_carnival){
    tf_activity <- tf_activity %>% select(source, score) %>% rename(id = source, value = score)
}


if (verbose) {
  sprintf("Exporting to %s...", arguments$args[2])
}

write_csv(tf_activity, arguments$args[2])
