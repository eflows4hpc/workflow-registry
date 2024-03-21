#!/usr/bin/env Rscript --vanilla

library(readr)
library(dplyr)
library(tibble)
library(optparse)

parser <- OptionParser(
  usage = "usage: %prog expression_file output_file [options]",
  option_list = list(
    make_option(c("-i", "--col_genes"), default="GENE_SYMBOLS", help="Name of the column in the diff_expr file with the gene identifiers. Default = ID"),
    make_option(c("-v", "--verbose"), default=F, help="Verbosity (default False)"),
    make_option(c("-t", "--tsv"), default=F, help="Assume file is TSV instead of CSV. Default = FALSE"),
    make_option(c("-e", "--exclude_cols"), default="GENE_title", help="Exclude columns containing the given string. Default = 'GENE_title'")
  ),
  add_help_option = T,
  prog = "Standardize expression matrix",
  formatter = IndentedHelpFormatter
)

arguments <- parse_args(parser, positional_arguments = T)
verbose <- arguments$options$verbose

if (verbose) {
  sprintf("Loading data from %s...", arguments$args[1])
}

if (arguments$options$tsv) {
  df <- read_tsv(arguments$args[1])  
} else {
  df <- read_csv(arguments$args[1])
}

df <- df[!is.na(df[arguments$options$col_genes]),]

df_exprs <- 
  df %>%
  select(-contains(arguments$options$exclude_cols)) %>%
  column_to_rownames(var = arguments$options$col_genes)
  
df_exprs <- (df_exprs - apply(df_exprs, 1, mean))/apply(df_exprs, 1, sd)
write_csv(df_exprs %>% rownames_to_column(var=arguments$options$col_genes), arguments$args[2])
