#!/usr/bin/env Rscript --vanilla

library(readr)
library(optparse)
library(tibble)
library(dplyr)

parser <- OptionParser(
  usage = "usage: %prog expression_csv_file_or_url output_file [options]",
  option_list = list(
    make_option(c("-c", "--col_genes"), default="GENE_SYMBOLS", help="Name of the column containing the gene symbols. Default = GENE_SYMBOLS"),
    make_option(c("-s", "--scale"), default=F, help="Scale the data. Default = TRUE"),
    make_option(c("-e", "--exclude_cols"), default="GENE_title", help="Exclude columns containing the given string. Default = 'GENE_title'"),
    make_option(c("-t", "--tsv"), default=F, help="Assume file is TSV instead of CSV. Default = FALSE"),
    make_option(c("-r", "--remove"), default="DATA.", help="Remove substring from columns"),
    make_option(c("-v", "--verbose"), default=F, help="Verbosity (default False)")
  ),
  add_help_option = T,
  prog = "Preprocess gene expression data",
  formatter = IndentedHelpFormatter
)

arguments <- parse_args(parser, positional_arguments = T)
verbose <- arguments$options$verbose
file <- arguments$args[1]

if (verbose) {
  sprintf("Loading expression data from %s...", file)
}

if (startsWith(file, "http") || startsWith(file, "www.")) {
  url <- file
  if (verbose) {
    print("Downloading file...")
  }
  file <- tempfile()
  download.file(url, file, mode = "wb")
}

if (arguments$options$tsv) {
  df <- read_tsv(file)  
} else {
  df <- read_csv(file)
}

df <- df[!is.na(df[arguments$options$col_genes]),]
df <- 
  df %>% 
  select(-contains(arguments$options$exclude_cols)) %>%
  rename_with(~gsub(arguments$options$remove, "", .x, fixed = TRUE)) %>%
  column_to_rownames(var = arguments$options$col_genes)

if (arguments$options$scale) {
    df <- (df - apply(df, 1, mean))/apply(df, 1, sd)
}

write_csv(df %>% rownames_to_column(var=arguments$options$col_genes), arguments$args[2])



