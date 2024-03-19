#!/usr/bin/env Rscript --vanilla

library(CARNIVAL)
library(optparse)
library(readr)

parser <- OptionParser(
  usage = "usage: %prog solver_name solver_path sif_file measurements_csv perturbations_csv [options]",
  option_list = list(
    make_option(c("-p", "--penalty"), default=1e-4, help="Sparsity penalty. Default = 1e-4"),
    make_option(c("-m", "--mipgap"), default=0.05, help="MIP Gap relative tolerance for optimality. Default = 0.05"),
    make_option(c("-t", "--timelimit"), default=3600, help="Time limit in seconds. Default = 3600"),
    make_option(c("-v", "--verbose"), default=F, help="Verbosity (default False)"),
    make_option(c("-d", "--output_dir"), default='/tmp', help="Output directory. Default = /tmp"),
    make_option(c("-a", "--export_attributes"), default='carnival_attributes.csv', help="Export node attributes to a file. Default = carnival_attributes.csv"),
    make_option(c("-s", "--export_weighted_sif"), default='carnival_wsif.csv', help="Export weighted SIF with the summary of all interactions across networks. Default = carnival_wsif.csv"),
    make_option(c("-i", "--solver"), default=F, help="Assume file is TSV instead of CSV. Default = FALSE")
  ),
  add_help_option = T,
  prog = "Run CARNIVAL(R)",
  formatter = IndentedHelpFormatter
)

arguments <- parse_args(parser, positional_arguments = T)
verbose <- arguments$options$verbose

sif <- read_csv(arguments$args[3])
df_measurements <- read_csv(arguments$args[4], col_types = list(id = col_character(), value = col_double()))
# Check if provided
print(length(arguments$args))
if (length(arguments$args) >= 5) {
    df_inputs <- read_csv(arguments$args[5], col_types = list(id = col_character(), value = col_double()))
    inputs <- df_inputs$value
    names(inputs) <- df_inputs$id
} else {
    cat("Using INVERSE Carnival")
    inputs <- NULL
}

m <- df_measurements$value
names(m) <- df_measurements$id


result = runCARNIVAL(solver = arguments$args[1],
                     solverPath = arguments$args[2],
                     netObj = sif,
                     measObj = m,
                     inputObj = inputs,
                     betaWeight = arguments$options$penalty,
                     timelimit = arguments$options$timelimit,
                     mipGAP =  arguments$options$mipgap,
                     dir_name = arguments$options$output_dir)

#print(result)
write_csv(result$nodesAttributes, arguments$options$export_attributes)
write_csv(result$weightedSIF, arguments$options$export_weighted_sif)




