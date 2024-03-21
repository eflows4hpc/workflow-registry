library(optparse)

parser <- OptionParser(
  usage = "usage: %prog folder_input rds_file csv_node_features",
  add_help_option = T,
  prog = "Export CARNIVALpy solution to Carnival R file",
  formatter = IndentedHelpFormatter
)

arguments <- parse_args(parser, positional_arguments = T)


source("/opt/carnival/carnivalpy/export.R")
sol <- export_folder(arguments$args[1])
df <- data.frame(sol$nodesAttributes$AvgAct/100, row.names=sol$nodesAttributes$Node)
write.csv(df, file=arguments$args[3])
saveRDS(sol, file=arguments$args[2])

