#!/usr/bin/env Rscript --vanilla

# Directory for installing packages
libpath <- .libPaths()[1]

install.packages('pacman', repos='http://cran.rstudio.com/', dependencies = TRUE, lib = libpath) # repos='http://cran.us.r-project.org')
install.packages('devtools', repos='http://cran.rstudio.com/', dependencies = TRUE, lib = libpath) # repos='http://cran.us.r-project.org')
list.p=c("dplyr","Matrix","future","pheatmap","ggplot2","optparse","hdf5r")
pacman::p_load(list.p, character.only = TRUE)
install.packages('rmatio', repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages('XML', repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
