#!/usr/bin/env Rscript --vanilla
#r = getOption("repos")
#r["CRAN"] = "http://cran.us.r-project.org"
#options(repos = r)

# Directory for installing packages
libpath <- .libPaths()[1]

# Install devtools and load it
install.packages("devtools", repos='http://cran.rstudio.com/', dependencies = TRUE, lib = libpath)
library("devtools")
install.packages('remotes', repos='http://cran.rstudio.com/', dependencies = TRUE, lib = libpath)

# Install Bioconductor
BiocManager::install(lib = libpath, ask = FALSE)
BiocManager::install("multtest", lib = libpath, update = FALSE)

# Install other packages
install.packages("pacman", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("usethis", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install_version("RcppAnnoy", version = "0.0.16", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
BiocManager::install("BiocNeighbors", lib = libpath, update = FALSE)
install_version("RcppAnnoy", version = "0.0.18", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
BiocManager::install("SingleR", lib = libpath, update = FALSE)
BiocManager::install("limma", lib = libpath, update = FALSE)
BiocManager::install("SingleCellExperiment", lib = libpath, update = FALSE)

install_version(package = "rsvd", version = package_version("1.0.2"), repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
# install.packages('Seurat', version = "3.2.3", repos = c('https://satijalab.r-universe.dev', 'https://cloud.r-project.org'), dependencies = TRUE, lib = libpath)
# install.packages('spatstat', version = "1.64-1", repos = c('https://spatstat.r-universe.dev', 'https://cloud.r-project.org'), dependencies = TRUE, lib = libpath)
remotes::install_version(package = 'spatstat', version = package_version('1.64-1'), repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
remotes::install_version(package = 'Seurat', version = package_version('3.2.3'), repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath, upgrade ="never")



install.packages("dplyr", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install_version("Matrix", repos='http://cran.us.r-project.org', version = package_version("1.6-3"), dependencies = TRUE, lib = libpath)
install.packages("future", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("pheatmap", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("ggplot2", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("optparse", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("hdf5r", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
