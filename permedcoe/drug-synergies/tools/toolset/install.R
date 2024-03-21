#!/usr/bin/env Rscript --vanilla
#r = getOption("repos")
#r["CRAN"] = "http://cran.us.r-project.org"
#options(repos = r)

# Directory for installing packages
libpath <- .libPaths()[1]

# Install devtools and load it
install.packages("devtools", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
library("devtools")

# Install Bioconductor
# BiocManager::install(lib = libpath, ask = FALSE)  # Updates seurat and we dont want to.

# Install required packages using install_version from devtools
install_version("remotes", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install_version("igraph", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install_version("stringi", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
BiocManager::install("tidyverse", lib = libpath, update = FALSE)
BiocManager::install("OmnipathR", lib = libpath, update = FALSE)
BiocManager::install("progeny", lib = libpath, update = FALSE)
BiocManager::install("dorothea", lib = libpath, update = FALSE)
BiocManager::install("decoupleR", lib = libpath, update = FALSE)
BiocManager::install("cosmosR", lib = libpath, update = FALSE)
BiocManager::install("optparse", lib = libpath, update = FALSE)
BiocManager::install("CellNOptR", lib = libpath, update = FALSE)
BiocManager::install("rhdf5", lib = libpath, update = FALSE)
remotes::install_github("saezlab/CARNIVAL", ref="963fbc1db2d038bfeab76abe792416908327c176")
