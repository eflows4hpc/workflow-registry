#!/usr/bin/env Rscript --vanilla
#r = getOption("repos")
#r["CRAN"] = "http://cran.us.r-project.org"
#options(repos = r)

# Directory for installing packages
libpath <- .libPaths()[1]

# Install devtools and load it
install.packages("devtools", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
library("devtools")

# Install required packages using install_version from devtools
install_version("dplyr", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install_version("tidyverse", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install_version("diptest", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install_version("mclust", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install_version("moments", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install_version("pheatmap", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install_version("optparse", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
