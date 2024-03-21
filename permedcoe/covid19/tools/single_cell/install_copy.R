#!/usr/bin/env Rscript --vanilla
#r = getOption("repos")
#r["CRAN"] = "http://cran.us.r-project.org"
#options(repos = r)

# Directory for installing packages
libpath <- .libPaths()[1]

# Install devtools and load it
install.packages("devtools", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
library("devtools")

devtools::install_github('immunogenomics/presto')

# Install Bioconductor
BiocManager::install(lib = libpath, ask = FALSE)

# Install other packages
install.packages("pacman", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("usethis", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install_version("RcppAnnoy", version = "0.0.16", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
BiocManager::install("BiocNeighbors", lib = libpath, update = FALSE)
install_version("RcppAnnoy", version = "0.0.18", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
BiocManager::install("SingleR", lib = libpath, update = FALSE)
BiocManager::install("limma", lib = libpath, update = FALSE)
BiocManager::install("SingleCellExperiment", lib = libpath, update = FALSE)
BiocManager::install("multtest", lib = libpath, update = FALSE)
BiocManager::install("Biobase", lib = libpath, update = FALSE)
install.packages("https://cran.r-project.org/src/contrib/Archive/slam/slam_0.1-49.tar.gz",
                 repos=NULL, type="source", INSTALL_opts = "--no-lock", dependencies = TRUE, lib = libpath)
install.packages("https://cran.r-project.org/src/contrib/Archive/sparsesvd/sparsesvd_0.2-1.tar.gz",
                 repos=NULL, type="source", INSTALL_opts = "--no-lock", dependencies = TRUE, lib = libpath)
install.packages("https://cran.r-project.org/src/contrib/Archive/docopt/docopt_0.7.tar.gz",
                 repos=NULL, type="source", INSTALL_opts = "--no-lock", dependencies = TRUE, lib = libpath)
install.packages("https://cran.r-project.org/src/contrib/Archive/qlcMatrix/qlcMatrix_0.9.7.tar.gz",
                 repos=NULL, type="source", INSTALL_opts = "--no-lock", dependencies = TRUE, lib = libpath)
BiocManager::install("monocle", lib = libpath, update = FALSE)
BiocManager::install("rtracklayer", lib = libpath, update = FALSE)
BiocManager::install("IRanges", lib = libpath, update = FALSE)
BiocManager::install("GenomeInfoDb", lib = libpath, update = FALSE)
BiocManager::install("GenomicRanges", lib = libpath, update = FALSE)
BiocManager::install("BiocGenerics", lib = libpath, update = FALSE)
BiocManager::install("DESeq2", lib = libpath, update = FALSE)
BiocManager::install("MAST", lib = libpath, update = FALSE)
BiocManager::install("SummarizedExperiment", lib = libpath, update = FALSE)
BiocManager::install("S4Vectors", lib = libpath, update = FALSE)
BiocManager::install("graph", lib = libpath, update = FALSE)

install_version(package = "rsvd", version = package_version("1.0.2"), repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
# install.packages("https://cran.r-project.org/src/contrib/Archive/spatstat/spatstat_1.64-1.tar.gz",
#                  repos=NULL, type="source", INSTALL_opts = "--no-lock", dependencies = TRUE, lib = libpath)
# Need to remove any updated spatstat and spatstat.core (with dependencies) packages to install specific version to work with seurat version
#remove.packages(grep("spatstat", installed.packages(), value = T))
#remove.packages(grep("spatstat.utils", installed.packages(), value = T))
#remove.packages(grep("spatstat.data", installed.packages(), value = T))
#remove.packages(grep("spatstat.geom", installed.packages(), value = T))
#remove.packages(grep("spatstat.random", installed.packages(), value = T))
#remove.packages(grep("spatstat.sparse", installed.packages(), value = T))
#remove.packages(grep("spatstat.core", installed.packages(), value = T))
# Newer version of spatstat since the 1.64-1 version is not working (also spatstat.core and its dependenciesfrom archive).
install.packages("https://cran.r-project.org/src/contrib/Archive/spatstat.utils/spatstat.utils_2.2-0.tar.gz",
                 repos=NULL, type="source", INSTALL_opts = "--no-lock", dependencies = TRUE, lib = libpath)
install.packages("https://cran.r-project.org/src/contrib/Archive/spatstat.data/spatstat.data_2.2-0.tar.gz",
                 repos=NULL, type="source", INSTALL_opts = "--no-lock", dependencies = TRUE, lib = libpath)
install.packages("polyclip", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("deldir", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("https://cran.r-project.org/src/contrib/Archive/spatstat.geom/spatstat.geom_2.4-0.tar.gz",
                 repos=NULL, type="source", INSTALL_opts = "--no-lock", dependencies = TRUE, lib = libpath)
install.packages("https://cran.r-project.org/src/contrib/Archive/spatstat.random/spatstat.random_2.2-0.tar.gz",
                 repos=NULL, type="source", INSTALL_opts = "--no-lock", dependencies = TRUE, lib = libpath)
install.packages("tensor", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("https://cran.r-project.org/src/contrib/Archive/spatstat.sparse/spatstat.sparse_2.1-1.tar.gz",
                 repos=NULL, type="source", INSTALL_opts = "--no-lock", dependencies = TRUE, lib = libpath)
install.packages("goftest", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("https://cran.r-project.org/src/contrib/Archive/spatstat.core/spatstat.core_2.4-0.tar.gz",
                 repos=NULL, type="source", INSTALL_opts = "--no-lock", dependencies = TRUE, lib = libpath)
install_version(package = "spatstat", version = package_version("2.2-0"), repos='http://cran.rstudio.org', dependencies = TRUE, lib = libpath)
# install_version(package = "Seurat", version = package_version("3.2.3"), repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath) # fails
install_version(package = "Seurat", version = package_version("4.0.1"), repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)

install.packages("dplyr", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("Matrix", repos='http://cran.us.r-project.org', version = package_version("1.6.3"), dependencies = TRUE, lib = libpath)
install.packages("future", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("pheatmap", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("ggplot2", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("optparse", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)
install.packages("hdf5r", repos='http://cran.us.r-project.org', dependencies = TRUE, lib = libpath)

#Warning messages:
#1: packages ‘multtest’, ‘limma’, ‘Biobase’, ‘monocle’, ‘rtracklayer’, ‘IRanges’, ‘GenomeInfoDb’, ‘GenomicRanges’, ‘BiocGenerics’, ‘DESeq2’, ‘MAST’, ‘SingleCellExperiment’, ‘SummarizedExperiment’, ‘S4Vectors’ are not available for this version of R
#
#Versions of these packages for your version of R might be available elsewhere,
#see the ideas at
#https://cran.r-project.org/doc/manuals/r-patched/R-admin.html#Installing-packages 
#2: In i.p(...) : installation of package ‘mutoss’ had non-zero exit status
#3: In i.p(...) : installation of package ‘metap’ had non-zero exit status

#‘monocle’, ‘rtracklayer’, ‘DESeq2’, ‘MAST’ not available for this version of R