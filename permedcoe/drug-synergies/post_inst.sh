#!/usr/bin/env bash

CURRENT_DIR=$(pwd)
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# cd ${SCRIPT_DIR}

echo "Starting post-install.sh"
echo "Folder: ${CURRENT_DIR}"

# Require to create a symbolic link since the PhysiBoSS BBs have this
# path hardcoded and do a copy to modify the configuration file
# before executing the binary
mkdir -p /usr/local/scm/COVID19/
ln -s /opt/view/physiboss/ /usr/local/scm/COVID19/PhysiCell
mkdir -p /usr/local/scm/
cp /opt/view/physiboss_invasion/src/myproj /opt/view/physiboss_invasion/bin/.
ln -s /opt/view/physiboss_invasion/ /usr/local/scm/Invasion_model_PhysiBoSS

# # Install R version higher than 4.0 - in particular this installs 4.3.3
# apt update
# # apt install -y libbz2-dev liblzma-dev libcurl4-openssl-dev libssl-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev libhdf5-dev libmagick++-dev libgsl-dev libglpk-dev libpq-dev libudunits2-dev libfftw3-dev libmysqlclient-dev default-libmysqlclient-dev libgdal-dev libproj-dev
# apt install software-properties-common dirmngr wget -y
# wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
# add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
# apt update
# apt install r-base r-base-dev -y
# apt install libmariadb-dev libmariadbclient-dev -y
# apt install libgdal-dev libproj-dev -y

# Rollback to R version 4.2.3  # EPS removed in versions 4.3.0+
# Version of Seurat and dependencies requirement (no EPS define)
ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && echo Etc/UTC > /etc/timezone
export DEBIAN_FRONTEND=noninteractive
apt install tzdata -y
export R_VERSION=4.2.3
curl -O https://cran.rstudio.com/src/base/R-4/R-${R_VERSION}.tar.gz
tar -xzvf R-${R_VERSION}.tar.gz
cd R-${R_VERSION}
./configure \
    --prefix=/opt/R/${R_VERSION} \
    --enable-R-shlib \
    --enable-memory-profiling \
    --with-blas \
    --with-lapack
make
make install
ln -s /opt/R/${R_VERSION}/bin/R /usr/local/bin/R
ln -s /opt/R/${R_VERSION}/bin/Rscript /usr/local/bin/Rscript

####################################################
#################### CarnivalPy ####################
####################################################

mkdir -p /opt/carnival/carnivalpy/
cp /covid19/tools/carnivalpy/carnivalpy/carnival.py /opt/carnival/carnivalpy/carnival.py
cp /covid19/tools/carnivalpy/carnivalpy/export.R /opt/carnival/carnivalpy/export.R
cp /covid19/tools/carnivalpy/scripts/carnivalpy.sh /opt/carnivalpy.sh
cp /covid19/tools/carnivalpy/scripts/export.R /opt/export.R
cp /covid19/tools/carnivalpy/scripts/feature_merge.py /usr/local/bin/feature_merger
chmod +x /usr/local/bin/feature_merger

####################################################
################## Meta-analysis ###################
####################################################

Rscript --vanilla  /covid19/tools/meta_analysis/install.R

####################################################
##################### ML-Jax #######################
####################################################

cp /covid19/tools/ml_jax/ml.py /opt/ml_jax.py
chmod +x /opt/ml_jax.py
ln -sf /opt/ml_jax.py /usr/local/bin/ml

####################################################
############### PhysiCell-Covid19 ##################
####################################################

Rscript --vanilla /covid19/tools/physicell_covid19/install.R

####################################################
################## Single Cell #####################
####################################################

Rscript --vanilla /covid19/tools/single_cell/install.R

####################################################
#################### Toolset #######################
####################################################

Rscript --vanilla /covid19/tools/toolset/install.R

cp /covid19/tools/toolset/scripts/tf_enrichment.R /opt/tf_enrichment.R
cp /covid19/tools/toolset/scripts/progeny.R /opt/progeny.R
cp /covid19/tools/toolset/scripts/preprocess.R /opt/preprocess.R
cp /covid19/tools/toolset/scripts/export.R /opt/export.R
cp /covid19/tools/toolset/scripts/normalize.R /opt/normalize.R
cp /covid19/tools/toolset/scripts/omnipath.R /opt/omnipath.R
cp /covid19/tools/toolset/scripts/export_carnival.R /opt/export_carnival.R
cp /covid19/tools/toolset/scripts/carnivalr.R /opt/carnivalr.R

cp /covid19/tools/toolset/scripts/tf_enrichment.R /usr/local/bin/tf_enrichment
cp /covid19/tools/toolset/scripts/progeny.R /usr/local/bin/progeny
cp /covid19/tools/toolset/scripts/preprocess.R /usr/local/bin/preprocess
cp /covid19/tools/toolset/scripts/export.R /usr/local/bin/export
cp /covid19/tools/toolset/scripts/normalize.R /usr/local/bin/normalize
cp /covid19/tools/toolset/scripts/omnipath.R /usr/local/bin/omnipath
cp /covid19/tools/toolset/scripts/export_carnival.R /usr/local/bin/export_carnival
cp /covid19/tools/toolset/scripts/carnivalr.R /usr/local/bin/carnivalr
chmod +x /usr/local/bin/*

####################################################

# cd ${CURRENT_DIR}
