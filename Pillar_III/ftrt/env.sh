. /etc/profile.d/z10_spack_environment.sh
export PTF_HYSEA_DIR=/opt/view/
export PTF_INSTALL_DIR=/gpfs/projects/bsc44/PTF_WF_container
export PYTHONPATH=$PTF_INSTALL_DIR/Code:$PTF_INSTALL_DIR/Code/Commons/py:$PYTHONPATH
export PTF_GPUS_NODE=4
export PTF_GPUS_EXEC=4
