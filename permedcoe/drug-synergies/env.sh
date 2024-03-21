old_PYTHONPATH=$PYTHONPATH
. /etc/profile.d/z10_spack_environment.sh
export PYTHONPATH=$PYTHONPATH:$old_PYTHONPATH
export COMPSS_HOME=/opt/view/compss
export LD_LIBRARY_PATH=$COMPSS_HOME/Bindings/bindings-common/lib/:$LD_LIBRARY_PATH
export PERMEDCOE_IMAGES=$(pwd)
