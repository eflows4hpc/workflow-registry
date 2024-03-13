. /etc/bashrc
module load singularity
module load openmpi/4.1.2-gcc-11.2.0 
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64/:\$LD_LIBRARY_PATH
#export SINGULARITY_BIND=/etc/libibverbs.d/,/lib64/libssl.so.1.0.0,/lib64/libcrypto.so.1.0.0,/usr/lib64/libnuma.so.1
export SINGULARITYENV_CONTACT_NAMES=$CONTACT_NAMES
