module load singularity
module unload impi
#module load intel/2019.5
#module load intel/2017.4
module load gcc/11.2.0_binutils
module load openmpi/4.1.2
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64/:\$LD_LIBRARY_PATH
export SINGULARITY_BIND=/etc/libibverbs.d/,/lib64/libssl.so.1.0.0,/lib64/libcrypto.so.1.0.0,/usr/lib64/libnuma.so.1
export SINGULARITYENV_CONTACT_NAMES=$CONTACT_NAMES
