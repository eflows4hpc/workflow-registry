export COMPSS_PYTHON_VERSION=3.9.10
module load COMPSs/3.3
module load singularity/3.7.3

num_nodes=1
queue=debug
time_limit=40
SINGULARITYENV_JAVA_TOOL_OPTIONS=-Xss1280k

data=/apps/COMPSs/PerMedCoE/resources/drug-synergies-workflow/Resources/data/
results=$(pwd)/results_uc2/

mkdir -p $results

pycompss job submit \
 --qos=$queue \
 -gd \
 --log_dir=$PWD \
 --exec_time=$time_limit \
 --container_image=$PWD/permedcoe_covid19_skylake_nompi_nogpu_v_latest.sif \
 --container_compss_path=/opt/view/compss --container_opts="-e" \
 --env_script=$(pwd)/env.sh \
 --num_nodes=$num_nodes \
 --worker_in_master_cpus=48 \
 --worker_working_dir=$(pwd)/working_dir/ \
 $(pwd)/drug-synergies/src/uc2.py \
    ${data}/Sub_genes.csv \
    ${data}/rnaseq_fpkm_20191101.csv \
    ${data}/mutations_20191101.csv \
    ${data}/cnv_gistic_20191101.csv \
    ${data}/genes_druggable.csv \
    ${data}/genes_target.csv \
    ${results}
