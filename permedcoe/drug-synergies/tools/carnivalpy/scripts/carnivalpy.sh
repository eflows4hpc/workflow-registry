#!/usr/bin/env bash

data_dir=$1
solver_name=$2
penalty=$3
tol=$4
maxtime=$5
outfile=$6
outfile2=$7

tmp_dir=$(mktemp -d)
echo "Copying data to ${tmp_dir}..."
cp -R ${data_dir}/* ${tmp_dir}/
python /opt/carnival/carnivalpy/carnival.py $tmp_dir --solver $solver_name --penalty $penalty --maxtime $maxtime --tol $tol --export ${tmp_dir}/solution.csv

if [ -f "${tmp_dir}/network_with_perturbations.csv" ]; then
    mv ${tmp_dir}/network.csv ${tmp_dir}/old_network.csv
    mv ${tmp_dir}/network_with_perturbations.csv ${tmp_dir}/network.csv
fi


Rscript --vanilla /opt/export.R $tmp_dir $outfile $outfile2

rm -rf ${tmp_dir}
