#!/bin/bash -e

config_temp=$1
config_file=$2
data_path=$3 
step2_path=$4
par_file=$5
val_kagan=$6
val_mare=$7
val_event=$8
val_user_pois=$9

echo "config_temp: $config_temp"
echo "config_file: $config_file"
echo "data_path: $data_path"
echo "step2_path: $step2_path"
echo "par_file: $par_file"
echo "val_kagan: $val_kagan"
echo "val_mare: $val_mare"
echo "val_event: $val_event"
echo "val_user_pois: $val_user_poi"

# Retreiving values from the parfile

val_OR_EM=$(awk '/^val_OR_EM/{print $NF}' $par_file)
val_OR_HC=$(awk '/^val_OR_HC/{print $NF}' $par_file)
val_MC_type=$(awk '/^val_MC_type/{print $NF}' $par_file)
val_MC_samp_scen=$(awk '/^val_MC_samp_scen/{print $NF}' $par_file)
val_MC_samp_run=$(awk '/^val_MC_samp_run/{print $NF}' $par_file)
val_RS_type=$(awk '/^val_RS_type/{print $NF}' $par_file)
val_RS_samp_scen=$(awk '/^val_RS_samp_scen/{print $NF}' $par_file)
val_RS_samp_run=$(awk '/^val_RS_samp_run/{print $NF}' $par_file)
#val_kagan=$(awk '/^val_kagan/{print $NF}' $par_file)
#val_mare=$(awk '/^val_mare/{print $NF}' $par_file)
val_fm=$(awk '/^val_fm/{print $NF}' $par_file)
val_nsigma=$(awk '/^val_nsigma/{print $NF}' $par_file)
val_percentiles=$(awk '/^val_percentiles/{print $NF}' $par_file)

# # Passing values to the cfg file
# 
# #sed -i s/localpath/$localpath/g Step1_config_template.txt > $localpath/$CFGDIR/ptf_main.config
# #sed s#bathyfile#${template_grid_file}# < ${template_sim_file} > $SIMDIR/parfile.txt
# #sed s#localpath#${localpath}# < $localpath/$SCRDIR/Step1_config_template.txt > $localpath/$CFGDIR/ptf_main.config
# #sed s/val_event/$val_event/ $localpath/$CFGDIR/ptf_main.config > $localpath/$CFGDIR/ptf_main.config
# 
# #cp $localpath/$CFGDIR/ptf_main.config $localpath/$CFGDIR/ptf_main.config
# #sed -i s/val_event/$val_event/ $localpath/$CFGDIR/ptf_main.config
sed "s/val_event/$val_event/" $config_temp > $config_file
sed -i "s/val_OR_EM/$val_OR_EM/" $config_file
sed -i "s/val_OR_HC/$val_OR_HC/" $config_file
sed -i "s/val_MC_type/$val_MC_type/" $config_file 
sed -i "s/val_MC_samp_scen/$val_MC_samp_scen/" $config_file
sed -i "s/val_MC_samp_run/$val_MC_samp_run/" $config_file
sed -i "s/val_RS_type/$val_RS_type/" $config_file
sed -i "s/val_RS_samp_scen/$val_RS_samp_scen/" $config_file
sed -i "s/val_RS_samp_run/$val_RS_samp_run/" $config_file
sed -i "s/val_kagan/$val_kagan/" $config_file
sed -i "s/val_mare/$val_mare/" $config_file
sed -i "s/val_fm/$val_fm/" $config_file
sed -i "s/val_nSigma/$val_nsigma/" $config_file
sed -i "s/val_percentiles/$val_percentiles/" $config_file
sed -i "s#datapath#$data_path#" $config_file
sed -i "s#step2path#$step2_path#" $config_file
sed -i "s#val_user_pois#$val_user_pois#" $config_file

echo "Config ended"

