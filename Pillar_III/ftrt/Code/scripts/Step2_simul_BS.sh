#!/bin/bash -e
ID=$(echo "1+$1" | bc)
eq_Mag=$2
eq_Lon=$3
eq_Lat=$4
eq_Top=$5
eq_Stk=$6
eq_Dip=$7
eq_Rak=$8
eq_A=$9
eq_L=${10}
eq_Slip=${11}
template_grid_file=${12}
prophours=${13}
simtime=$(echo "2+$prophours*3600" | bc)   #8h=28800sec
nd=${14}
template_ts_file=${15}
template_sim_file=${16}

savets=true #saving time series at POIs? if true, the ts.dat file must be provided
#w_Depth=0 #Kajiura?

#DIRNAME=$ID
#DIRNAME=BS_scenario$(echo $ID | awk -F. '{print $1}')
DIRNAME=BS_scenario_$(echo $ID | awk -F. -v num=$nd '{ printf("%0*i\n", num,$1) }')
SIMDIR=Step2_BS/$DIRNAME

if [ -d $SIMDIR ]; then
    echo $DIRNAME ': this scenario folder already exists'
else
    mkdir $SIMDIR

    echo ' creating parfile '
    ##HySEA simulation input file##
    eq_W=$(awk "BEGIN {printf \"%.3f\n\", $eq_A/$eq_L}")
    if [ $(echo "$eq_Top == 1" | bc ) -eq 1 ]; then
        eq_Top=0.    #correction for okada
    fi
    eq_Dep=$(awk "BEGIN {printf \"%.3f\n\", $eq_Top+$eq_W/2*sin($eq_Dip*3.141592654/180.)}")
    echo ' creating parfile for ' $SIMDIR
    sed s#bathyfile#${template_grid_file}# < ${template_sim_file} > $SIMDIR/parfile.txt
    sed -i s#poistsfile#${template_ts_file}#  $SIMDIR/parfile.txt
    sed -i s#idscen#$DIRNAME#  $SIMDIR/parfile.txt
    sed -i s#simdir#$SIMDIR# $SIMDIR/parfile.txt
    sed -i s/eqlon/$eq_Lon/ $SIMDIR/parfile.txt
    sed -i s/eqlat/$eq_Lat/  $SIMDIR/parfile.txt
    sed -i s/eqdep/$eq_Dep/ $SIMDIR/parfile.txt
    sed -i s/eqlen/$eq_L/  $SIMDIR/parfile.txt
    sed -i s/eqwid/$eq_W/  $SIMDIR/parfile.txt
    sed -i s/eqstk/$eq_Stk/  $SIMDIR/parfile.txt
    sed -i s/eqdip/$eq_Dip/  $SIMDIR/parfile.txt
    sed -i s/eqrak/$eq_Rak/  $SIMDIR/parfile.txt
    sed -i s/eqslip/$eq_Slip/  $SIMDIR/parfile.txt
    #if [ $(echo "$w_Depth >= 0" | bc ) -eq 1 ]; then
	#	sed -i  s/KFLAG/0/  $SIMDIR/parfile.txt
	#	sed -i '5d' $SIMDIR/parfile.txt
    #else
	#	sed -i  s/KFLAG/1/ $SIMDIR/parfile.txt
	#	wdep=$(echo $w_Depth | awk '{print  -$1}')
	#	sed -i  s/wdep/$wdep/ $SIMDIR/parfile.txt
    #fi
    sed -i s/simtime/$simtime/  $SIMDIR/parfile.txt
    if $savets; then
		sed -i  s/TSFLAG/1/  $SIMDIR/parfile.txt
    else
	        sed -i '/TSFLAG/{n;N;d}' $SIMDIR/parfile.txt
	        sed -i  s/TSFLAG/0/  $SIMDIR/parfile.txt
    fi
fi
