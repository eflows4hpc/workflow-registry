#!/usr/bin/env python
import sys
import os
import subprocess
from cdo import *
from os.path import join
cdo = Cdo()
files=sys.argv[1]
inputPath=sys.argv[2]
year=sys.argv[3]
monthlyfiles = []
# Creation of monthly files starting from daily files in the year
#REAL CASE: Remove path_dailyfiles
path_dailyfiles = "/work/csp/dp16116/CMCC-CM3/2000_cam6-nemo4_025deg_tc/atm/hist/"
for i in range(1,13):
    #REAL CASE: dailyfiles=files
    dailyfiles = glob.glob(join(inputPath, "CMCC-CM3/2000_cam6-nemo4_025deg_tc.cam.h1.0030-" + str(i).zfill(2) + "-*.nc"))
    #REAL CASE: monthlyfile = join(inputPath, "2000_cam6-nemo4_025deg_tc.cam.h1." + str(year).zfill(4) + "-" + str(i).zfill(2) + ".nc")
    monthlyfile = join(inputPath, "TSTORMS/2000_cam6-nemo4_025deg_tc.cam.h1.0030-" + str(i).zfill(2) + ".nc")
    cdo.mergetime(input=dailyfiles, output=monthlyfile)
    monthlyfiles.append(monthlyfile)
old=''
with open(join(inputPath, "nml_input_zeus"), "r+") as f:
    for i,line in enumerate(f):
        if i<27:
            old += line
    f.seek(0)
    f.write(old + ' ' + str(monthlyfiles)[1:-1].replace(',', '\n') + '\n' + '/') 
command = 'csh submit_tstorms_job.csh ' + inputPath
os.chdir(inputPath)
p = subprocess.run(command, capture_output=True, shell=True)

