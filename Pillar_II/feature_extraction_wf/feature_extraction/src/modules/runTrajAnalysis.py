#!/usr/bin/env python
import sys
import os
import subprocess
from os.path import join

files=sys.argv[1]
inputPath=sys.argv[2]
sorted_files = sorted(eval(files))

#Append TSTORMS files in the nml_traj_zeus file
old = ''

with open(join(inputPath, "nml_traj_zeus"), "r+") as f:
    x = len(f.readlines())
      
with open(join(inputPath, "nml_traj_zeus"), "r+") as f:
    for i,line in enumerate(f):
        if i<x-1:
            old += line
    f.seek(0)
    f.write(old + ' ' + str(sorted_files)[1:-1].replace("'/work", "   '/work").replace(',', ',\n') + ',' + '\n' + '/')

command = './trajectory_analysis_csc.exe <' + inputPath + 'nml_traj_zeus'
os.chdir(inputPath)
p = subprocess.run(command, capture_output=True, shell=True)

