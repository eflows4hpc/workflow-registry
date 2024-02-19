#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
TSTORMS tasks definition
==================
    This file declares the TSTORMS tasks in PyCOMPSs.
"""

# Imports
import os
from os.path import join
import uuid
import time
from cdo import *
cdo = Cdo()
import subprocess

from pycompss.api.task import task
from pycompss.api.parameter import *

# Constant values
SLEEP = 5

# Task definitions

@task(dailyfiles=COLLECTION_IN, returns=str)
def MergeFiles(input_path, dailyfiles, monthlyfile):
    # Creation of monthly files starting from daily files in the year
    cdo.mergetime(input=dailyfiles, output=monthlyfile) 
    return monthlyfile   

@task(files=COLLECTION_IN)
def RunTSTORMSDriver(input_path, files):
    old=''
    with open(join(input_path, "nml_input_zeus"), "r+") as f:
        for i,line in enumerate(f):
            if i<27:
                old += line
        f.seek(0)
        f.write(old + ' ' + str(files)[1:-1].replace(',', '\n') + '\n' + '/')
    command = 'csh submit_tstorms_job.csh ' + input_path
    os.chdir(input_path)
    p = subprocess.run(command, capture_output=True, shell=True)

@task(fds=STREAM_IN, returns=list)
def read_monthlyfiles(fds,year):
    num_total = 0
    list_files = []
    #REAL CASE: remove testyear
    testyear = 30
    while num_total < 12:
        # Poll new files
        print("Polling files")
        new_files = fds.poll()
        # Process files
        for nf in new_files:
            print("RECEIVED FILE: " + str(nf))
            #REAL CASE: if str(nf).endswith('tstorm.nc') and int(str(nf).split('.')[3].split('-')[0]) == year:
            if str(nf).endswith('tstorm.nc') and int(str(nf).split('.')[3].split('-')[0]) == testyear:
                list_files.append(str(nf))
        # Accumulate read files
        num_total = len(list_files)
        # Sleep between requests
        time.sleep(SLEEP)
    # Return the number of processed files
    return list_files

@task(files=COLLECTION_IN, returns=int)
def RunTrajAnalysis(str_files, input_path, files):
    sorted_files = sorted(eval(str_files))

    #Append TSTORMS files in the nml_traj_zeus file
    old = ''

    with open(join(input_path, "nml_traj_zeus"), "r+") as f:
        x = len(f.readlines())

    with open(join(input_path, "nml_traj_zeus"), "r+") as f:
        for i,line in enumerate(f):
            if i<x-1:
                old += line
        f.seek(0)
        f.write(old + ' ' + str(sorted_files)[1:-1].replace("'/work", "   '/work").replace(',', ',\n') + ',\n' + '/')

    command = './trajectory_analysis_csc.exe < ' + join(input_path, 'nml_traj_zeus')
    os.chdir(input_path)
    p = subprocess.run(command, capture_output=True, shell=True)
    return 1
