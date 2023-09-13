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

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.binary import binary

# Constant values
SLEEP = 5

# Task definitions

@binary(binary=join(os.getcwd(), "modules/runTSTORMS.py"))
@task(files=COLLECTION_IN)
def RunTSTORMSDriver(str_files, input_path, year, files):
    pass

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

@binary(binary=join(os.getcwd(), "modules/runTrajAnalysis.py"))
@task(files=COLLECTION_IN, returns=int)
def RunTrajAnalysis(str_files, input_path, files):
    #pass
    return 1 
