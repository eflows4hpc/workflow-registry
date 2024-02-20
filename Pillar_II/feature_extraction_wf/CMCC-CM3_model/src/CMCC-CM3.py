#/usr/bin/env python

# -*- coding: utf-8 -*-

"""
PyCOMPSs Feature Extraction workflow
==================
    This Python script computes the steps for the Feature Extraction Workflow in PyCOMPSs.
"""

# Imports

import os
from os.path import join
import uuid
import glob
import time
import datetime
from datetime import date
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import pandas as pd
import modules
from CMCC_CM3_tasks import *
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import *
from pycompss.streams.distro_stream import FileDistroStream

def main_program():
    # Get the input path
    inputPath = sys.argv[1]
    # Get the output path
    outputPath = sys.argv[2]
    # Set input files and paths
    cmcc_cm3_dir = join(inputPath, 'CMCC-CM3') # CMCC-CM3 data directory
    test_model_dir = join(inputPath, 'CMCC-CM3/test/hist') # semplified model version
    listfiles = []
    
    # Start counting time...
    st = time.time()
    # MODEL SUBMISSION
    generatedfiles = len([entry for entry in os.listdir(test_model_dir) if os.path.isfile(os.path.join(test_model_dir, entry))])
    print("[LOG] THE NUMBER OF FILES IS: " + str(generatedfiles))
    if(generatedfiles < 5):
        print("[LOG] CMCC-CM3 SUBMISSION")
        RunCMCCModel(cmcc_cm3_dir)

    # Create stream
    #REAL CASE: fds = FileDistroStream(base_dir=cmcc_cm3_dir)
    fds = FileDistroStream(base_dir=test_model_dir)
    
    # MAIN BLOCK FOR EACH YEAR OF DATA
    while True:
        listfiles = read_files(fds)
        print("[LOG] STREAM OF CMCC-CM3 SIMULATION FILES")
        # Sync and print value
        listfiles = compss_wait_on(listfiles)
        print("[LOG] PROCESSED CMCC-CM3 FILES: " + str(listfiles))
        
        break
    fds.close()
    elapsed = time.time() - st
    print("Elapsed time (s): {}".format(elapsed))

if __name__ == "__main__":
    main_program()
