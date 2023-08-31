#/usr/bin/env python

# -*- coding: utf-8 -*-

"""
PyCOMPSs Feature Extraction workflow
==================
    This Python script computes the steps for the Feature Extraction Workflow in PyCOMPSs.
    (TEST with a CMCC-CM3 model that produces five days)
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
from PyOphidia import cube, client
import tensorflow as tf
import tensorflow.keras as keras
import pickle
import joblib

#import modules
from CMCC_CM3_tasks import *
from Ophidia_tasks import *
from TSTORMS_tasks import *
from ML_tasks import *
from lib import *
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import *
from pycompss.streams.distro_stream import FileDistroStream

# Constant values
NYEARS = 20
PATCH_SIZE = 40

def main_program():
    # Get the input path
    inputPath = sys.argv[1]
    # Get the output path
    outputPath = sys.argv[2]
    # Set input files
    maxBaseline = join(inputPath, 'max_clim_avg.nc')
    minBaseline = join(inputPath, 'min_clim_avg.nc')
    # whether or not load best model (checkpoint) or the last obtained (not necessarily the best)
    is_ckpt = True
    # Start counting time...
    st = time.time()
    # Declare variables
    listfiles = []
    tstormsfiles = []
    maxdatacubes = [0 for i in range(NYEARS)]
    mindatacubes = [0 for i in range(NYEARS)]
    # Variables for inference phase
    cmcc_cm3_dir = join(inputPath, 'CMCC-CM3') # CMCC-CM3 data directory
    cmcc_cm3_remap_dir = join(cmcc_cm3_dir, 'remapped')
    grid_file = join(inputPath, 'grid.txt')
    model_dir = inputPath
    driver_vars = ['WSPDSRFMX','PSL','T500','T300']  #CMCC-CM3 drivers to be used in inference
    grid_res = 0.25 # grid resolution of dataset
    batch_size = 2048 # size of a single batch
    label_no_cyclone = -1. # label for no cyclones

    # Connection to the Ophidia server
    cube.Cube.setclient(read_env=True, project="0459")
    cube.Cube.createcontainer(container="Test",dim='lat|lon|time',dim_type='double|double|double',hierarchy='oph_base|oph_base|oph_time')
    # Create streams
    fds = FileDistroStream(base_dir=join(inputPath, 'CMCC-CM3'))
    fds1 = FileDistroStream(base_dir=join(inputPath, 'TSTORMS'))
    # Computation of the climatological mean for the TSMX variable
    print("[LOG] CLIMATOLOGICAL MEAN IMPORT")
    if(compss_file_exists(maxBaseline)):
        print("Exists")
        clim_avg_tsmx = OphImportClimAvg(cube.Cube.client, inputPath, 'max_clim_avg.nc', 'TSMX')
    else:
        for i in range(NYEARS):
            maxdatacubes[i] = OphImportForBaseline(cube.Cube.client, inputPath, str(i+12).zfill(4), 'TSMX', "'OPH_MAX'")
        clim_avg_tsmx = OphClimAvg(cube.Cube.client, inputPath, maxdatacubes, 'max_clim_avg')
        OphDelete(cube.Cube.client, maxdatacubes,clim_avg_tsmx)
    # Computation of the climatological mean for the TSMN variable
    if(compss_file_exists(minBaseline)):
        print("Exists")
        clim_avg_tsmn = OphImportClimAvg(cube.Cube.client, inputPath, 'min_clim_avg.nc', 'TSMN')
    else:
        for i in range(NYEARS):
            mindatacubes[i] = OphImportForBaseline(cube.Cube.client, inputPath, str(i+12).zfill(4), 'TSMN', "'OPH_MIN'")
        clim_avg_tsmn = OphClimAvg(cube.Cube.client, inputPath, mindatacubes, 'min_clim_avg')
        OphDelete(cube.Cube.client, mindatacubes,clim_avg_tsmn)

    while True:
        listfiles = read_files(fds)
        print("[LOG] STREAM OF CMCC-CM3 SIMULATION FILES")
        # Sync and print value
        listfiles = compss_wait_on(listfiles)
        #REAL CASE: Change with real CMCC-CM3 files
        #filenames = listfiles
        filenames = sorted([join(cmcc_cm3_dir, f) for f in listdir(cmcc_cm3_dir) if f.endswith('.nc')])
        print("[LOG] PROCESSED CMCC-CM3 FILES: " + str(listfiles))
        # ML variables
        remapped_files = [0 for i in range(len(filenames))]
        #dataset patches result
        dataset_patches_list = [0 for i in range(len(filenames))]
        y_pred_list = [0 for i in range(len(filenames))]
        cyclone_latlon_coords_list = [0 for i in range(len(filenames))]
        # TSTORMS variables
        year = int(listfiles[0].split('.')[3].split('-')[0]) 
        # Import of five days into Ophidia
        maxcubes = [OphImport(cube.Cube.client, inputPath, listfiles, 'TSMX', "'OPH_MAX'"), clim_avg_tsmx]
        mincubes = [OphImport(cube.Cube.client, inputPath, listfiles, 'TSMN', "'OPH_MIN'"), clim_avg_tsmn]
        hwdi = OphCompare(cube.Cube.client, maxcubes, "oph_predicate('OPH_FLOAT','OPH_FLOAT',measure,'x-100','>0','0','x')", "'>0'")
        cwdi = OphCompare(cube.Cube.client, mincubes, "measure", "'<0'")
        #Computation of the indicators
        resultmaxcubes = [IndexDurationMax(cube.Cube.client, outputPath, hwdi, 'HWD'), IndexDurationNumber(cube.Cube.client, outputPath, hwdi, 'HWN'), IndexDurationFrequency(cube.Cube.client, outputPath, hwdi, 'HWF')]
        resultmincubes = [IndexDurationMax(cube.Cube.client, outputPath, cwdi, 'CWD'), IndexDurationNumber(cube.Cube.client, outputPath, cwdi, 'CWN'), IndexDurationFrequency(cube.Cube.client, outputPath, cwdi, 'CWF')]
        # ML inference
        for idx_filename, filename in enumerate(filenames):
            # regridding
            remapped_files[idx_filename] = regridding(filename=filename, vars=driver_vars, cmcc_cm3_remap_dir=cmcc_cm3_remap_dir, grid_file=grid_file)
            # get the dataset patches
            dataset_patches_list[idx_filename] = get_dataset_patches(filename=remapped_files[idx_filename], vars=driver_vars, patch_size=PATCH_SIZE)
            # predict with the model
            y_pred_list[idx_filename] = predict(X=dataset_patches_list[idx_filename], is_ckpt=is_ckpt, model_dir=model_dir, batch_size=batch_size, patch_size=PATCH_SIZE, label_no_cyclone=label_no_cyclone)
            # retrieve cyclone latlon coordinates from the predictions
            cyclone_latlon_coords_list[idx_filename] = retrieve_predicted_tc(outputPath=outputPath, dataset_patches=dataset_patches_list[idx_filename], y_pred=y_pred_list[idx_filename], patch_size=PATCH_SIZE) 
        # TSTORMS execution
        RunTSTORMSDriver(str(listfiles), inputPath, year, listfiles)
        print("[LOG] TSTORMS EXECUTION")
        tstormsfiles = read_monthlyfiles(fds1,year)
        # Sync and print value
        tstormsfiles = compss_wait_on(tstormsfiles)
        print("[LOG] PROCESSED TSTORMS FILES: " + str(tstormsfiles))
        tstorms_result = RunTrajAnalysis(str(tstormsfiles), inputPath, tstormsfiles)
        results = [resultmaxcubes, resultmincubes, tstorms_result, cyclone_latlon_coords_list]
        results = compss_wait_on(results)
        #REAL CASE: remove break
        break
    fds.close()
    fds1.close()
    elapsed = time.time() - st
    print("Elapsed time (s): {}".format(elapsed))

if __name__ == "__main__":
    main_program()
