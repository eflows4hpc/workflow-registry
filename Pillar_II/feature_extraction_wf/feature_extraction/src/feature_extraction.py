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
from PyOphidia import cube, client
import tensorflow as tf
import tensorflow.keras as keras
import pickle
import joblib
import xesmf as xe
import modules
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
    # Set input files and paths
    maxBaseline = join(inputPath, 'CMCC-CM3','max_clim_avg.nc')
    minBaseline = join(inputPath, 'CMCC-CM3','min_clim_avg.nc')
    cmcc_cm3_dir = join(inputPath, 'CMCC-CM3') # CMCC-CM3 data directory
    ML_dir = join(inputPath, 'ML') # ML directory
    listfiles = []
    tstormsfiles = []
    
    # Variables for inference phase
    driver_vars = ['WSPDSRFMX','PSL','T500','T300']  #CMCC-CM3 drivers to be used in inference
    grid_res = 0.25 # grid resolution of dataset
    batch_size = 648 # size of a single batch
    label_no_cyclone = -1. # label for no cyclones
    is_ckpt = True # whether or not load best model (checkpoint) or the last obtained (not necessarily the best)
    # Start counting time...
    st = time.time()

    # CONNECTION TO THE OPHIDIA SERVER
    # Set connection to Ophidia server
    cube.Cube.setclient(username="oph-test",password="abcd",server="s03r1b09",port="11732")
    #cube.Cube.setclient(read_env=True, project="0459")
    cube.Cube.createcontainer(container="Test",dim='lat|lon|time',dim_type='double|double|double',hierarchy='oph_base|oph_base|oph_time')
    # Create streams
    #REAL CASE: fds = FileDistroStream(base_dir=cmcc_cm3_dir)
    fds = FileDistroStream(base_dir=cmcc_cm3_dir)
    fds1 = FileDistroStream(base_dir=join(inputPath, 'TSTORMS', 'data'))
    # CLIMATOLOGICAL MEAN COMPUTATION FOR TSMX
    print("[LOG] CLIMATOLOGICAL MEAN IMPORT")
    if(compss_file_exists(maxBaseline)):
        print("Exists")
        clim_avg_tsmx = OphImportClimAvg(cube.Cube.client, cmcc_cm3_dir, 'max_clim_avg.nc', 'TSMX')
    else:
        maxdatacubes = [0 for i in range(NYEARS)]
        for i in range(NYEARS):
            maxdatacubes[i] = OphImportForBaseline(cube.Cube.client, inputPath, str(i+12).zfill(4), 'TSMX', "'OPH_MAX'")
        clim_avg_tsmx = OphClimAvg(cube.Cube.client, inputPath, maxdatacubes, 'max_clim_avg')
        OphDelete(cube.Cube.client, maxdatacubes,clim_avg_tsmx)
    
    # CLIMATOLOGICAL MEAN COMPUTATION FOR TSMN
    if(compss_file_exists(minBaseline)):
        print("Exists")
        clim_avg_tsmn = OphImportClimAvg(cube.Cube.client, cmcc_cm3_dir, 'min_clim_avg.nc', 'TSMN')
    else:
        mindatacubes = [0 for i in range(NYEARS)]
        for i in range(NYEARS):
            mindatacubes[i] = OphImportForBaseline(cube.Cube.client, inputPath, str(i+12).zfill(4), 'TSMN', "'OPH_MIN'")
        clim_avg_tsmn = OphClimAvg(cube.Cube.client, inputPath, mindatacubes, 'min_clim_avg')
        OphDelete(cube.Cube.client, mindatacubes,clim_avg_tsmn)
    # MAIN BLOCK FOR EACH YEAR OF DATA
    while True:
        listfiles = read_files(fds)
        print("[LOG] STREAM OF CMCC-CM3 SIMULATION FILES")
        # Sync and print value
        listfiles = compss_wait_on(listfiles)
        print("[LOG] PROCESSED CMCC-CM3 FILES: " + str(listfiles))
        #REAL CASE: Change with real CMCC-CM3 files
        #filenames = listfiles
        filenames = sorted([join(cmcc_cm3_dir, f) for f in listdir(cmcc_cm3_dir) if f.endswith('01-00000.nc')])
        # ML variables
        remapped_files = [0 for i in range(len(filenames))]
        dataset_patches_list = [0 for i in range(len(filenames))]
        y_pred_list = [0 for i in range(len(filenames))]
        cyclone_latlontime_coords_list = [0 for i in range(len(filenames))]
        # TSTORMS variables
        year = int(listfiles[0].split('.')[3].split('-')[0]) 
        monthlyfiles = [0 for i in range(12)]
        # HEAT/COLD WAVES INDICATORS COMPUTATION
        maxcubes = [OphImport(cube.Cube.client, inputPath, listfiles, 'TSMX', "'OPH_MAX'"), clim_avg_tsmx]
        mincubes = [OphImport(cube.Cube.client, inputPath, listfiles, 'TSMN', "'OPH_MIN'"), clim_avg_tsmn]
        hwdi = OphCompare(cube.Cube.client, maxcubes, "oph_predicate('OPH_FLOAT','OPH_FLOAT',measure,'x-100','>0','0','x')", "'>0'")
        cwdi = OphCompare(cube.Cube.client, mincubes, "measure", "'<0'")
        resultmaxcubes = [IndexDurationMax(cube.Cube.client, outputPath, hwdi, 'HWD'), IndexDurationNumber(cube.Cube.client, outputPath, hwdi, 'HWN'), IndexDurationFrequency(cube.Cube.client, outputPath, hwdi, 'HWF')]
        resultmincubes = [IndexDurationMax(cube.Cube.client, outputPath, cwdi, 'CWD'), IndexDurationNumber(cube.Cube.client, outputPath, cwdi, 'CWN'), IndexDurationFrequency(cube.Cube.client, outputPath, cwdi, 'CWF')]       
        # ML INFERENCE FOR TC DETECTION
        for idx_filename, filename in enumerate(filenames):
            # regridding
            remapped_files[idx_filename] = regridding(filename, driver_vars,inputPath)
            # get the dataset patches
            dataset_patches_list[idx_filename] = get_dataset_patches(remapped_files[idx_filename], driver_vars, PATCH_SIZE)
            # predict with the model
            y_pred_list[idx_filename] = predict(dataset_patches_list[idx_filename], is_ckpt, ML_dir, batch_size, PATCH_SIZE, label_no_cyclone)
            # retrieve cyclone latlon coordinates from the predictions
            cyclone_latlontime_coords_list[idx_filename] = retrieve_predicted_tc(outputPath, dataset_patches_list[idx_filename], y_pred_list[idx_filename], PATCH_SIZE) 
        TC_output=CreatePrediction(cyclone_latlontime_coords_list, outputPath)
        # TSTORMS EXECUTION FOR TC DETECTION
        # Creation of monthly files starting from daily files in the year
        for i in range(0,12):
            #REAL CASE: dailyfiles = glob.glob(join(inputPath, "CMCC-CM3/2000_cam6-nemo4_025deg_tc.cam.h1." + str(year).zfill(4) + "-" + str(i).zfill(2) + "-*.nc"))
            dailyfiles = glob.glob(join(cmcc_cm3_dir, "2000_cam6-nemo4_025deg_tc.cam.h1.0030-" + str(i+1).zfill(2) + "-*.nc"))
            #REAL CASE: monthlyfile = join(inputPath, "2000_cam6-nemo4_025deg_tc.cam.h1." + str(year).zfill(4) + "-" + str(i).zfill(2) + ".nc")
            monthlyfile = join(inputPath, "TSTORMS/data/2000_cam6-nemo4_025deg_tc.cam.h1.0030-" + str(i+1).zfill(2) + ".nc")
            # Creation of monthly files starting from daily files in the year
            monthlyfiles[i] = MergeFiles(inputPath, dailyfiles, monthlyfile)
        RunTSTORMSDriver(join(inputPath, "TSTORMS"), monthlyfiles)
        print("[LOG] TSTORMS EXECUTION")
        tstormsfiles = read_monthlyfiles(fds1,year)
        # Sync and print value
        tstormsfiles = compss_wait_on(tstormsfiles)
        print("[LOG] PROCESSED TSTORMS FILES: " + str(tstormsfiles))
        tstorms_result = RunTrajAnalysis(str(tstormsfiles), join(inputPath, "TSTORMS"), tstormsfiles)

        # SYNCHRONIZE RESULTS
        results = [resultmaxcubes, resultmincubes, tstorms_result, TC_output]
        results = compss_wait_on(results)
        #REAL CASE: remove break
        break
    fds.close()
    #fds1.close()
    elapsed = time.time() - st
    print("Elapsed time (s): {}".format(elapsed))

if __name__ == "__main__":
    main_program()
