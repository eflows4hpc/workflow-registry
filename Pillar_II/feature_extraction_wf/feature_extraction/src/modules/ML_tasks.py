#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
PyCOMPSs ML Tasks
==================
    This file declares the Machine Learning tasks in PyCOMPSs.
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
import tensorflow as tf
import tensorflow.keras as keras
import pickle
import joblib
#from cdo import *
#cdo = Cdo()
import xesmf as xe
from numpy import asarray
from numpy import savetxt
from pycompss.api.task import task
from pycompss.api.parameter import *

# custom library containing utility functions
from lib import load_model_and_scaler, TFScaler, from_local_to_global

REGRIDDER = None
MODEL = None
SCALER = None

# ML Tasks

@task(returns=object, vars=COLLECTION_IN)
def regridding(filename, vars, input_path):
    vars_string = ','.join(vars)
    global REGRIDDER
    if not REGRIDDER:
        
        fn = join(input_path,'ML','conservative.nc')
        ds = xr.open_dataset(join(input_path,'CMCC-CM3','2000_cam6-nemo4_025deg_tc.cam.h1.0030-12-31-00000.nc'))
        ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.flip(np.arange(-90.0, 90.25, 0.25))),
            "lon": (["lon"], np.arange(0.0, 360.0, 0.25)),
        })
        REGRIDDER = xe.Regridder(ds,ds_out,'conservative',weights=fn)
    ds = xr.open_dataset(filename)
    ds_out = REGRIDDER(ds)
    #ds_out = REGRIDDER(ds).drop_dims(['ilev','lev'])
    return ds_out

@task(returns=list, vars=COLLECTION_IN)
def get_dataset_patches(ds, vars, patch_size):
    # patch the dataset
    patch_ds = ds.coarsen({'lat':patch_size, 'lon':patch_size}, boundary="trim").construct({'lon':("cols", "lon_range"), 'lat':("rows", "lat_range")})

    # get the number of patches for rows and columns
    row_blocks = len(patch_ds['rows'])
    col_blocks = len(patch_ds['cols'])
    time_blocks = len(patch_ds['time'])
    times = patch_ds.time.values
    X = []
    patch_ds_reshaped = (patch_ds.stack(num_patches=['time','rows','cols'])).expand_dims(dim='vars').transpose('num_patches', 'lat_range', 'lon_range','vars')
    X = np.concatenate(([(patch_ds_reshaped[var]).values for var in vars]), axis=-1)
    del patch_ds_reshaped, patch_ds
    return [X, ds, time_blocks, row_blocks, col_blocks, times]


@task(returns=list, X=COLLECTION_IN)
def predict(X, is_ckpt, model_dir, batch_size, patch_size, label_no_cyclone):
    global MODEL,SCALER
    if not MODEL or not SCALER:
        MODEL, SCALER = load_model_and_scaler(model_dir, is_ckpt=is_ckpt)
    # scale the data
    X_scaled = SCALER.transform(X[0])
    # predict with the model
    y_pred = MODEL.predict(X_scaled, batch_size)
    # FILTER RESULTS (CONSIDER AS A BLACK BOX)
    # set to nan all negative coordinates
    y_pred = np.where(y_pred < 0.0, np.nan, (np.where(y_pred > patch_size-1, patch_size-1, y_pred)))
    # build a filter to remove spurious coordinates
    filter = np.repeat((y_pred >= 0.0).min(axis=1)[:,np.newaxis], repeats=2, axis=1)
    # get filtered predictions
    y_pred = np.where(filter > 0, y_pred, label_no_cyclone)
    # round coordinates
    y_pred = np.round(y_pred, 0)
    del filter, X_scaled
    return list(y_pred)

@task(returns=list, dataset_patches=COLLECTION_IN, y_pred=COLLECTION_IN)
def retrieve_predicted_tc(outputPath, dataset_patches, y_pred, patch_size):
    y_pred = np.array(y_pred)
    ds = dataset_patches[1]
    n_time = dataset_patches[2]
    n_rows = dataset_patches[3]
    n_cols = dataset_patches[4]
    times = dataset_patches[5]
    # reshape y pred to the dimensions: time x rows x cols x coordinate
    y_pred_reshaped = y_pred.reshape(n_time, n_rows, n_cols, 2)
    # create a latlons matrix with the same shape of y_pred_reshaped filled with nan
    cyclone_latlon_coords = np.full_like(y_pred_reshaped, fill_value=np.nan)
    # for each timestep
    for t in range(y_pred_reshaped.shape[0]):
        # for each row
        for i in range(y_pred_reshaped.shape[1]):
            # for each column
            for j in range(y_pred_reshaped.shape[2]):
                # if the model prediction is valid
                if y_pred_reshaped[t,i,j,0] >= 0.0 and y_pred_reshaped[t,i,j,1] >= 0.0:
                    # retrieve global row-col coordinates of the TC
                    global_rowcol = from_local_to_global((i,j), y_pred_reshaped[t,i,j,:], patch_size)
                    # retrieve global lat-lon coordinates of the TC
                    cyclone_latlon_coords[t,i,j,:] = (ds['lat'].data[global_rowcol[0]], ds['lon'].data[global_rowcol[1]])
    del y_pred_reshaped 
    return [np.expand_dims(cyclone_latlon_coords, axis=0), times]

@task(returns=list, cyclone_latlontime_coords_list=COLLECTION_IN)
def CreatePrediction(cyclone_latlontime_coords_list, output_path):
    cyclone_latlontime_coords_array = np.asarray(cyclone_latlontime_coords_list, dtype="object")
    cyclone_latlontime_coords_list1 = np.transpose(cyclone_latlontime_coords_array)
    cyclone_latlon_list = np.concatenate((cyclone_latlontime_coords_list1[0]), axis=0)
    cyclone_time_list = np.concatenate((cyclone_latlontime_coords_list1[1]), axis=0)
    cyclone_latlon_coords_array = np.concatenate((cyclone_latlon_list), axis=0)
    n_rows = np.arange(0,cyclone_latlon_list.shape[2])
    n_cols = np.arange(0,cyclone_latlon_list.shape[3])
    ds_out = xr.Dataset(
        data_vars=dict(
            lat_point=(["time", "rows", "cols"], cyclone_latlon_coords_array[:,:,:,0]),
            lon_point=(["time", "rows", "cols"], cyclone_latlon_coords_array[:,:,:,1]),
        ),
        coords=dict(
            time=(["time"], cyclone_time_list),
            rows=(["rows"], n_rows),
            cols=(["cols"], n_cols)
        ),
        attrs=dict(description="TC detection points"),
    )
    ds_out.to_netcdf(path=join(output_path,'TCprediction.nc'))
    return ds_out

