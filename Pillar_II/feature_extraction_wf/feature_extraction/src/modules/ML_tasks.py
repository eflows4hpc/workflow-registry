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
from cdo import *
cdo = Cdo()
from numpy import asarray
from numpy import savetxt
from pycompss.api.task import task
from pycompss.api.parameter import *

# custom library containing utility functions
from lib import load_model_and_scaler, TFScaler, from_local_to_global

# ML Tasks

@task(returns=str, vars=COLLECTION_IN)
def regridding(filename:str, vars:list, cmcc_cm3_remap_dir:str, grid_file:str):
    cdo = Cdo()
    vars_string = ','.join(vars)
    remapped_file = join(cmcc_cm3_remap_dir, filename.split('/')[-1])
    cdo.remapcon(grid_file, input='-select,name='+vars_string+' '+filename, output=remapped_file)
    return remapped_file

@task(returns=list, vars=COLLECTION_IN)
def get_dataset_patches(filename:str, vars:list, patch_size:int=40):
    # open xarray dataset from filenames
    ds = xr.open_dataset(filename)
    # patch the dataset
    patch_ds = ds.coarsen({'latitude':patch_size, 'longitude':patch_size}, boundary="trim").construct({'longitude':("cols", "lon_range"), 'latitude':("rows", "lat_range")})

    # get the number of patches for rows and columns
    row_blocks = len(patch_ds['rows'])
    col_blocks = len(patch_ds['cols'])
    X = []
    for i in range(row_blocks):
        for j in range(col_blocks):
            xij = np.stack([patch_ds.isel({'rows':i, 'cols':j})[var] for var in vars], axis=-1)
            X.append(xij)
    X = np.ravel(np.column_stack(tuple(X))).reshape(-1, patch_size, patch_size, len(vars))
    return [X, ds, patch_ds]


@task(returns=list, X=COLLECTION_IN)
def predict(X, is_ckpt, model_dir, batch_size, patch_size, label_no_cyclone):
    model, scaler = load_model_and_scaler(model_dir, is_ckpt=is_ckpt)
    # scale the data
    X_scaled = scaler.transform(X[0])
    # predict with the model
    y_pred = model.predict(X_scaled, batch_size)
    # FILTER RESULTS (CONSIDER AS A BLACK BOX)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # set to nan all negative coordinates
    y_pred = np.where(y_pred < 0.0, np.nan, y_pred)
    # round to patch_size-1 all predictions greater than patch_size-1
    y_pred = np.where(y_pred > patch_size-1, patch_size-1, y_pred)
    # build a filter to remove spurious coordinates
    filter = np.repeat((y_pred >= 0.0).min(axis=1)[:,np.newaxis], repeats=2, axis=1)
    # get filtered predictions
    y_pred = np.where(filter > 0, y_pred, label_no_cyclone)
    # round coordinates
    y_pred = np.round(y_pred, 0)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    return list(y_pred)

@task(returns=list, dataset_patches=COLLECTION_IN, y_pred=COLLECTION_IN)
def retrieve_predicted_tc(outputPath, dataset_patches, y_pred, patch_size:int=40):
    y_pred = np.array(y_pred)
    ds = dataset_patches[1]
    patch_ds = dataset_patches[2]
    # reshape y pred to the dimensions: time x rows x cols x coordinate
    y_pred_reshaped = y_pred.reshape(len(patch_ds['time']), len(patch_ds['rows']), len(patch_ds['cols']), 2)

    # create a latlons matrix with the same shape of y_pred_reshaped filled with nan
    cyclone_latlon_coords = np.full_like(y_pred_reshaped, fill_value=np.nan)
    # row-column coordinates of TCs
    cyclone_global_rowcol_coords = []
    # patch id containing a TC
    cyclone_patch_ids = []

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
                    cyclone_latlon_coords[t,i,j,:] = (ds['latitude'].data[global_rowcol[0]], ds['longitude'].data[global_rowcol[1]])

                    # append the global row column
                    cyclone_global_rowcol_coords.append(global_rowcol)

                    # append the patch id
                    cyclone_patch_ids.append((i,j))
    
    pred_list = asarray(cyclone_latlon_coords.reshape(-1,2)) # just one timestep
    savetxt(outputPath + 'prediction_list.csv', pred_list, delimiter=',')

    return cyclone_latlon_coords


