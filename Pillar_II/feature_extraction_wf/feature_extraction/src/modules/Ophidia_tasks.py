#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
PyCOMPSs PyOphidia tasks
==================
    This file declares the PyOphidia tasks in PyCOMPSs.
"""

# Imports
import os
from os.path import join
import uuid
import glob
import time
from datetime import date
from PyOphidia import cube, client

from pycompss.api.task import task
from pycompss.api.parameter import *

#Constant variables
NHOST = 1

@task(returns=object)
def OphImportClimAvg(client, input_path, file, measure):
    cube.Cube.client = client
    src_path=join(input_path, file)
    # Import of yearly file containing 4 values per day (6-hours)
    ClimAvg = cube.Cube.importnc2(src_path= src_path ,
    container='Test',
    measure=measure,
    import_metadata='yes',
    nfrag=4,
    nthreads=4*NHOST,
    ncores=1,
    nhost=NHOST,
    imp_dim='time',
    imp_concept_level='d', vocabulary='CF',hierarchy='oph_base|oph_base|oph_time',
    description='Clim Avg Temps'
    )
    return ClimAvg

@task(returns=object)
def OphImportForBaseline(client, input_path, file, measure, op):
    cube.Cube.client = client
    src_path = join(input_path, 'CMCC-CM3/2000_cam6-nemo4_025deg_tc.cam.h1.' + file + '-*-*-00000.nc')
    # Import of yearly file containing 4 values per day (6-hours)
    Year = cube.Cube.importncs(src_path= src_path,
    container='Test',
    measure=measure,
    import_metadata='yes',
    nfrag=4,
    nthreads=4*NHOST,
    ncores=1,
    nhost=NHOST,
    imp_dim='time',
    imp_concept_level='6', vocabulary='CF',hierarchy='oph_base|oph_base|oph_time',
    description='6-Hours Temps'
    )
    # Computation of the maximum or minimum on the 6-hours values to obtain a value per day and of the moving average on a 5-day window
    movingAvg = Year.apply(
    query="oph_shift('OPH_FLOAT', 'OPH_FLOAT', oph_moving_avg('OPH_FLOAT', 'OPH_FLOAT', oph_reduce2('OPH_FLOAT', 'OPH_FLOAT', measure," + op + ",4), 5, 'OPH_SMA'), -2, 0)",nthreads=4*NHOST)
    Year.delete()
    return movingAvg

@task(returns=object, movingavgcubes=COLLECTION_IN)
def OphClimAvg(client, input_path, movingavgcubes, filename):
    cube.Cube.client = client
    baseline=cube.Cube.intercube2(cubes=movingavgcubes, ncores=4*NHOST)
    baseline.exportnc2(output_path=input_path,output_name=filename)
    return baseline

@task(datacubes=COLLECTION_IN, dependences=COLLECTION_IN)
def OphDelete(client, datacubes, dependences):
    cube.Cube.client = client
    for x in range(len(datacubes)):
        datacubes[x].delete()

@task(returns=object, files=COLLECTION_IN)
def OphImport(client, input_path, files, measure, op):
    cube.Cube.client = client
    #REAL CASE: src_path = files
    src_path = join(input_path, 'CMCC-CM3/2000_cam6-nemo4_025deg_tc.cam.h1.0030-*-*-00000.nc')
    Year = cube.Cube.importncs(src_path= src_path,
        container='Test',
        measure=measure,
        import_metadata='yes',
        nfrag=4,
        nthreads=4*NHOST,
        ncores=1,
        nhost=NHOST,
        imp_dim='time',
        imp_concept_level='6', vocabulary='CF',hierarchy='oph_base|oph_base|oph_time',
        description='6-Hours Temps'
        )
    # Computation of the maximum or minimum on the 6-hours values to obtain a value per day 
    dailySeries = Year.apply(
    query="oph_reduce2('OPH_FLOAT', 'OPH_FLOAT', measure," + op + ",4)", nthreads=4*NHOST)
    Year.delete()
    return dailySeries

@task(returns=object, cubes=COLLECTION_IN)
def OphCompare(client, cubes, firstvalues, mask):
    cube.Cube.client = client
    #Computation of the difference between a year and the baseline; 
    #fixing of a mask by setting the values greater (for HWDI) or less (for CWDI) than 5°C to 1 , otherwise 0; 
    #count the number of consecutive 1s and set numbers less than 6 to 0 to identify durations
    Diff = cubes[0].intercube(cube2=cubes[1].pid, operation='sub', ncores=4*NHOST)
    Duration = Diff.apply(
    query="oph_predicate('OPH_INT','OPH_INT',oph_sequence('OPH_INT','OPH_INT', oph_predicate('OPH_FLOAT','OPH_INT'," + firstvalues + ",'x-5'," + mask + ",'1','0'), 'length', 'yes'),'x-5','>0','x','0')", nthreads=4*NHOST)
    Diff.delete()
    cubes[0].delete()
    return Duration

@task(returns=object)
def IndexDurationMax(client, output_path, duration, filename):
    cube.Cube.client = client
    #Take the maximum of each duration
    DurationMax = duration.reduce(operation='max',nthreads=4*NHOST, description="Max Duration cube")
    DurationMax.exportnc2(output_path=output_path,output_name=filename)
    return DurationMax

@task(returns=object)
def IndexDurationNumber(client, output_path, duration, filename):
    cube.Cube.client = client
    #Computation of the number of durations in a year (HWN or CWN)
    DurationMask = duration.apply(query="oph_predicate('OPH_INT','OPH_INT',measure,'x','>0','1','0')", nthreads=4*NHOST)
    DurationCount = DurationMask.reduce(operation='sum', nthreads=4*NHOST, description="Number of durations cube")
    DurationMask.delete()
    DurationCount.exportnc2(output_path=output_path,output_name=filename)
    return DurationCount

@task(returns=object)
def IndexDurationFrequency(client, output_path, duration, filename):
    cube.Cube.client = client
    #Computation of the Heat Wave frequency (HWF) or Cold Wave frequency (CWF)
    durationSum=duration.reduce(operation="sum")
    Freq=durationSum.apply(query="oph_mul_scalar('OPH_INT', 'OPH_FLOAT', measure,"+ str(1/365) +")", nthreads=4*NHOST, description="Frequency cube")
    durationSum.delete()
    Freq.exportnc2(output_path=output_path,output_name=filename)
    return Freq
