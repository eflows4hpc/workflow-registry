#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
PyCOMPSs CMCC-CM3 Model tasks definition
==================
    This file declares the CMCC-CM3 Model tasks in PyCOMPSs.
"""

# Imports
import os
from os.path import join
import uuid
import time
import subprocess
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.binary import binary

# Constant values
SLEEP = 5

@task(fds=STREAM_IN, returns=list)
def read_files(fds):
    num_total = 0
    list_files = []
    while num_total < 364:
        # Poll new files
        print("Polling files")
        new_files = fds.poll()
        # Process files
        for nf in new_files:
            print("RECEIVED FILE: " + str(nf))
            if str(nf).endswith('.nc'):
                if str(nf).split('/')[-1].split('.')[2] == 'h1':
                    list_files.append(str(nf))
        # Sort read files
        list_files = sorted(list_files)[ : -1]
        # Accumulate read files
        num_total = len(list_files)
        # Sleep between requests
        time.sleep(SLEEP)
    # Return the number of processed files
    return list_files
