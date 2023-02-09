import dislib as ds
import numpy as np
import sys
import os
from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.api import compss_barrier
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import Type, COLLECTION_IN, FILE_IN, Depth
from dislib.data.array import Array
from dts import *

@constraint(computingUnits="$QR_CUS")
@task(Y_blocks={Type: COLLECTION_IN, Depth: 2}, returns=2)
def my_qr(Y_blocks):
    Y = np.block(Y_blocks)
    Q,R = np.linalg.qr(Y, mode='reduced')
    return Q,R

@constraint(computingUnits="$SVD_CUS")
@task(B_blocks={Type: COLLECTION_IN, Depth: 2}, returns=2)
def my_svd(B_blocks):
    B = np.block(B_blocks)
    U_hat, s, _ = np.linalg.svd(B, full_matrices=False)
    return U_hat, s

def rsvd(A, desired_rank, rsvd_A_row_chunk_size, rsvd_A_column_chunk_size):

#-----Dimensions--------

    k = desired_rank

    oversampling = 10
    p = k + oversampling
    n = A.shape[0]
    m = A.shape[1]

    # Matrix Omega Initialization
    omega_column_chunk_size = p
    omega_row_chunk_size = rsvd_A_column_chunk_size
    Omega = ds.random_array(shape=(m, p), block_size=(omega_row_chunk_size, omega_column_chunk_size) ) # Create a random projection matrix Omega of size mxp, for this test, p is of 110.

    Y = A @ Omega
    Q,R = my_qr(Y._blocks)
    Q=load_blocks_rechunk([Q], shape = (n, p), block_size= (n, p), new_block_size=(rsvd_A_row_chunk_size, omega_column_chunk_size))
    B = Q.T @ A
    U_hat, s = my_svd(B._blocks)
    U_hat = load_blocks_rechunk([U_hat], shape = (p, m), block_size = (p, m), new_block_size=(omega_column_chunk_size, omega_column_chunk_size))
    U_hat = U_hat[:,:k] #U_hat results into a matrix of pxk.
    U = Q @ U_hat  #STEP 5 (DISTRIBUTED): Project the reduced basis into Q. U results into a matrix of nxk.
    U = U.collect() #STEP 6 (collecting the required basis):
    s = compss_wait_on(s)
    s = s[:k]
    print('succesfully computed rsvd :)')
    return U, s
