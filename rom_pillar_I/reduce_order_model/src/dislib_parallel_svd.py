import dislib as ds
import numpy as np

from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import Type, COLLECTION_IN, FILE_IN, Depth
from dislib.data.array import Array

from dts import *
from dislib.decomposition.tsqr.base import tsqr

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

def tsqr_svd(ds_array,partitions, tol = 1e-6):
    M,N = ds_array.shape
    Q, R = tsqr(ds_array, n_reduction = partitions, mode="reduced", indexes=None)
    shape_for_rechunking = Q.shape[1]
    B = Q.T@ds_array
    u_hat, s = my_svd(B._blocks)
    #u_hat_dislib = ds.array(u_hat, block_size=(shape_for_rechunking,shape_for_rechunking))
    u_hat_dislib = load_blocks_rechunk([u_hat], shape = (shape_for_rechunking, N), block_size = (shape_for_rechunking, N), new_block_size=(shape_for_rechunking,shape_for_rechunking))
    number_of_singular_values = get_number_of_singular_values_for_given_tolerance(M, N, s, tol)
    U = Q@u_hat_dislib
    U = U.collect()
    number_of_singular_values = compss_wait_on(number_of_singular_values)
    U = U[:,:number_of_singular_values]
    return U

@constraint(computingUnits="$ComputingUnits")
@task(returns=1)
def get_number_of_singular_values_for_given_tolerance(M, N, s, epsilon):
    dimMATRIX = max(M,N)
    tol = dimMATRIX*np.finfo(float).eps*max(s)/2
    R = np.sum(s > tol)  # Definition of numerical rank
    if epsilon == 0:
        K = R
    else:
        SingVsq = np.multiply(s,s)
        SingVsq.sort()
        normEf2 = np.sqrt(np.cumsum(SingVsq))
        epsilon = epsilon*normEf2[-1] #relative tolerance
        T = (sum(normEf2<epsilon))
        K = len(s)-T
    K = min(R,K)
    return K

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
    return U, s

def DivideInPartitions(NumTerms, NumTasks):
    Partitions = np.zeros(NumTasks+1,dtype=np.int)
    PartitionSize = int(NumTerms / NumTasks)
    Partitions[0] = 0
    Partitions[NumTasks] = NumTerms
    for i in range(1,NumTasks):
        Partitions[i] = Partitions[i-1] + PartitionSize
    return Partitions

def get_numpy_array(arr, global_ids):
    b_locks = arr[global_ids]
    blocks = b_locks._blocks
    if len(blocks) == 1:
        if len(blocks[0]) == 1:
            return to_block([blocks[0][0]])
    return to_block(blocks)


@constraint(computingUnits="$ComputingUnits")
@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns = np.array)
def to_block(blocks):
    return np.block(blocks)

def Initialize_ECM_Lists(arr): # Generate initial list of ids and weights
    number_of_rows = arr.shape[0]
    global_ids = np.array(range(number_of_rows))
    global_weights = np.ones(len(global_ids))

    return global_ids, global_weights

@constraint(computingUnits="$ComputingUnits")
@task(returns=3)
def GetElementsOfPartition(np_array, global_ids, global_weights, block_len, block_num, title, final_truncation = 1e-6):
    from kratos_simulations import CalculateAndSelectElements 
    projected_residuals_matrix = np_array * global_weights[:, np.newaxis] #try making the weights and indexes ds arrays?
    local_ids, weights = CalculateAndSelectElements(projected_residuals_matrix, title, final_truncation)
    indexes_2 = np.argsort(local_ids) #this is necessary, since dislib cannot return un-ordered indexes
    return global_ids[local_ids[indexes_2]], local_ids[indexes_2]+block_len*block_num, weights[indexes_2]*global_weights[local_ids[indexes_2]]

def Parallel_ECM(arr,block_len,global_ids,global_weights,final=False, final_truncation=1e-6):
    # arr is a dislib array where all chunks have length block_len. Contains the info for all the elements
    # that were selected in the last recursion
    # global_ids and global_weights are lists with the global values for all the elements in arr

    number_of_rows = arr.shape[0]

    if final:
        ecm_mode='final'
        if len(arr._blocks)>1:
            arr=arr.rechunk(arr.shape)
        block_len=number_of_rows
    else:
        ecm_mode='intermediate'

    # Creating lists to store ids and weights
    ids_global_store = []
    ids_local_store= []
    weights_store = []

    for j in range(len(arr._blocks)):
        # We get the part of the global ids and weights corresponding to the elements in the chunk.
        chunk_global_ids = global_ids[list(range(block_len*j,min(block_len*(j+1),int(number_of_rows))))]
        chunk_global_weights = global_weights[list(range(block_len*j,min(block_len*(j+1),int(number_of_rows))))]
        # We run the ECM algorithm and get the list of chosen elements in both local and global notation, and their global weights
        ids,l_ids,w = GetElementsOfPartition(arr._blocks[j][0],chunk_global_ids,chunk_global_weights,block_len,j,ecm_mode,final_truncation)
        ids_global_store.append(ids)
        ids_local_store.append(l_ids)
        weights_store.append(w)

    # Synchronize the ids and weights lists
    for i in range(len(ids_global_store)):
        if i==0:
            temp_global_ids = compss_wait_on(ids_global_store[i])
            temp_local_ids = compss_wait_on(ids_local_store[i])
            temp_global_weights = compss_wait_on(weights_store[i])
        else:
            temp_global_ids = np.r_[temp_global_ids,compss_wait_on(ids_global_store[i])]
            temp_local_ids = np.r_[temp_local_ids,compss_wait_on(ids_local_store[i])]
            temp_global_weights = np.r_[temp_global_weights,compss_wait_on(weights_store[i])]

    global_ids = temp_global_ids
    local_ids = temp_local_ids
    global_weights = temp_global_weights

    # We return the rows of the array corresponding to the chosen elements. Also the global ids and weights.
    # This dislib array will still have the the same chunk size, just with less chunks
    return arr[local_ids], global_ids, global_weights
