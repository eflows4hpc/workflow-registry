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



# import dislib as ds
# import numpy as np
# import sys
# import os

# from pycompss.api.task import task
# from pycompss.api.constraint import constraint
# from pycompss.api.api import compss_barrier
# from pycompss.api.api import compss_wait_on
# from dislib.data.array import Array



# #@constraint(computingUnits=48)
# @task(returns=2)
# def my_qr(Y):
#     Q,R = np.linalg.qr(Y, mode='reduced')
#     return Q,R

# #@constraint(computingUnits=48)
# @task(returns=1)
# def my_svd(B):
#     U_hat, _, _ = np.linalg.svd(B, full_matrices=False)
#     return U_hat



# """
# latest version of workflow
# """


# def load_blocks_array(blocks, shape, block_size):
#     if shape[0] < block_size[0] or  shape[1] < block_size[1]:
#         raise ValueError("The block size is greater than the ds-array")
#     return Array(blocks, shape=shape, top_left_shape=block_size,
#                      reg_shape=block_size, sparse=False)
# def load_blocks_rechunk(blocks, shape, block_size, new_block_size):
#     if shape[0] < new_block_size[0] or  shape[1] < new_block_size[1]:
#         raise ValueError("The block size requested for rechunk"
#                          "is greater than the ds-array")
#     final_blocks = [[]]
#     # Este bucle lo puse por si los Future objects se guardan en una lista, en caso de que la forma de guardarlos cambie, también cambiará un poco este bucle.
#     # Si blocks se pasa ya como (p. ej) [[Future_object, Future_object]] no hace falta.
#     for block in blocks:
#         final_blocks[0].append(block)
#     arr = load_blocks_array(final_blocks, shape, block_size)
#     return arr.rechunk(new_block_size)




# def rsvd(A, desired_rank):

# #-----Dimensions--------

#     k = desired_rank

#     oversampling = 10
#     p = k + oversampling

#     row_splits = 10
#     column_splits = 1

#     n = A.shape[0]
#     m = A.shape[1]
#     A_row_chunk_size = int( n / row_splits)
#     A_column_chunk_size = int( m / column_splits)
#     A = A.rechunk((A_row_chunk_size,A_column_chunk_size))

# #-----Matrix Omega Initialization--------
# #    omega_column_chunk_size =  int( p/column_splits_to_use)
#     omega_column_chunk_size = p
#     omega_row_chunk_size = A_column_chunk_size
#     #print(f"Generating a random array of shape ({m}, {p}) and block size ({omega_row_chunk_size}, {omega_column_chunk_size})")
#     Omega = ds.random_array(shape=(m, p), block_size=(omega_row_chunk_size, omega_column_chunk_size) ) # Create a random projection matrix Omega of size mxp, for this test, p is of 110.
# #----------------------------------
#     Y = A @ Omega
# # STEP 1 (DISTRIBUTED): Sample the column space of A. Y results into an nxp matrix.

# #""SERIAL STEPS START""
#     #Y = Y.collect()
# # STEP 2 (SERIAL): Serial QR, to be done in distributed. Q results into a matrix of nxp
#     Q,R = my_qr(Y._blocks)
#     #Q = compss_wait_on(Q)
#     Q=load_blocks_rechunk([Q], shape = (n, p), block_size= (n, p), new_block_size=(A_row_chunk_size, omega_column_chunk_size))
#     #Q = ds.array(Q, block_size=(A_row_chunk_size, omega_column_chunk_size))
# #"SERIAL STEPS FINISH"
#     B = Q.T @ A
# # STEP 3 (DISTRIBUTED): Project A into the orthonormal basis. B results into a matrix of pxm.

# #""""SERIAL STEPS START"""
#     #B = B.collect()
# #    U_hat, _, _ = np.linalg.svd(B, full_matrices=False)   # STEP 4 (SERIAL): Economy SVD.
#     U_hat, s = my_svd(B._blocks)
#     #U_hat = compss_wait_on(U_hat)
#     #U_hat = ds.array(U_hat, block_size=(omega_column_chunk_size, omega_column_chunk_size))
#     U_hat = load_blocks_rechunk([U_hat], shape = (40, 4), block_size = (40, 4), new_block_size=(omega_column_chunk_size, 1))# The new block size should be (omega_column_chunk_size, omega_column_chunk_size) but I do not have the real measure, I am only executing with 4 files.
# #""""SERIAL STEPS FINISH"""
# # The desired rank is k, therefore, a truncation of the SVD is applied.
#     U_hat = U_hat[:,:k] #U_hat results into a matrix of pxk.
#     U = Q @ U_hat  #STEP 5 (DISTRIBUTED): Project the reduced basis into Q. U results into a matrix of nxk.
#     U = U.collect() #STEP 6 (collecting the required basis):
#     s = compss_wait_on(s)
#     s = s[:k]
#     print('succesfully computed rsvd :)')
#     return U, s


@constraint(computingUnits=2)
@task(Y_blocks={Type: COLLECTION_IN, Depth: 2}, returns=2)
def my_qr(Y_blocks):
    Y = np.block(Y_blocks)
    Q,R = np.linalg.qr(Y, mode='reduced')
    return Q,R
    #
    #
#Q,R = my_qr(Y._blocks)


'''@constraint(computingUnits=1)
@task(returns=2)
def my_qr(Y):
    Q,R = np.linalg.qr(Y, mode='reduced')
    return Q,R
'''
@constraint(computingUnits=1)
@task(B_blocks={Type: COLLECTION_IN, Depth: 2}, returns=2)
def my_svd(B_blocks):
    B = np.block(B_blocks)
    U_hat, s, _ = np.linalg.svd(B, full_matrices=False)
    return U_hat, s

def load_blocks_array(blocks, shape, block_size):
    if shape[0] < block_size[0] or  shape[1] < block_size[1]:
        raise ValueError("The block size is greater than the ds-array")
    return Array(blocks, shape=shape, top_left_shape=block_size,
                     reg_shape=block_size, sparse=False)
def load_blocks_rechunk(blocks, shape, block_size, new_block_size):
    if shape[0] < new_block_size[0] or  shape[1] < new_block_size[1]:
        raise ValueError("The block size requested for rechunk"
                         "is greater than the ds-array")
    final_blocks = [[]]
    # Este bucle lo puse por si los Future objects se guardan en una lista, en caso de que la forma de guardarlos cambie, también cambiará un poco este bucle.
    # Si blocks se pasa ya como (p. ej) [[Future_object, Future_object]] no hace falta.
    for block in blocks:
        final_blocks[0].append(block)
    arr = load_blocks_array(final_blocks, shape, block_size)
    return arr.rechunk(new_block_size)


def rsvd(A, desired_rank):

#-----Dimensions--------

    k = desired_rank

    oversampling = 10
    p = k + oversampling

    row_splits = 10
    column_splits = 1

    n = A.shape[0]
    m = A.shape[1]
    A_row_chunk_size = int( n / row_splits)
    A_column_chunk_size = int( m / column_splits)
    A = A.rechunk((A_row_chunk_size,A_column_chunk_size))

#-----Matrix Omega Initialization--------
#    omega_column_chunk_size =  int( p/column_splits_to_use)
    omega_column_chunk_size = p
    omega_row_chunk_size = A_column_chunk_size
    #print(f"Generating a random array of shape ({m}, {p}) and block size ({omega_row_chunk_size}, {omega_column_chunk_size})")
    Omega = ds.random_array(shape=(m, p), block_size=(omega_row_chunk_size, omega_column_chunk_size) ) # Create a random projection matrix Omega of size mxp, for this test, p is of 110.
#----------------------------------
    Y = A @ Omega
# STEP 1 (DISTRIBUTED): Sample the column space of A. Y results into an nxp matrix.

#""SERIAL STEPS START""
    #Y = Y.collect()
# STEP 2 (SERIAL): Serial QR, to be done in distributed. Q results into a matrix of nxp
    Q,R = my_qr(Y._blocks)
    #Q = compss_wait_on(Q)
    Q=load_blocks_rechunk([Q], shape = (n, p), block_size= (n, p), new_block_size=(A_row_chunk_size, omega_column_chunk_size))
    #Q = ds.array(Q, block_size=(A_row_chunk_size, omega_column_chunk_size))
#"SERIAL STEPS FINISH"
    B = Q.T @ A
# STEP 3 (DISTRIBUTED): Project A into the orthonormal basis. B results into a matrix of pxm.

#""""SERIAL STEPS START"""
    #B = B.collect()
#    U_hat, _, _ = np.linalg.svd(B, full_matrices=False)   # STEP 4 (SERIAL): Economy SVD.
    U_hat, s = my_svd(B._blocks)
    #U_hat = compss_wait_on(U_hat)
    #U_hat = ds.array(U_hat, block_size=(omega_column_chunk_size, omega_column_chunk_size))
    #U_hat = load_blocks_rechunk([U_hat], shape = (40, 4), block_size = (40, 4), new_block_size=(omega_column_chunk_size, 1))# The new block size should be (omega_column_chunk_size, omega_column_chunk_size) but I do not have the real measure, I am only executing with 4 files.
    U_hat = load_blocks_rechunk([U_hat], shape = (p, m), block_size = (p, m), new_block_size=(omega_column_chunk_size, omega_column_chunk_size))
#""""SERIAL STEPS FINISH"""
# The desired rank is k, therefore, a truncation of the SVD is applied.
    U_hat = U_hat[:,:k] #U_hat results into a matrix of pxk.
    U = Q @ U_hat  #STEP 5 (DISTRIBUTED): Project the reduced basis into Q. U results into a matrix of nxk.
    U = U.collect() #STEP 6 (collecting the required basis):
    s = compss_wait_on(s)
    s = s[:k]
    print('succesfully computed rsvd :)')
    return U, s
