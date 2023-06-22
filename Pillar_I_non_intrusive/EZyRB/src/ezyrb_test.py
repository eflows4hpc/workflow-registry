from pycompss.api.api import compss_wait_on, compss_barrier
from pyeddl import eddl
import torch
from torch import nn
import numpy as np
import time
import sys
import os
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
plt.rc('font', size = 16)
from ezyrb_v3 import Database
from ezyrb_v3 import ReducedOrderModel as ROM
from ezyrb_v3 import POD, AE, AE_EDDL, Linear, RBF, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, ANN
# ----------------------------------------------------------------
if __name__ == "__main__":
    CaseType = (sys.argv[1]).upper() # "2D" or "3D
    ReductionMethod = (sys.argv[2]).upper() # "POD" "AE" "AE_EDDL"
Number_of_predictions = 1
Number_of_configurations = 3
# ----------------------------------------------------------------
if CaseType == "2D":
    files = '2d'
    param_h_lim = 59
    batch = 59
else:
    files = '3d'
    param_h_lim = 451
    batch = 50
# ----------------------------------------------------------------
# Load data as NumPy arrays:
# The snapshots represented in rows
snapshots_pressure = np.load('./data_{}/snapshots_pressure.npy'.format(files))
snapshots_vx = np.load('./data_{}/snapshots_vx.npy'.format(files))
snapshots_vy = np.load('./data_{}/snapshots_vy.npy'.format(files))
snapshots_vz = np.load('./data_{}/snapshots_vz.npy'.format(files))
time_parameter = np.load('./data_{}/time.npy'.format(files))
cells = np.load('./data_{}/cells.npy'.format(files))
pts = np.load('./data_{}/pts.npy'.format(files))
tr = cells if CaseType == "2D" else cells[:,[0,1,2]]
triang = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles=tr)

print("."*61)
print("Dataset info -->")
print('max cells',np.max(cells))
print('pts shape', pts.shape)
print('Param Mat shape {}, Snap Mat shape {}'.format(time_parameter.shape, snapshots_vx.shape))

print("."*61)
print("Solver info -->")
print("CaseType: {} Data".format(CaseType))
print("Reduction: ", ReductionMethod)
# print('Approximations: Linear, RBF, GPR, KNeighbors, RadiusNeighbors, ANN')
# print("Number of operations = 6 fittings (each 3 tasks) * {} predictions (each \
# 2 tasks) = {} operations".format(Number_of_predictions, 6*Number_of_predictions))
print('Approximation: RBF')
print("Number of operations = 1 fittings (3 tasks) * {} predictions (each \
2 tasks) = {} operations".format(Number_of_predictions, 1*Number_of_predictions))

print("."*61)
print("Resources info -->")
print("Total Number of Nodes = ({}), Number of Workers Nodes = ({})".format(int(os.environ["TotalNumNodes"]), int(os.environ["NumWorkerNodes"])))
print("Utilised CPUs(Threads) per Node = ({}), Utilised GPUs per Node = ({})".format(int(os.environ["CPUsPerNode"]), int(os.environ["GPUsPerNode"])))
print("Requested CPUs(Threads) = ({}), Actual CPUs(Threads) = ({})".format(int(os.environ["RequestedTotalNumCPUs"]), int(os.environ["ActualTotalNumCPUs"])))

# # Remove snap 55 
# snapshots_vx_train = np.delete(snapshots_vx,55,0)
# time_train = np.delete(time_parameter,55,0)
snapshots_vx_train = snapshots_vx
time_train = time_parameter
db = Database(time_train, snapshots_vx_train)
# ----------------------------------------------------------------
# # Independent instances of the approximation classes
# if ReductionMethod == "POD":
#     approximations_1 = {'RBF': RBF()}
#     #     'Linear': Linear(),
#     #     'RBF': RBF(),
#     #     'GPR': GPR(),
#     #     'KNeighbors': KNeighborsRegressor(),
#     #     'RadiusNeighbors':  RadiusNeighborsRegressor(),
#     #     'ANN': ANN([500, 100, 4], nn.Tanh(), [1,1e-5]),
#     # }
# elif ReductionMethod == "AE":
#     approximations_2 = {'RBF': RBF()}
#     #     'Linear': Linear(),
#     #     'RBF': RBF(),
#     #     'GPR': GPR(),
#     #     'KNeighbors': KNeighborsRegressor(),
#     #     'RadiusNeighbors':  RadiusNeighborsRegressor(),
#     #     'ANN': ANN([500, 100, 4], nn.Tanh(), [1,1e-5]),
#     # }
# elif ReductionMethod == "AE_EDDL":
#     approximations_3 = {'RBF': RBF()}
#     #     'Linear': Linear(),
#     #     'RBF': RBF(),
#     #     'GPR': GPR(),
#     #     'KNeighbors': KNeighborsRegressor(),
#     #     'RadiusNeighbors':  RadiusNeighborsRegressor(),
#     #     'ANN': ANN([500, 100, 4], nn.Tanh(), [1,1e-5]),
#     # }
# ----------------------------------------------------------------
list_roms = []; 
predictions = []; 
# methods = [];
parameters_list = list(np.random.randint(low = 0,high=param_h_lim,size=Number_of_predictions))
# start_time = time.time()
# ----------------------------------------------------------------
if ReductionMethod == "POD":
    for i in range(Number_of_configurations):
        rbf = RBF()
        pod = POD('svd',rank=4)
        rom = ROM(db, pod, rbf)
        list_roms.append(rom)

    for i in range(int(os.environ["ActualTotalNumCPUs"])):
        rom.warmup()

    compss_barrier()
    start_time = time.time()
    for model in list_roms:
        model.fit()
    compss_barrier()
    exec_time = (time.time() - start_time)

    for model in list_roms:
        for element in parameters_list:
            predictions.append(model.predict([element]))

elif ReductionMethod == "AE":
    for i in range(Number_of_configurations):
        rbf = RBF()
        ae = AE([500,100,4], [4,100,500], nn.Tanh(), nn.Tanh(),
                10, optimizer=torch.optim.Adam, lr=1e-4)
        rom = ROM(db, ae, rbf)
        list_roms.append(rom)

    for i in range(int(os.environ["ActualTotalNumCPUs"])):
        rom.warmup()

    compss_barrier()
    start_time = time.time()
    for model in list_roms:
        model.fit()
    compss_barrier()
    exec_time = (time.time() - start_time)

    for model in list_roms:
        for element in parameters_list:
            predictions.append(model.predict([element]))

elif ReductionMethod == "AE_EDDL":
    for i in range(Number_of_configurations):
        rbf = RBF()
        ae = AE_EDDL([500,100,4], [4,100,500], eddl.Tanh, eddl.Tanh,
                 5, batch, optimizer=eddl.adam, lr=1e-4,
                 cs=eddl.CS_CPU, training_type=1)
        rom = ROM(db, ae, rbf)
        list_roms.append(rom)

    for i in range(int(os.environ["ActualTotalNumCPUs"])):
        rom.warmup()

    compss_barrier()
    start_time = time.time()
    for model in list_roms:
        model.fit()
    compss_barrier()
    exec_time = (time.time() - start_time)

    for model in list_roms:
        for element in parameters_list:
            predictions.append(model.predict([element]))
# ----------------------------------------------------------------
# if Number_of_predictions == 3:
#     predictions= compss_wait_on(predictions)
#     fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(20, 18), sharey=True, sharex=True)
#     ax = ax.flatten()
#     j = 0;
#     for i in range(18):
#         ax[i].tricontourf(triang, predictions[i], levels=16)
#         ax[i].set_title('{} @ t = {}'.format(methods[j], parameters_list[(i) % 3]))
#         if (i+1) % 3 == 0: j+=1
#     plt.tight_layout()
#     plt.savefig("results")
print("."*61)
print('(p={}) --> {}-{} execution time = {} (sec)'.format(int(os.environ["ActualTotalNumCPUs"]), ReductionMethod, CaseType, exec_time))
print("."*61)
