import torch
from torch import nn
import numpy as np
import sys
import os
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
plt.rc('font', size = 16)
from ezyrb import Database
from ezyrb import ReducedOrderModel as ROM
from ezyrb import POD, AE, RBF
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

# # Remove snap 55 
# snapshots_vx_train = np.delete(snapshots_vx,55,0)
# time_train = np.delete(time_parameter,55,0)
snapshots_vx_train = snapshots_vx
time_train = time_parameter
db = Database(time_train, snapshots_vx_train)
# ----------------------------------------------------------------

param = 5
# ----------------------------------------------------------------
if ReductionMethod == "POD":
    rbf = RBF()
    pod = POD('svd',rank=4)
    rom = ROM(db, pod, rbf)
    rom.fit()
    prediction = rom.predict([param])

elif ReductionMethod == "AE":
    rbf = RBF()
    ae = AE([500,100,4], [4,100,500], nn.Tanh(), nn.Tanh(),
            3, optimizer=torch.optim.Adam, lr=1e-4)
    rom = ROM(db, ae, rbf)
    rom.fit()
    prediction = rom.predict([param])

print(prediction)