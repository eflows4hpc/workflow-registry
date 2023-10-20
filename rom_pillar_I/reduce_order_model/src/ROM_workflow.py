# Importing the Kratos Library
import sys
import time
import os
import KratosMultiphysics

# Import packages
import numpy as np

# Import pickle for serialization
import pickle

# Import pycompss
from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.api import compss_wait_on, compss_barrier
from pycompss.api.parameter import *
from pycompss.api.api import compss_barrier
from pycompss.api.software import software
from pycompss.api.data_transformation import *

from dts import *

SW_CATALOG = os.environ.get("SW_CATALOG","/software-catalog/packages")

# Workflows constants
TotalNumberOFCases = 5
number_of_dofs = 604264
snapshots_per_simulation = 11
number_of_columns = TotalNumberOFCases * snapshots_per_simulation
expected_shape = (number_of_dofs,number_of_columns)  # We will know the size of the array!
row_splits = 10
column_splits = 1
A_row_chunk_size = int(number_of_dofs / row_splits)
A_column_chunk_size = int(number_of_columns / column_splits)
desired_block_size = (A_row_chunk_size, A_column_chunk_size)
simulation_block_size = (number_of_dofs, snapshots_per_simulation)
desired_rank=30

@software(config_file = SW_CATALOG+"/kratos/fom.json")
def execute_FOM_instance(model,parameters, sample):
    import KratosMultiphysics
    from kratos_simulations import GetTrainingData
    current_model = KratosMultiphysics.Model()
    model.Load("ModelSerialization",current_model)
    del(model)
    current_parameters = KratosMultiphysics.Parameters()
    parameters.Load("ParametersSerialization",current_parameters)
    del(parameters)
    # get sample
    simulation = GetTrainingData(current_model,current_parameters,sample)
    simulation.Run()
    return simulation.GetSnapshotsMatrix()


@dt(target="rom", function=ROM_file_generation, type=OBJECT_TO_FILE, destination=sys.argv[3])
@software(config_file = SW_CATALOG + "/kratos/rom.json")
def execute_ROM_instance(model,parameters,sample,rom):
    import KratosMultiphysics 
    from kratos_simulations import RunROM_SavingData
    load_ROM(rom)
    current_model = KratosMultiphysics.Model()
    model.Load("ModelSerialization",current_model)
    del(model)
    current_parameters = KratosMultiphysics.Parameters()
    parameters.Load("ParametersSerialization",current_parameters)
    del(parameters)
    # get sample
    simulation = RunROM_SavingData(current_model,current_parameters,sample)
    simulation.Run()
    return simulation.GetSnapshotsMatrix()


@software(config_file = SW_CATALOG+"/kratos/model.json")
def load_model_parameters(model_file):
    import KratosMultiphysics
    from kratos_simulations import GetTrainingData
    with open(model_file,'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    model = KratosMultiphysics.Model()
    fake_sample = [5]
    simulation = GetTrainingData(model,parameters,fake_sample)
    serialized_model = KratosMultiphysics.StreamSerializer()
    serialized_model.Save("ModelSerialization",simulation.model)
    serialized_parameters = KratosMultiphysics.StreamSerializer()
    serialized_parameters.Save("ParametersSerialization",simulation.project_parameters)
    return serialized_model,serialized_parameters

@dt("blocks", load_blocks_rechunk, shape=expected_shape, block_size=simulation_block_size,
    new_block_size=desired_block_size, is_workflow=True)
@software(config_file = SW_CATALOG + "/py-dislib/dislib.json")
def rSVD(blocks, desired_rank=30):
    from dislib_parallel_svd import rsvd
    u,s = rsvd(blocks, desired_rank, A_row_chunk_size, A_column_chunk_size)
    return u


@dt("SnapshotsMatrixROM", load_blocks_rechunk, shape=expected_shape, block_size=simulation_block_size,
    new_block_size=desired_block_size, is_workflow=True)
@dt("SnapshotsMatrixFOM", load_blocks_rechunk, shape=expected_shape, block_size=simulation_block_size,
    new_block_size=desired_block_size, is_workflow=True)
@software(config_file = SW_CATALOG + "/py-dislib/dislib.json")
def compare_ROM_vs_FOM(SnapshotsMatrixROM, SnapshotsMatrixFOM):
    import dislib as ds
    import numpy as np
    #using the Frobenious norm of the snapshots of the solution
    original_norm= np.linalg.norm((SnapshotsMatrixFOM.norm().collect()))
    intermediate = ds.data.matsubtract(SnapshotsMatrixROM, SnapshotsMatrixFOM) #(available on latest release)
    intermediate = np.linalg.norm((intermediate.norm().collect()))
    final = intermediate/original_norm
    np.save('relative_error_rom.npy', final)

if __name__ == '__main__':

    data_path = sys.argv[1]
    parameters_template = sys.argv[2]
    rom_file = sys.argv[3]
    model_file="ProjectParameters_run.json"
    replace_template(parameters_template, model_file, '%MODEL_PATH%', data_path)    

    """
    Here we define the parameters for the simulation.
    In this case a sinlge parameter is defined.
    More parameters are possible.
    """
    sim_cfgs = range(5,10)
    model, parameters = load_model_parameters(model_file)
    """
    Stage 1
    - launches in parallel a Full Order Model (FOM) simulation for each simulation parameter.
    """
    sim_results=[]
    for cfg in sim_cfgs:
        sim_results.append(execute_FOM_instance(model,parameters,[cfg]))
    """
    Stage 2
    - computes the "fixed rank" randomized SVD in parallel using the dislib library  #TODO implement the fixed presicion RSVD
    """
    rom = rSVD(sim_results, desired_rank)
    """
    Stage 3
    - launches the Reduced Order Model simulations for the same simulation parameters used for the FOM
    """
    rom_results=[]
    for cfg in sim_cfgs:
        sim_results.append(execute_ROM_instance(model,parameters,[cfg],rom))
    
    #compare_ROM_vs_FOM(rom_results, sim_results)




