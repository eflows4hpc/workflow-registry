# Importing the Kratos Library
import KratosMultiphysics
import sys
import time
import os

# Import packages
import numpy as np

# Import pickle for serialization
import pickle

# Import pycompss
from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.api import compss_wait_on, compss_barrier
from pycompss.api.parameter import *
# from exaqute.ExaquteTaskPyCOMPSs import *   # to execute with runcompss

from parallel_randomized_svd import rsvd

from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis
from KratosMultiphysics.RomApplication.fluid_dynamics_analysis_rom import FluidDynamicsAnalysisROM
from pycompss.api.api import compss_barrier
import json

#import relevant dislib array
import dislib as ds
from dislib.data.array import Array




"""
Functions by Fernando BSC, start
"""

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

"""
Functions by Fernando BSC, finish
"""





class GetTrainingData(FluidDynamicsAnalysis):

    def __init__(self, model, project_parameters, sample):
        super().__init__(model, project_parameters)
        self.velocity = sample[0]
        #self.ith_parameter = sample[i] #more paramateres possible
        self.time_step_solution_container = []

    def ModifyInitialProperties(self):
        super().ModifyInitialProperties()
        self.project_parameters["processes"]["boundary_conditions_process_list"][0]["Parameters"]["modulus"].SetString(str(self.velocity)+'*y*(1-y)*sin(pi*t*0.5)')
        self.project_parameters["processes"]["boundary_conditions_process_list"][1]["Parameters"]["modulus"].SetString(str(self.velocity)+"*y*(1-y)" )

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        ArrayOfResults = []
        for node in self._GetSolver().GetComputingModelPart().Nodes:
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X, 0))
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y, 0))
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Z, 0))
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.PRESSURE, 0))
        self.time_step_solution_container.append(ArrayOfResults)

    def GetSnapshotsMatrix(self):
        ### Building the Snapshot matrix ####
        SnapshotMatrix = np.zeros((len(self.time_step_solution_container[0]), len(self.time_step_solution_container)))
        for i in range(len(self.time_step_solution_container)):
            Snapshot_i= np.array(self.time_step_solution_container[i])
            SnapshotMatrix[:,i] = Snapshot_i.transpose()
        self.time_step_solution_container = []
        return SnapshotMatrix


###############################################################################################################################################################################





class RunROM_SavingData(FluidDynamicsAnalysisROM):

    def __init__(self, model, project_parameters, sample):
        super().__init__(model, project_parameters)
        self.velocity = sample[0]
        #self.ith_parameter = sample[i] #more paramateres possible
        self.time_step_solution_container = []

    def ModifyInitialProperties(self):
        super().ModifyInitialProperties()
        self.project_parameters["processes"]["boundary_conditions_process_list"][0]["Parameters"]["modulus"].SetString(str(self.velocity)+'*y*(1-y)*sin(pi*t*0.5)')
        self.project_parameters["processes"]["boundary_conditions_process_list"][1]["Parameters"]["modulus"].SetString(str(self.velocity)+"*y*(1-y)" )
        print(self.project_parameters)

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        ArrayOfResults = []
        for node in self._GetSolver().GetComputingModelPart().Nodes:
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X, 0))
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y, 0))
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Z, 0))
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.PRESSURE, 0))
        self.time_step_solution_container.append(ArrayOfResults)

    def GetSnapshotsMatrix(self):
        ### Building the Snapshot matrix ####
        SnapshotMatrix = np.zeros((len(self.time_step_solution_container[0]), len(self.time_step_solution_container)))
        for i in range(len(self.time_step_solution_container)):
            Snapshot_i= np.array(self.time_step_solution_container[i])
            SnapshotMatrix[:,i] = Snapshot_i.transpose()
        self.time_step_solution_container = []
        return SnapshotMatrix


###############################################################################################################################################################################

# function generating the sample
def GetValueFromListList(Cases,iteration):
    Case = Cases[iteration]
    return Case

@constraint(computing_units="$ComputingUnits")
@task(returns = np.array)
def ExecuteInstance_Task(pickled_model,pickled_parameters,Cases,instance):
    # overwrite the old model serializer with the unpickled one
    model_serializer = pickle.loads(pickled_model)
    current_model = KratosMultiphysics.Model()
    model_serializer.Load("ModelSerialization",current_model)
    del(model_serializer)
    # overwrite the old parameters serializer with the unpickled one
    serialized_parameters = pickle.loads(pickled_parameters)
    current_parameters = KratosMultiphysics.Parameters()
    serialized_parameters.Load("ParametersSerialization",current_parameters)
    del(serialized_parameters)
    # get sample
    sample = GetValueFromListList(Cases,instance) # take one of them
    simulation = GetTrainingData(current_model,current_parameters,sample)
    simulation.Run()
    return simulation.GetSnapshotsMatrix()

@constraint(computing_units="$ComputingUnits")
@task(returns = np.array, rom_file=FILE_IN)
def ExecuteInstance_Task_ROM(pickled_model,pickled_parameters,Cases,instance,rom_file):
    load_ROM(rom_file)
    # overwrite the old model serializer with the unpickled one
    model_serializer = pickle.loads(pickled_model)
    current_model = KratosMultiphysics.Model()
    model_serializer.Load("ModelSerialization",current_model)
    del(model_serializer)
    # overwrite the old parameters serializer with the unpickled one
    serialized_parameters = pickle.loads(pickled_parameters)
    current_parameters = KratosMultiphysics.Parameters()
    serialized_parameters.Load("ParametersSerialization",current_parameters)
    del(serialized_parameters)
    # get sample
    sample = GetValueFromListList(Cases,instance) # take one of them
    print("CWD:" + str(os.getcwd()))
    simulation = RunROM_SavingData(current_model,current_parameters,sample)
    simulation.Run()
    return simulation.GetSnapshotsMatrix()

@constraint(computing_units="$ComputingUnits")
@task(parameter_file_name=FILE_IN,returns=2)
def SerializeModelParameters_Task(parameter_file_name):
    with open(parameter_file_name,'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    model = KratosMultiphysics.Model()
    fake_sample = [5]
    simulation = GetTrainingData(model,parameters,fake_sample)
    serialized_model = KratosMultiphysics.StreamSerializer()
    serialized_model.Save("ModelSerialization",simulation.model)
    serialized_parameters = KratosMultiphysics.StreamSerializer()
    serialized_parameters.Save("ParametersSerialization",simulation.project_parameters)
    # pickle dataserialized_data
    pickled_model = pickle.dumps(serialized_model, 2) # second argument is the protocol and is NECESSARY (according to pybind11 docs)
    pickled_parameters = pickle.dumps(serialized_parameters, 2)
    print("\n","#"*50," SERIALIZATION COMPLETED ","#"*50,"\n")
    return pickled_model,pickled_parameters

def load_ROM(rom_file):
    print(str(os.environ))
    working_dir = os.environ["COMPSS_WORKING_DIR"]
    print("working dir: " + working_dir)
    if not os.path.exists('RomParameters.json'):
        print("Creating a symlink")
        try:
            os.symlink(rom_file, 'RomParameters.json')
        except:
            print("Ignoring exception in symlink creation")
    else:
        print("ROM already loaded")
    print(str(os.getcwd()))
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    print("Files: " + str(files))
    sys.stdout.flush()

def replace_template(template_file, parameter_file_name, keyword, replacement):
    fin = open(template_file, "rt")
    fout = open(parameter_file_name, "wt")
    for line in fin:
        fout.write(line.replace(keyword, replacement))
    fin.close()
    fout.close()

def Stage1_RunFOM(parameters, parameter_file_name):
    # set the ProjectParameters.json path
    # create a serialization of the model and of the project parameters
    pickled_model,pickled_parameters = SerializeModelParameters_Task(parameter_file_name)

    TotalNumberOFCases = len(parameters)

    blocks = []
    # start algorithm
    for instance in range (0,TotalNumberOFCases):
        blocks.append(ExecuteInstance_Task(pickled_model,pickled_parameters,parameters,instance))

    number_of_dofs = 604264 # change manually :(
    snapshots_per_simulation = 11 #change manually :(
    expected_shape = (number_of_dofs, TotalNumberOFCases*snapshots_per_simulation) #We will know the size of the array!
    desired_block_size = (54900,snapshots_per_simulation*TotalNumberOFCases)
    simulation_shape = (number_of_dofs, snapshots_per_simulation)
    # But what happens when an instance of the simulations fails? that part of the array will be empty?

    print('about to create ds-array')
    arr = load_blocks_rechunk(blocks, shape = expected_shape, block_size = simulation_shape, new_block_size = desired_block_size)
    print("ds-array created and it looks like this: ")
    print(arr)

    return arr



def Stage2_rSVD(SnapshotsMatrix, rom_file_name = 'RomParameters.json', desired_rank = 30):

    print("entered svd")
    u,s = rsvd(SnapshotsMatrix, desired_rank)
    print("computed svd")

    ### Saving the nodal basis ###  #TODO improve format AND to everything in parallel
    basis_POD={"rom_settings":{},"nodal_modes":{}}
    basis_POD["rom_settings"]["nodal_unknowns"] = ["VELOCITY_X","VELOCITY_Y","VELOCITY_Z","PRESSURE"]
    basis_POD["rom_settings"]["number_of_rom_dofs"] = np.shape(u)[1]
    Dimensions = len(basis_POD["rom_settings"]["nodal_unknowns"])
    N_nodes=np.shape(u)[0]/Dimensions
    N_nodes = int(N_nodes)
    node_Id=np.linspace(1,N_nodes,N_nodes)
    i = 0
    for j in range (0,N_nodes):
        basis_POD["nodal_modes"][int(node_Id[j])] = (u[i:i+Dimensions].tolist())
        i=i+Dimensions

    with open(rom_file_name, 'w') as f:
        json.dump(basis_POD,f, indent=2)
    print('\n\nNodal basis printed in json format\n\n')


def Stage3_RunROM(parameters, parameter_file_name, rom_file_name):

    # create a serialization of the model and of the project parameters
    pickled_model,pickled_parameters = SerializeModelParameters_Task(parameter_file_name)

    TotalNumberOFCases = len(parameters)

    blocks = []
    # start algorithm
    for instance in range (0,TotalNumberOFCases):
        blocks.append(ExecuteInstance_Task_ROM(pickled_model,pickled_parameters,parameters,instance, rom_file_name))

    # print('TotalNumberOFCases: ', TotalNumberOFCases)
    # #print('snapshots_per_simulation: ', snapshots_per_simulation)
    # #print('number_of_dofs.npy: ',number_of_dofs)
    # number_of_dofs = 604260
    # snapshots_per_simulation = 10
    # expected_shape = (number_of_dofs, TotalNumberOFCases*snapshots_per_simulation) #We will know the size of the array!
    # desired_block_size = (100000,snapshots_per_simulation*TotalNumberOFCases)
    # simulation_shape = (number_of_dofs, snapshots_per_simulation)
    # # But what happens when an instance of the simulations fails? that part of the array will be empty?

    # print('about to create ds-array')
    # arr = load_blocks_rechunk(blocks, shape = expected_shape, block_size = simulation_shape, new_block_size = desired_block_size)
    # print("ds-array created and it looks like this: ")
    # print(arr)

    number_of_dofs = 604264
    snapshots_per_simulation = 11
    expected_shape = (number_of_dofs, TotalNumberOFCases*snapshots_per_simulation) #We will know the size of the array!
    desired_block_size = (54900,snapshots_per_simulation*TotalNumberOFCases)
    simulation_shape = (number_of_dofs, snapshots_per_simulation)
    # But what happens when an instance of the simulations fails? that part of the array will be empty?

    print('about to create ds-array')
    arr = load_blocks_rechunk(blocks, shape = expected_shape, block_size = simulation_shape, new_block_size = desired_block_size)
    print("ds-array created and it looks like this: ")
    print(arr)

    return arr


def compare_ROM_vs_FOM(SnapshotsMatrixROM, SnapshotsMatrix):
    #using the Frobenious norm of the snapshots of the solution
    original_norm= np.linalg.norm((SnapshotsMatrix.norm().collect()))
    intermediate = ds.data.matsubtract(SnapshotsMatrixROM,SnapshotsMatrix) #(available on latest release)
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
    parameters=[]
    for i in range(5,10):
        parameters.append([i]) #inlet fluid velocity

    SnapshotsMatrix = Stage1_RunFOM(parameters, model_file)
    """
    Stage 1
    - launches in parallel a Full Order Model (FOM) simulation for each simulation parameter.
    - returns a distributed array (ds-array) with the results of the simulations.
    """

    Stage2_rSVD(SnapshotsMatrix, rom_file)
    """
    Stage 2
    - computes the "fixed rank" randomized SVD in parallel using the dislib library  #TODO implement the fixed presicion RSVD
    - stores the ROM basis in JSON format #TODO improve format
    """

    SnapshotsMatrixROM = Stage3_RunROM(parameters, model_file, rom_file)
    """
    Stage 3
    - launches the Reduced Order Model simulations for the same simulation parameters used for the FOM
    - stores the results in a distributed array
    """

    #compare_ROM_vs_FOM(SnapshotsMatrixROM, SnapshotsMatrix)




