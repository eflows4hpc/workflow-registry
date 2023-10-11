# Importing the Kratos Library
import KratosMultiphysics

# Import packages
import numpy as np
import json

# Import pycompss
from pycompss.api.parameter import *
from pycompss.api.software import software
from pycompss.api.data_transformation import *
from pycompss.api.api import compss_barrier

#library for passing arguments to the script from bash
from sys import argv

from dts import *
from utils import load_ROM_folder, prepare_files_cosim, update_simulation_data, write_model_metadata
from kratos_simulations import GetWorkflowROMParameters, GetSimulationsData

###############################################################################################################################################################################
SW_CATALOG = os.environ.get("SW_CATALOG","/software-catalog/packages")
working_path = argv[1]
rom_folder = argv[2]


@software(config_file = SW_CATALOG+"/kratos/run_cosim.json")
def ExecuteInstance_Task(serialized_parameters,sample,rom_folder=None):
    from kratos_simulations import TrainROM
    if rom_folder is not None:
        load_ROM_folder(rom_folder)
    current_parameters = KratosMultiphysics.Parameters()
    serialized_parameters.Load("ParametersSerialization",current_parameters)
    del(serialized_parameters)
    simulation = TrainROM(current_parameters,sample)
    simulation.Run()
    snapshots = simulation.GetSnapshotsMatrices()
    return snapshots[0], snapshots[1]



@software(config_file = SW_CATALOG+"/kratos/run_cosim.json")
def ExecuteInstance_TrainHROM_Task(serialized_parameters,sample, rom_folder):
    from kratos_simulations import TrainHROM
    if rom_folder is not None:
        load_ROM_folder(rom_folder)
    current_parameters = KratosMultiphysics.Parameters()
    serialized_parameters.Load("ParametersSerialization",current_parameters)
    del(serialized_parameters)
    # get sample
    simulation = TrainHROM(current_parameters,sample)
    simulation.Run()
    snapshots = simulation.GetSnapshotsMatrices()
    return snapshots[0], snapshots[1]


@software(config_file = SW_CATALOG+"/kratos/load_parameters.json")
def SerializeParameters_Task(parameter_file_name):
    with open(parameter_file_name,'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    serialized_parameters = KratosMultiphysics.StreamSerializer()
    serialized_parameters.Save("ParametersSerialization", parameters)
    return serialized_parameters

def FOM(working_path, parameters):
    # set the ProjectParameters.json path
    parameter_file_name = working_path + "/ProjectParameters_CoSimulation_workflow.json"
    # create a serialization of the model and of the project parameters
    pickled_parameters = SerializeParameters_Task(parameter_file_name)   #Can we launch the simulations without taking this into account??

    SnapshotsMatrix1 = []
    SnapshotsMatrix2 = []

    # start algorithm
    for sample in parameters:
        b1, b2 = ExecuteInstance_Task(pickled_parameters, sample)
        SnapshotsMatrix1.append(b1)
        SnapshotsMatrix2.append(b2)

    return [SnapshotsMatrix1, SnapshotsMatrix2]

@dt("SnapshotsMatrix", load_and_rechunk, is_workflow=True, number_cases="{{TotalNumberOfCases}}", simulation_data="{{simulation_data}}")
@software(config_file = SW_CATALOG + "/py-dislib/dislib.json")
def train_ROM(SnapshotsMatrix, simulation_data, workflow_rom_parameters, TotalNumberOfCases):
    from dislib_parallel_svd import rsvd, tsqr_svd
    name = simulation_data["solver_name"]
    """RSVD #OPTION 1"""
    #desired_rank = simulation_data["number_of_modes"]
    #u,_ = rsvd(SnapshotsMatrix, desired_rank)

    """TSRQ SVD #OPTION 2"""
    partitions = workflow_rom_parameters[name]["ROM"]["number_of_partitions"].GetInt()
    tolerance = workflow_rom_parameters[name]["ROM"]["svd_truncation_tolerance"].GetDouble()
    u = tsqr_svd(SnapshotsMatrix, partitions, tolerance)

    """LANCZOS SVD #OPTION 3"""
    #DISLIB_ARRAY-COMPATIBLE VERSION OF LANCZOS
    return u 

@dt(target="roms", function=serialize_ROM, type=COLLECTION_TO_DIRECTORY, destination=rom_folder, simulations_data="{{simulations_data}}", simulation_to_run="{{simulation_to_run}}")
@software(config_file = SW_CATALOG + "/compss/workflow.json")
def ROM(working_path, parameters, simulations_data, workflow_rom_parameters, simulation_to_run, roms):
    prepare_files_cosim(working_path, workflow_rom_parameters, simulation_to_run)

    # set the ProjectParameters.json path
    parameter_file_name = working_path + "/ProjectParameters_CoSimulation_workflow_ROM.json"

    # create a serialization of the model and of the project parameters
    pickled_parameters = SerializeParameters_Task(parameter_file_name)   #Can we launch the simulations without taking this into account?? Not doing any heavy lifting in CoSim

    SnapshotsMatrix1 = []
    SnapshotsMatrix2 = []

    # start algorithm
    for sample in parameters:
        b1, b2 = ExecuteInstance_Task(pickled_parameters,sample, roms)
        SnapshotsMatrix1.append(b1)
        SnapshotsMatrix2.append(b2)

    return [SnapshotsMatrix1, SnapshotsMatrix2]

@dt(target="hroms", function=serialize_HROM, type=COLLECTION_TO_DIRECTORY, destination=rom_folder, simulations_data="{{simulations_data}}", simulation_to_run="{{simulation_to_run}}", roms="{{roms}}")
@software(config_file = SW_CATALOG + "/compss/workflow.json")
def HROM(working_path, parameters, simulations_data, workflow_rom_parameters, simulation_to_run, hroms, roms):
    prepare_files_cosim(working_path, workflow_rom_parameters, simulation_to_run)

    # set the ProjectParameters.json path
    parameter_file_name = working_path + "/ProjectParameters_CoSimulation_workflow_ROM.json"

    # create a serialization of the model and of the project parameters
    pickled_parameters = SerializeParameters_Task(parameter_file_name)   #Can we launch the simulations without taking this into account?? Not doing any heavy lifting in CoSim

    SnapshotsMatrix1 = []
    SnapshotsMatrix2 = []

    # start algorithm
    for sample in parameters:
        b1, b2 = ExecuteInstance_Task(pickled_parameters,sample, hroms)
        SnapshotsMatrix1.append(b1)
        SnapshotsMatrix2.append(b2)

    return [SnapshotsMatrix1, SnapshotsMatrix2]

@dt(target="roms", function=serialize_ROM, type=COLLECTION_TO_DIRECTORY, destination=rom_folder, 
    simulations_data="{{simulations_data}}", simulation_to_run="trainHROMGalerkin")
@software(config_file = SW_CATALOG + "/compss/workflow.json")
def TrainHROM_simulations(roms, parameters, simulations_data, workflow_rom_parameters):
    prepare_files_cosim(working_path, workflow_rom_parameters, "trainHROM")
    # set the ProjectParameters.json path
    parameter_file_name = working_path + "/ProjectParameters_CoSimulation_workflow_ROM.json"

    # create a serialization of the model and of the project parameters
    pickled_parameters = SerializeParameters_Task(parameter_file_name)   #Can we launch the simulations without taking this into account??
   

    #bouble_blocks = []
    blocks1 = []
    blocks2 = []


    # start algorithm
    for sample in parameters:
        b1, b2 = ExecuteInstance_TrainHROM_Task(pickled_parameters,sample,roms)
        blocks1.append(b1)
        blocks2.append(b2)

    return [blocks1,blocks2]

@dt("SnapshotMatrix", load_and_rechunk, number_cases="{{TotalNumberOfCases}}", simulation_data="{{simulation_data}}", workflow_rom_parameters="{{workflow_rom_parameters}}", is_workflow=True)
@software(config_file = SW_CATALOG + "/py-dislib/dislib.json")
def train_HROM (SnapshotMatrix, simulation_data, workflow_rom_parameters, TotalNumberOfCases):
        from dislib_parallel_svd import tsqr_svd, Parallel_ECM, Initialize_ECM_Lists
        from kratos_simulations import SelectElements
        type_of_ecm = workflow_rom_parameters[simulation_data["solver_name"]]["HROM"]["empirical_cubature_type"].GetString()
        size = SnapshotMatrix._reg_shape[0]

        if type_of_ecm == "partitioned":
            ###Partitioned ECM:
            # Run ECM in recursion, with given tolerance in the final iteration.
            ecm_iterations = 2 # This should be obtained form the simulation parameters
            z,w = Initialize_ECM_Lists(SnapshotMatrix)
            for j in range(ecm_iterations):
                if j < ecm_iterations-1:
                    SnapshotMatrix,z,w = Parallel_ECM(SnapshotMatrix,size,z,w)
                else:
                    _,z,w = Parallel_ECM(SnapshotMatrix,size,z,w,final=True,
                final_truncation = workflow_rom_parameters[simulation_data["solver_name"]]["HROM"]["element_selection_svd_truncation_tolerance"].GetDouble())

        elif type_of_ecm == "monolithic":
            ###Monolithic ECM:
            NumberOfPartitions =  workflow_rom_parameters[simulation_data["solver_name"]]["HROM"]["number_of_partitions"].GetInt()
            #TSRQ SVD
            tolerance = workflow_rom_parameters[simulation_data["solver_name"]]["HROM"]["element_selection_svd_truncation_tolerance"].GetDouble()
            u = tsqr_svd(SnapshotMatrix,NumberOfPartitions, tolerance)
            z,w = SelectElements(u, False)
        return (z,w)

@dt("SnapshotsMatrixA", load_and_rechunk, is_workflow=True, number_cases="{{TotalNumberOfCases}}", 
    simulation_data="{{simulation_data}}")
@dt("SnapshotsMatrixB", load_and_rechunk, is_workflow=True, number_cases="{{TotalNumberOfCases}}", 
    simulation_data="{{simulation_data}}")
@software(config_file = SW_CATALOG + "/py-dislib/dislib.json")
def compare_matrices(SnapshotsMatrixA, SnapshotsMatrixB, simulation_data, workflow_rom_parameters, TotalNumberOfCases):
    import dislib as ds
    #using the Frobenious norm of the snapshots of the solution 
    original_norm = SnapshotsMatrixB.norm()
    intermediate = ds.data.matsubtract(SnapshotsMatrixA,SnapshotsMatrixB).norm() #(available on latest release)
    original_norm= np.linalg.norm(original_norm.collect())
    intermediate = np.linalg.norm(intermediate.collect())
    return intermediate/original_norm


if __name__ == '__main__':

    num_args=len(argv)
    if num_args < 3 or num_args > 4:
        print("ERROR: Incorrect parameters(" + str(num_args) + " ): Workflow.py <working_path> <output_rom_folder> [<heat flux parameters>]")
        exit(1)

    """
    We define the parameters for the simulation.
    In this case a single parameter is defined.
    More parameters are possible.
    """
    if num_args > 3:
        mu = json.loads(argv[3])
    else:
        mu = [[100000],[300000]]# default heat flux in the exterior    
    TotalNumberOfCases = len(mu)
   
    workflow_rom_parameters = GetWorkflowROMParameters()
    prepare_files_cosim(working_path, workflow_rom_parameters, "FOM")

    """
    Stage 0
    - The shape of the expected matrices is obtained to allow paralelization
    """    
    parameter_file_name = working_path + "/ProjectParameters_CoSimulation_workflow.json"
    simulations_data = GetSimulationsData(parameter_file_name)
    
    """
    Stage 1
    - launches in parallel a Full Order Model (FOM) simulation for each simulation parameter.
    - returns a distributed array (ds-array) with the results of the simulations.
    """
    SnapshotsMatricesFOM = FOM(working_path, mu)
    """
    Stage 2
    - computes the "fixed rank" randomized SVD in parallel using the dislib library
    - stores the ROM basis in JSON format #TODO improve format
    """
    roms=[]
    for i in range(len(SnapshotsMatricesFOM)):
        roms.append(train_ROM(SnapshotsMatricesFOM[i], simulations_data[i], workflow_rom_parameters, TotalNumberOfCases))
    
    update_simulation_data(simulations_data, roms, mu)
    
    """
    Stage 3
    - launches the Reduced Order Model simulations for the same simulation parameters used for the FOM
    - stores the results in a distributed array
    """
    SnapshotsMatricesROM = ROM(working_path, mu, simulations_data, workflow_rom_parameters, "ROM", roms)
    
    """
    Stage 4
    - Launches the same ROM simulation and builds the projected residual in distributed array
    - Analyses the matrix of projected residuals and obtains the elements and weights
    """
    SnapshotsMatricesTrainHROM = TrainHROM_simulations(roms, mu, simulations_data, workflow_rom_parameters)
    
    hroms=[]
    for i in range(len(SnapshotsMatricesTrainHROM)):
        hroms.append(train_HROM(SnapshotsMatricesTrainHROM[i], simulations_data[i], workflow_rom_parameters, TotalNumberOfCases))

    """
    Stage 5
    - launches the Hyper Reduced Order Model simulations for the same simulation parameters used for the FOM and ROM
    - stores the results in a distributed array
    """
    SnapshotsMatricesHROM = HROM(working_path, mu, simulations_data, workflow_rom_parameters, 'RunHROM', hroms, roms)
    
    """
    - Computes the Frobenius norm of the difference of Snapshots ROM and HROM
    - TODO add more parameters in a smart way if norm is above a given threshold
    """
    rom_vs_fom_errors=[]
    for i in range(len(SnapshotsMatricesFOM)):
        rom_vs_fom_errors.append(compare_matrices(SnapshotsMatricesFOM[i], SnapshotsMatricesROM[i], simulations_data[i], workflow_rom_parameters, TotalNumberOfCases))
    hrom_vs_fom_errors=[]
    for i in range(len(SnapshotsMatricesFOM)):
        hrom_vs_fom_errors.append(compare_matrices(SnapshotsMatricesFOM[i], SnapshotsMatricesHROM[i], simulations_data[i], workflow_rom_parameters, TotalNumberOfCases))
    hrom_vs_rom_errors=[]
    for i in range(len(SnapshotsMatricesROM)):
        hrom_vs_rom_errors.append(compare_matrices(SnapshotsMatricesROM[i], SnapshotsMatricesHROM[i], simulations_data[i], workflow_rom_parameters, TotalNumberOfCases))

    write_model_metadata(rom_vs_fom_errors, hrom_vs_fom_errors, hrom_vs_rom_errors, simulations_data, rom_folder)

