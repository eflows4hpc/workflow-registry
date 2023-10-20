# Importing the Kratos Library
import KratosMultiphysics
import sys
import time
import os

# Import packages
import numpy as np

# Import pickle for serialization
import pickle

from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis
from KratosMultiphysics.RomApplication.fluid_dynamics_analysis_rom import FluidDynamicsAnalysisROM
from KratosMultiphysics.CoSimulationApplication.co_simulation_analysis import CoSimulationAnalysis
from KratosMultiphysics.RomApplication.calculate_rom_basis_output_process import CalculateRomBasisOutputProcess
from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
from KratosMultiphysics.RomApplication.empirical_cubature_method import EmpiricalCubatureMethod

import json

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


###################################################################################################################################

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

class TrainROM(CoSimulationAnalysis):

    def __init__(self,cosim_parameters,sample):
        super().__init__(cosim_parameters)
        self.sample=sample


    def Initialize(self):
        for solver in self._solver.solver_wrappers.keys():
            if solver == 'outside':
                this_analysis_stage = self._solver._GetSolver(solver)._analysis_stage
                this_analysis_stage.project_parameters["processes"]["constraints_process_list"][0]["Parameters"]["value"].SetDouble(self.sample[0])
                this_analysis_stage.project_parameters["processes"]["constraints_process_list"][1]["Parameters"]["value"].SetDouble(self.sample[0])
                this_analysis_stage.project_parameters["processes"]["constraints_process_list"][2]["Parameters"]["value"].SetDouble(self.sample[0])
                #this_analysis_stage.project_parameters["processes"]["constraints_process_list"][3]["Parameters"]["value"].SetDouble(self.sample[0])
        super().Initialize()


    def GetSnapshotsMatrices(self):
        matrices = []
        for solver in self._solver.solver_wrappers.keys():
            for process in self._solver._GetSolver(solver)._analysis_stage._GetListOfOutputProcesses():
                if isinstance(process, CalculateRomBasisOutputProcess):
                    BasisOutputProcess = process
            matrices.append(BasisOutputProcess._GetSnapshotsMatrix())

        return matrices


class RetrieveDataForSimulations(CoSimulationAnalysis):

    def __init__(self, project_parameters, sample):
        super().__init__(project_parameters)
        self.velocity = sample[0]


    def Run(self):
        self.Initialize()
        self.time = self._GetSolver().AdvanceInTime(self.time)
        self.InitializeSolutionStep()
        self.Finalize()

    def GetData(self):

        cosim_simulation_data = []
        final_time = self.end_time
        for solver in self._solver.solver_wrappers.keys():
            simulations_data = {}
            simulations_data["solver_name"] = solver
            this_analysis_stage = self._solver._GetSolver(solver)._analysis_stage

            time_step_size = this_analysis_stage.project_parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()

            snapshots_per_simulation = int(final_time/time_step_size) # + 1 #TODO fix extra time step that sometimes appears due to rounding
            number_of_dofs = int(this_analysis_stage._GetSolver().GetComputingModelPart().NumberOfNodes()) #made for TEMPERATURE #TODO make it more robust
            number_of_elements = int(this_analysis_stage._GetSolver().GetComputingModelPart().NumberOfElements())
            number_of_conditions = int(this_analysis_stage._GetSolver().GetComputingModelPart().NumberOfConditions())
            number_of_modes = 30 #hard coded here #TODO incorporate in the workflow a parallel fixed presicion SVD

            simulations_data["snapshots_per_simulation"] = snapshots_per_simulation
            simulations_data["desired_block_size_svd_rom"] = int(number_of_dofs/10),snapshots_per_simulation #TODO automate definition of block size
            simulations_data["desired_block_size_svd_hrom"] = int((number_of_elements+number_of_conditions)/10) , snapshots_per_simulation+number_of_modes  #TODO automate definition of block size
            simulations_data["number_of_dofs"] = number_of_dofs
            simulations_data["number_of_elements"] = number_of_elements
            simulations_data["number_of_conditions"] = number_of_conditions
            simulations_data["number_of_modes"] = number_of_modes
            cosim_simulation_data.append(simulations_data)

        return cosim_simulation_data

class TrainHROM(CoSimulationAnalysis):

    def __init__(self,cosim_parameters,sample):
        super().__init__(cosim_parameters)
        self.sample = sample


    def Initialize(self):
        for solver in self._solver.solver_wrappers.keys():
            if solver == 'outside':
                this_analysis_stage = self._solver._GetSolver(solver)._analysis_stage
                this_analysis_stage.project_parameters["processes"]["constraints_process_list"][0]["Parameters"]["value"].SetDouble(self.sample[0])
                this_analysis_stage.project_parameters["processes"]["constraints_process_list"][1]["Parameters"]["value"].SetDouble(self.sample[0])
                this_analysis_stage.project_parameters["processes"]["constraints_process_list"][2]["Parameters"]["value"].SetDouble(self.sample[0])
                #this_analysis_stage.project_parameters["processes"]["constraints_process_list"][3]["Parameters"]["value"].SetDouble(self.sample[0])
        super().Initialize()


    def GetSnapshotsMatrices(self):
        residuals_projected = []
        for solver in self._solver.solver_wrappers.keys():
            this_analysis_stage = self._solver._GetSolver(solver)._analysis_stage
            residuals_projected.append(this_analysis_stage.GetHROM_utility()._GetResidualsProjectedMatrix())

        return residuals_projected


class CreateHROMModelParts(CoSimulationAnalysis):


    def __init__(self, project_parameters):
        super().__init__(project_parameters)


    def Run(self):
        self.Initialize()
        self.time = self._GetSolver().AdvanceInTime(self.time)
        self.InitializeSolutionStep()
        self.Finalize()


    def ComputeParts(self):
        for solver in self._solver.solver_wrappers.keys():
            this_analysis_stage = self._solver._GetSolver(solver)._analysis_stage
            this_analysis_stage.GetHROM_utility().hyper_reduction_element_selector.w = np.load(this_analysis_stage.GetHROM_utility().rom_basis_output_folder / 'aux_w.npy')
            this_analysis_stage.GetHROM_utility().hyper_reduction_element_selector.z = np.load(this_analysis_stage.GetHROM_utility().rom_basis_output_folder / 'aux_z.npy')
            this_analysis_stage.GetHROM_utility().AppendHRomWeightsToRomParameters()
            this_analysis_stage.GetHROM_utility().CreateHRomModelParts()

def GetWorkflowROMParameters():

    workflow_rom_parameters = KratosMultiphysics.Parameters("""{
            "outside":{
                "ROM":{
                    "svd_truncation_tolerance": 1e-6,
                    "model_part_name": "ThermalModelPart",
                    "nodal_unknowns": ["TEMPERATURE"],
                    "number_of_partitions":  10
                },
                "HROM":{
                    "number_of_partitions":  4,
                    "empirical_cubature_type": "monolithic",
                    "element_selection_svd_truncation_tolerance": 1e-8,
                    "include_conditions_model_parts_list": ["ThermalModelPart.GENERIC_Interface_outside"],
                    "include_nodal_neighbouring_elements_model_parts_list": ["ThermalModelPart.GENERIC_Interface_outside"],
                    "include_elements_model_parts_list": []
                }
            },
            "center":{
                "ROM":{
                    "svd_truncation_tolerance": 1e-6,
                    "model_part_name": "ThermalModelPart",
                    "nodal_unknowns": ["TEMPERATURE"],
                    "number_of_partitions":  10
                },
                "HROM":{
                    "number_of_partitions":  4,
                    "empirical_cubature_type": "partitioned",
                    "element_selection_svd_truncation_tolerance": 1e-8,
                    "include_conditions_model_parts_list": ["ThermalModelPart.GENERIC_Interface_center"],
                    "include_nodal_neighbouring_elements_model_parts_list": ["ThermalModelPart.GENERIC_Interface_center"],
                    "include_elements_model_parts_list": []
                }
            }
        }""")

    return workflow_rom_parameters

def GetSimulationsData(parameter_file_name):
    with open(parameter_file_name,'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    fake_sample = [5]
    simulation = RetrieveDataForSimulations(parameters,fake_sample)
    simulation.Run()
    return simulation.GetData()

def CalculateAndSelectElements(projected_residuals_matrix, title, final_truncation = 1e-6):
    
    if title == 'intermediate':
        u, _, _, _ = RandomizedSingularValueDecomposition().Calculate(projected_residuals_matrix) #randomized version with machine precision
        constrain_sum_of_weights = False # setting it to "True" worsens the approximation. Need to implement the orthogonal complement rather and not the row of 1's is implemented
    else:
        u, _, _, _ = RandomizedSingularValueDecomposition().Calculate(projected_residuals_matrix,final_truncation) #randomized version with user-defined tolerance
        constrain_sum_of_weights = False # setting it to "True" worsens the approximation. Need to implement the orthogonal complement rather and not the row of 1's is implemented
    return SelectElements(u, constrain_sum_of_weights)

def SelectElements(u, constrain_sum_of_weights):
    ElementSelector = EmpiricalCubatureMethod()
    ElementSelector.SetUp( u, constrain_sum_of_weights)
    ElementSelector.Initialize()
    ElementSelector.Calculate()
    local_ids = np.squeeze(ElementSelector.z)
    weights = np.squeeze(ElementSelector.w)
    return local_ids,weights
