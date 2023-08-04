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
