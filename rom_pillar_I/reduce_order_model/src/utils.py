import json
import numpy as np
import os
import sys

from pycompss.api.api import compss_delete_file, compss_delete_object, compss_wait_on_directory

def get_rom_output_defaults():
    defaults={
            "python_module": "calculate_rom_basis_output_process",
            "kratos_module": "KratosMultiphysics.RomApplication",
            "process_name": "rom_output",
            "Parameters": {
                "help": "A process to set the snapshots matrix and calculate the ROM basis from it.",
                "model_part_name": "",
                "rom_manager" : True,
                "snapshots_control_type": "step",
                "snapshots_interval": 1.0,
                "nodal_unknowns": [],
                "rom_basis_output_format": "numpy",
                "rom_basis_output_name": "RomParameters",
                "rom_basis_output_folder" : "rom_data",
                "svd_truncation_tolerance": 1.0e-6
            }
    }
    return defaults

def prepare_files_physical_problem(working_path, physics_project_parameters_name, solver, simulation_to_run, workflow_rom_parameters):
    """pre-pending the absolut path of the files in the Project Parameters"""
    with open(f'{physics_project_parameters_name}.json','r') as f:
        updated_project_parameters = json.load(f)
        file_input_name = updated_project_parameters["solver_settings"]["model_import_settings"]["input_filename"]
        materials_filename = updated_project_parameters["solver_settings"]["material_import_settings"]["materials_filename"]
        updated_project_parameters["output_processes"]["rom_output"] = [get_rom_output_defaults()]
        updated_project_parameters["output_processes"]["rom_output"][0]["Parameters"]["model_part_name"] = workflow_rom_parameters[solver]["ROM"]["model_part_name"].GetString()
        updated_project_parameters["output_processes"]["rom_output"][0]["Parameters"]["nodal_unknowns"] = workflow_rom_parameters[solver]["ROM"]["nodal_unknowns"].GetStringArray()
        updated_project_parameters["output_processes"]["rom_output"][0]["Parameters"]["rom_basis_output_folder"] =  "rom_data/" + solver
        if simulation_to_run=="FOM":
            updated_project_parameters["solver_settings"]["model_import_settings"]["input_filename"] = working_path + '/'+ file_input_name
            updated_project_parameters["solver_settings"]["material_import_settings"]["materials_filename"] = working_path +'/'+ materials_filename
            with open(f'{physics_project_parameters_name}_workflow.json','w') as f:
                json.dump(updated_project_parameters, f, indent = 4)
        else:
            if simulation_to_run=="trainHROM":
                updated_project_parameters["output_processes"]["rom_output"][0]["Parameters"]["snapshots_interval"] = 1e6

            if simulation_to_run=="HHROM":
                updated_project_parameters["solver_settings"]["model_import_settings"]["input_filename"] = file_input_name+"HROM"
            with open(f'{physics_project_parameters_name}.json','w') as f:
                json.dump(updated_project_parameters, f, indent = 4)

def prepare_files_cosim(working_path, workflow_rom_parameters, simulation_to_run):
    """pre-pending the absolut path of the files in the Project Parameters"""

    if simulation_to_run == "FOM":
        original_project_parameter= working_path + '/ProjectParameters_CoSimulation.json'
        new_project_parameter= working_path + '/ProjectParameters_CoSimulation_workflow.json'
        compss_delete_file(new_project_parameter)
        with open(original_project_parameter,'r') as f:
            updated_project_parameters = json.load(f)
            solver_keys = updated_project_parameters["solver_settings"]["solvers"].keys()
            for solver in solver_keys:
                file_input_name = updated_project_parameters["solver_settings"]["solvers"][solver]["solver_wrapper_settings"]["input_file"]
                prepare_files_physical_problem(working_path, working_path + '/' + file_input_name, solver, simulation_to_run, workflow_rom_parameters)
                updated_project_parameters["solver_settings"]["solvers"][solver]["solver_wrapper_settings"]["input_file"] = working_path + '/'+ file_input_name + '_workflow'

        with open(new_project_parameter,'w') as f:
            json.dump(updated_project_parameters, f, indent = 4)

    else:
        original_project_parameter= working_path + '/ProjectParameters_CoSimulation_workflow.json'
        with open(original_project_parameter,'r') as f:
            updated_project_parameters = json.load(f)
            solver_keys = updated_project_parameters["solver_settings"]["solvers"].keys()
            for solver in solver_keys:
                file_input_name = updated_project_parameters["solver_settings"]["solvers"][solver]["solver_wrapper_settings"]["input_file"]
                prepare_files_physical_problem(working_path, file_input_name, solver, simulation_to_run, workflow_rom_parameters)
                updated_project_parameters["solver_settings"]["solvers"][solver]["type"] = 'solver_wrappers.kratos.rom_wrapper'

        if simulation_to_run == "ROM":
            new_project_parameter= working_path + '/ProjectParameters_CoSimulation_workflow_ROM.json'
            compss_delete_file(new_project_parameter)
            with open(new_project_parameter,'w') as f:
                json.dump(updated_project_parameters, f, indent = 4)

def update_simulation_data(simulations_data, rom, mu):
    for i in range(len(simulations_data)):
        compss_delete_object(simulations_data[i])
        snapshots_per_simulation = simulations_data[i]["snapshots_per_simulation"]
        simulations_data[i]["number_of_modes"] = int(rom[i].shape[1])
        size, _ = simulations_data[i]["desired_block_size_svd_hrom"]
        simulations_data[i]["desired_block_size_svd_hrom"] = size,int(simulations_data[i]["number_of_modes"]*len(mu)*snapshots_per_simulation)



def write_model_metadata(rom_vs_fom_errors, hrom_vs_fom_errors, hrom_vs_rom_errors, simulations_data, rom_folder):
    compss_wait_on_directory(rom_folder)
    for i in range(len(simulations_data)):
        name = simulations_data[i]["solver_name"]
        folder = f'{rom_folder}/{name}'
        attrs = {"name": f'Kratos_model_tutorial_experiment_{name}',
                     "params": {"snapshots_per_simulation": simulations_data[i]["snapshots_per_simulation"], 
                                "number_of_modes": simulations_data[i]["number_of_modes"]},
                     "metrics": {"rom_error": rom_vs_fom_errors[i], "hrom_error": hrom_vs_fom_errors[i]},
                     "artifacts": [f'{folder}/RightBasisMatrix.npy', f'{folder}/NodeIds.npy', 
                                   f'{folder}/aux_w.npy', f'{folder}/aux_z.npy', 
                                   f'{folder}/HROM_ElementWeights.npy', f'{folder}/HROM_ConditionWeights.npy',
                                   f'{folder}/HROM_ElementIds.npy', f'{folder}/HROM_ConditionIds.npy', 
                                   f'{folder}/RomParameters.json']}

        with open(f'{folder}/attrs.json','w') as f:
            json.dump(attrs, f, indent = 4)

def load_ROM_folder(rom_folder):
    working_dir = os.getcwd()
    if not os.path.exists('rom_data'):
        print("Creating a symlink")
        try:
            os.symlink(rom_folder, 'rom_data')
        except:
            print("Ignoring exception in symlink creation")
    else:
        print("ROM already loaded")
        

