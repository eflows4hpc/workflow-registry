import os
import sys
import numpy as np
import json
from dislib.data.array import Array

def ROM_file_generation(rom, rom_file_name):
    u = np.block(rom)
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

def load_blocks_array(blocks, shape, block_size):
    if shape[0] < block_size[0] or  shape[1] < block_size[1]:
        raise ValueError("The block size is greater than the ds-array")
    return Array(blocks, shape=shape, top_left_shape=block_size,
                     reg_shape=block_size, sparse=False, delete=False)

def serialize_ROM(roms, rom_folder, simulation_to_run, simulations_data):
    from kratos_simulations import GetWorkflowROMParameters
    workflow_rom_parameters = GetWorkflowROMParameters()
    for i in range(len(roms)):
        name = simulations_data[i]["solver_name"]
        conditions_list = workflow_rom_parameters[name]["HROM"]["include_conditions_model_parts_list"].GetStringArray()
        nodal_neighbours_list = workflow_rom_parameters[name]["HROM"]["include_nodal_neighbouring_elements_model_parts_list"].GetStringArray()
        _SavingRomParameters(rom_folder, roms[i], name,conditions_list, nodal_neighbours_list)
    if simulation_to_run=="RunHROM" or simulation_to_run=="HHROM":
        ChangeRomFlags(rom_folder, simulations_data, 'runHROMGalerkin')
    elif simulation_to_run == "trainHROMGalerkin":
        ChangeRomFlags(rom_folder, simulations_data, 'trainHROMGalerkin')

def serialize_HROM(hroms, rom_folder, simulations_data, simulation_to_run, roms):
    serialize_ROM(roms, rom_folder, simulation_to_run, simulations_data)
    for i in range(len(hroms)):
        number_of_elements = simulations_data[i]["number_of_elements"]
        folder = rom_folder + '/' + simulations_data[i]["solver_name"]
        z,w = hroms[i]
        _SavingElementsAndWeights(folder, number_of_elements,z,w)
  
def ChangeRomFlags(rom_folder, simulations_data, simulation_to_run = 'trainHROMGalerkin'):
    folders = [simulations_data[0]["solver_name"], simulations_data[1]["solver_name"]]
    for folder in folders:
        parameters_file_name = f'{rom_folder}/{folder}/RomParameters.json'
        with open(parameters_file_name, 'r+') as parameter_file:
            f=json.load(parameter_file)
            f['assembling_strategy'] = 'global'
            if simulation_to_run=='GalerkinROM':
                f['projection_strategy']="galerkin"
                f['train_hrom']=False
                f['run_hrom']=False
            elif simulation_to_run=='trainHROMGalerkin':
                f['train_hrom']=True
                f['run_hrom']=False
            elif simulation_to_run=='runHROMGalerkin':
                f['projection_strategy']="galerkin"
                f['train_hrom']=False
                f['run_hrom']=True
            else:
                raise Exception(f'Unknown flag "{simulation_to_run}" change for RomParameters.json')
            parameter_file.seek(0)
            json.dump(f,parameter_file,indent=4)
            parameter_file.truncate()

def _SavingElementsAndWeights(working_path,number_of_elements,z,w):
    weights = np.squeeze(w)
    indexes = z
    element_indexes = np.where( indexes < number_of_elements )[0]
    condition_indexes = np.where( indexes >= number_of_elements )[0]
    np.save(working_path+'/aux_w.npy',weights)
    np.save(working_path+'/aux_z.npy',indexes)
    np.save(working_path+'/HROM_ElementWeights.npy',weights[element_indexes])
    np.save(working_path+'/HROM_ConditionWeights.npy',weights[condition_indexes])
    np.save(working_path+'/HROM_ElementIds.npy',indexes[element_indexes]) #FIXME fix the -1 in the indexes of numpy and ids of Kratos
    np.save(working_path+'/HROM_ConditionIds.npy',indexes[condition_indexes]-number_of_elements) #FIXME fix the -1 in the indexes of numpy and ids of Kratos

def _SavingRomParameters(working_path,u,name,conditions_list,nodal_neighbours_list):
    import KratosMultiphysics
    from pathlib import Path
    rom_basis_dict = {
        "rom_manager" : True,
        "train_hrom": False,
        "run_hrom": False,
        "projection_strategy": "galerkin",
        "assembling_strategy": "global",
        "rom_format": "numpy",
        "train_petrov_galerkin": {
            "train": False,
            "basis_strategy": "residuals",
            "include_phi": False,
            "svd_truncation_tolerance": 1e-6
        },
        "rom_settings": {},
        "hrom_settings": {},
        "nodal_modes": {},
        "elements_and_weights" : {}
    }

    nodal_unknowns = ["TEMPERATURE"]#RomParams["nodal_unknowns"].GetStringArray()
    rom_basis_dict["rom_basis_output_folder"] = "rom_data" #RomParams["rom_basis_output_folder"]
    rom_basis_dict["hrom_settings"]["hrom_format"] = "numpy"
    rom_basis_dict["hrom_settings"]["include_conditions_model_parts_list"] = conditions_list
    rom_basis_dict["hrom_settings"]["include_nodal_neighbouring_elements_model_parts_list"] = nodal_neighbours_list
    rom_basis_dict["hrom_settings"]["create_hrom_visualization_model_part"] = False
    rom_basis_dict["hrom_settings"]["include_elements_model_parts_list"] = []
    rom_basis_dict["hrom_settings"]["include_minimum_condition"] = False
    rom_basis_dict["hrom_settings"]["include_condition_parents"] = True
    n_nodal_unknowns = len(nodal_unknowns)
    snapshot_variables_list = []
    for var_name in nodal_unknowns:
        if not KratosMultiphysics.KratosGlobals.HasVariable(var_name):
            err_msg = "\'{}\' variable in \'nodal_unknowns\' is not in KratosGlobals. Please check provided value.".format(var_name)
        if not KratosMultiphysics.KratosGlobals.GetVariableType(var_name):
            err_msg = "\'{}\' variable in \'nodal_unknowns\' is not double type. Please check provide double type variables (e.g. [\"DISPLACEMENT_X\",\"DISPLACEMENT_Y\"]).".format(var_name)
        snapshot_variables_list.append(KratosMultiphysics.KratosGlobals.GetVariable(var_name))


    # Save the nodal basis
    rom_basis_dict["rom_settings"]["nodal_unknowns"] = [var.Name() for var in snapshot_variables_list]
    rom_basis_dict["rom_settings"]["number_of_rom_dofs"] = np.shape(u)[1] #TODO: This is way misleading. I'd call it number_of_basis_modes or number_of_rom_modes
    #rom_basis_dict["rom_settings"]["rom_bns_settings"] = {} required in the latest release
    rom_basis_dict["projection_strategy"] = "galerkin" # Galerkin: (Phi.T@K@Phi dq= Phi.T@b), LSPG = (K@Phi dq= b), Petrov-Galerkin = (Psi.T@K@Phi dq = Psi.T@b)
    rom_basis_dict["assembling_strategy"] = "global" # Assemble the ROM globally or element by element: "global" (Phi_g @ J_g @ Phi_g), "element by element" sum(Phi_e^T @ K_e @ Phi_e)
    rom_basis_dict["rom_settings"]["petrov_galerkin_number_of_rom_dofs"] = 0
    rom_basis_output_folder = Path(working_path + '/' + name)  #rom_basis_dict["rom_basis_output_folder"] + '_' + #TODO use the full path and not only the name of the sover!!
    if not rom_basis_output_folder.exists():
        rom_basis_output_folder.mkdir(parents=True)

    # Storing modes in Numpy format
    np.save(rom_basis_output_folder / "RightBasisMatrix.npy", u)
    np.save(rom_basis_output_folder / "NodeIds.npy", np.arange(1,((u.shape[0]+1)/n_nodal_unknowns), 1, dtype=int))

    # Creating the ROM JSON file containing or not the modes depending on "self.rom_basis_output_format"

    output_filename = rom_basis_output_folder / "RomParameters.json"
    with output_filename.open('w') as f:
        json.dump(rom_basis_dict, f, indent = 4)






def load_and_rechunk(blocks, number_cases, simulation_data, workflow_rom_parameters=None):
    if workflow_rom_parameters is not None:
        number_of_modes = simulation_data["number_of_modes"] # I added a 1 here to refer to the outside modelpart
        number_of_elements = simulation_data["number_of_elements"]
        number_of_conditions = simulation_data["number_of_conditions"]
        snapshots_per_simulation = simulation_data["snapshots_per_simulation"]
        expected_shape = (number_of_elements+number_of_conditions, number_of_modes*number_cases*snapshots_per_simulation) #We will know the size of the array!
        desired_block_size = simulation_data["desired_block_size_svd_hrom"]
        simulation_shape = (number_of_elements+number_of_conditions, number_of_modes*snapshots_per_simulation)
        NumberOfPartitions =  workflow_rom_parameters[simulation_data["solver_name"]]["HROM"]["number_of_partitions"].GetInt()
        desired_block_size = (int(np.ceil(expected_shape[0]/NumberOfPartitions)),expected_shape[1])
        return load_blocks_rechunk(blocks, shape = expected_shape, block_size = simulation_shape, new_block_size = desired_block_size)
    else:
        number_of_dofs = simulation_data["number_of_dofs"]
        snapshots_per_simulation = simulation_data["snapshots_per_simulation"]
        expected_shape = (number_of_dofs, number_cases*snapshots_per_simulation)  # We will know the size of the array!
        desired_block_size = simulation_data["desired_block_size_svd_rom"]
        simulation_shape = (number_of_dofs, snapshots_per_simulation)
        return load_blocks_rechunk(blocks, shape = expected_shape, block_size = simulation_shape, new_block_size = desired_block_size)



def load_blocks_rechunk(blocks, shape, block_size, new_block_size):
    if shape[0] < new_block_size[0] or  shape[1] < new_block_size[1]:
        raise ValueError("The block size requested for rechunk"
                         "is greater than the ds-array")
    final_blocks = [[]]
    for block in blocks:
        final_blocks[0].append(block)
    arr = load_blocks_array(final_blocks, shape, block_size)
    return arr.rechunk(new_block_size)

def load_ROM(rom_file):
    working_dir = os.environ["COMPSS_WORKING_DIR"]
    if not os.path.exists('RomParameters.json'):
        print("Creating a symlink")
        try:
            os.symlink(rom_file, 'RomParameters.json')
        except:
            print("Ignoring exception in symlink creation")
    else:
        print("ROM already loaded")

def replace_template(template_file, parameter_file_name, keyword, replacement):
    fin = open(template_file, "rt")
    fout = open(parameter_file_name, "wt")
    for line in fin:
        fout.write(line.replace(keyword, replacement))
    fin.close()
    fout.close()


