
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
    print('\n\nNodal basis printed in json format\n\n')

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


