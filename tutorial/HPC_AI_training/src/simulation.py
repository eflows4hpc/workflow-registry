from pycompss.api.task import task
from pycompss.api.mpi import mpi
from pycompss.api.parameter import *
from utils import build_variables, create_env_simulations
import os
import yaml

ALYA_PROCS = int(os.environ.get("ALYA_PROCS", 16))
ALYA_PPN = int(os.environ.get("ALYA_PPN", 16))
if (ALYA_PROCS < ALYA_PPN) :
    ALYA_PPN = ALYA_PROCS
ALYA_TIMEOUT = int(os.environ.get("ALYA_TIMEOUT", 3600))

@task(returns=1, simulation_wdir=DIRECTORY_OUT)
def prepare_alya(values, sampling_parameters, simulation_parameters, simulation_wdir, original_name, name_sim):
    mesh_dir = simulation_parameters.get("mesh_directory")
    template = simulation_parameters.get("template_sld")
    variables = build_variables(values, sampling_parameters)
    os.makedirs(simulation_wdir, exist_ok=True)
    create_env_simulations(mesh_dir, simulation_wdir, original_name, name_sim)
    simulation = os.path.join(simulation_wdir, name_sim + ".sld.dat")
    with open(simulation, 'w') as f2:
        with open(template, 'r') as f:
            filedata = f.read()
            for i in range(len(variables)):
                item = variables[i]
                for name, bound in item.items():
                    filedata = filedata.replace("%" + name + "%", str(bound))
            f2.write(filedata)
            f.close()
        f2.close()
    template = simulation_parameters.get("template_dom")
    simulation = os.path.join(simulation_wdir, name_sim + ".dom.dat")
    with open(simulation, 'w') as f2:
        with open(template, 'r') as f:
            filedata = f.read()
            filedata = filedata.replace("%sim_num%", str(name_sim))
            filedata = filedata.replace("%data_folder%", str(mesh_dir))
            f2.write(filedata)
            f.close()
        f2.close()

@task(returns=1, simulation_wdir=DIRECTORY_IN)
def collect_alya_results(simulation_wdir, name_sim, **kwargs):
    y = 0
    path = os.path.join(simulation_wdir, name_sim + "-output.sld.yaml")
    try:
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            if data:
                variables = data.get("variables")
                if variables:
                    y = variables.get("FRXID", -1)
                else:
                    # If variables don't exist, return 0
                    return 0
            else:
                # If file is empty, return 0
                return 0
    except FileNotFoundError:
        # If file not found, return -1
        return -1
    except Exception as e:
        print("Error:", e)
        return 0
    return y


@task(y=COLLECTION_IN, x_file=FILE_OUT, y_file=FILE_OUT)
def write_results(y, sample_set, x_file, y_file):
    with open(x_file, 'wb') as f:
        write(sample_set)
        f.close()
    with open(y_file, 'wb') as f:
        write(y)
        f.close()

@mpi(runner="$ALYA_RUNNER", binary="$ALYA_BIN", args="{{name}}", processes=ALYA_PROCS, processes_per_node=ALYA_PPN, working_dir="{{wdir}}")
@task(wdir=DIRECTORY_INOUT, time_out=ALYA_TIMEOUT)
def alya_simulation(wdir, name, **kwargs):
    pass

