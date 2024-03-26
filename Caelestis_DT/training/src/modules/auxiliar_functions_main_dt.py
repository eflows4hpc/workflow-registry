import os
import yaml
from pycompss.api.task import task
from pycompss.api.mpi import mpi
from modules.alya_file import alya_parser, parser_dom

@task(returns=1)
def generator_arguments(input_yaml, data_folder):
    mesh_source, templateSld, templateDom = get_input(input_yaml, data_folder)  # Encapsular estas funciones,
    # dentro de generate parameters simulation, tal vez falta un método, pensar como sería más adecuado generalizarlo
    parent_directory, original_name = os.path.split(mesh_source)
    simulation_wdir = "/home/bsc19/bsc19756/My_DT_Caelestis/DigitalTwin" + "/SIMULATIONS/" + original_name + "-s"
    nameSim = original_name + "-s"
    return {"mesh_source": mesh_source, "templateSld": templateSld, "simulation_wdir": simulation_wdir,
                            "original_name": original_name, "nameSim": nameSim, "templateDom": templateDom}


def get_input(input_yaml, data_folder):
    mesh = input_yaml.get("mesh", [])
    mesh_folder = ""
    for item in mesh:
        if isinstance(item, dict) and 'path' in item:
            mesh_folder = item['path']
            break  # Exit the loop after finding the first 'folder'

    template_sld = input_yaml.get("template_sld", [])
    templateSld_folder = ""
    for item in template_sld:
        if isinstance(item, dict) and 'path' in item:
            templateSld_folder = item['path']
            break  # Exit the loop after finding the first 'folder'

    template_dom = input_yaml.get("template_dom", [])
    templateDom_folder = ""
    for item in template_dom:
        if isinstance(item, dict) and 'path' in item:
            templateDom_folder = item['path']
            break  # Exit the loop after finding the first 'folder'

    # Now use these folder paths as needed
    mesh_source = os.path.join(data_folder, mesh_folder) if mesh_folder else None
    templateSld = os.path.join(data_folder, templateSld_folder) if templateSld_folder else None
    templateDom = os.path.join(data_folder, templateDom_folder) if templateDom_folder else None
    return mesh_source, templateSld, templateDom

@task(returns=1)
def collect_results(wdir, nameSim, out):
    y = 0
    path = wdir + "/" + nameSim + "-output.sld.yaml"
    try:
        f = open(path)
        data = yaml.load(f, Loader=yaml.FullLoader)
        variables = data.get("variables")
        y = variables.get("FRXID")
    except Exception as e:
        print("NOT FINDING THE RESULT FILE OF ALYA")
        return 0
    return y

@mpi(runner="mpirun", binary="$ALYA_BIN", args="{{name}} ", processes="$ALYA_PROCS", processes_per_node="$ALYA_PPN", working_dir="{{wdir}}")
@task(returns=1, time_out=3600)
def simulation(wdir, name, **kwargs):
        return

def simulation_workflow(variables_sld, execution_number=None, **kwargs):
    mesh_source = kwargs["mesh_source"]
    templateSld = kwargs["templateSld"]
    simulation_wdir = kwargs["simulation_wdir"]
    if execution_number is not None:
        simulation_wdir = simulation_wdir + str(execution_number) + "/"
    if not os.path.isdir(simulation_wdir):
        os.makedirs(simulation_wdir)
    original_name = kwargs["original_name"]
    nameSim = kwargs["nameSim"]
    if execution_number is not None:
        nameSim = nameSim + str(execution_number)
    templateDom = kwargs["templateDom"]
    output_alya_parser = alya_parser(variables_sld, mesh_source, templateSld, simulation_wdir, original_name, nameSim)
    out_ = parser_dom(simulation_wdir, nameSim, templateDom, mesh_source, output_alya_parser)
    out = simulation(simulation_wdir, nameSim, out3=out_)
    y_output = collect_results(simulation_wdir, nameSim, out)
    return y_output
