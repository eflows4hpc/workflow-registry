import os
import shutil
from pycompss.api.task import task
from pycompss.api.parameter import (DIRECTORY_OUT, DIRECTORY_INOUT)

@task(returns=1)
def alya_parser(variables, wdir, template, simulation_wdir, original_name, nameSim):
    create_env_simulations(wdir, simulation_wdir, original_name, nameSim)
    simulation = simulation_wdir+"/"+nameSim + ".sld.dat"
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
    return

@task(returns=1)
def parser_fie(variables, template, simulation_wdir, nameSim, wdir, original_name, out):
    simulation = simulation_wdir+"/"+nameSim + ".fie.dat"
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
    return

@task(returns=1)
def parser_dom(simulation_wdir, nameSim, template, mesh_source, out):
    simulation = simulation_wdir+"/"+nameSim + ".dom.dat"
    with open(simulation, 'w') as f2:
        with open(template, 'r') as f:
            filedata = f.read()
            filedata = filedata.replace("%sim_num%", str(nameSim))
            filedata = filedata.replace("%data_folder%", str(mesh_source))
            f2.write(filedata)
            f.close()
        f2.close()
    return

def copy(src_dir, src_name, tgt_dir, tgt_name):
    src_file=os.path.join(src_dir, src_name)
    tgt_file=os.path.join(tgt_dir, tgt_name)
    shutil.copyfile(src_file, tgt_file)
    return

def create_env_simulations(wdir,sim_dir, original_name, nameSim):
    copy(wdir, original_name + ".ker.dat", sim_dir, nameSim+ ".ker.dat")
    copy(wdir, original_name + ".dat", sim_dir, nameSim + ".dat")
    copy(wdir, original_name + ".dom.dat", sim_dir, nameSim+ ".dom.dat")
    copy(wdir, original_name + ".fie.dat", sim_dir, nameSim + ".fie.dat")
    copy(wdir, original_name + ".post.alyadat", sim_dir, nameSim+ ".post.alyadat")
    return