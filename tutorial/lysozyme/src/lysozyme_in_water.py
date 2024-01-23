"""
GROMACS Tutorial with PyCOMPSs
Lysozyme in Water

This example will guide a new user through the process of setting up a
simulation system containing a set of proteins (lysozymes) in boxes of water,
with ions. Each step will contain an explanation of input and output,
using typical settings for general use.

Extracted from: http://www.mdtutorials.com/gmx/lysozyme/index.html
Originally done by: Justin A. Lemkul, Ph.D.
From: Virginia Tech Department of Biochemistry

This example reaches up to stage 4 (energy minimization) and includes resulting
images merge.
"""
import os
from os import listdir
from os.path import isfile, join
import sys
from time import time

from pycompss.api.constraint import constraint

from pycompss.api.task import task
from pycompss.api.binary import binary
from pycompss.api.container import container
from pycompss.api.api import compss_wait_on, compss_barrier
from pycompss.api.api import compss_open
from pycompss.api.parameter import *

cu=os.environ['MDCU']

# ############ #
# Step 1 tasks #
# ############ #

@binary(binary="$GMX_BIN", args="pdb2gmx -f {{protein}} -o {{structure}} -p {{topology}} -ff {{forcefield}} -water {{water}} {{flags}}")
@task(protein=FILE_IN, structure=FILE_OUT, topology=FILE_OUT)
def generate_topology(protein=None, structure=None, topology=None, flags='-ignh', forcefield='oplsaa', water='spce'):
    # Command: gmx pdb2gmx -f protein.pdb -o structure.gro -p topology.top -ignh -ff oplsaa -water spce
    pass

# ############ #
# Step 2 tasks #
# ############ #

@binary(binary="$GMX_BIN", args="editconf -f {{structure}} -o {{structure_newbox}} -d {{distance}} -bt {{boxtype}} {{flags}}")
@task(structure=FILE_IN, structure_newbox=FILE_OUT)
def define_box(structure=None, structure_newbox=None, distance='1.0', boxtype='cubic', flags='-c'):
    # Command: gmx editconf -f structure.gro -o structure_newbox.gro -d 1.0 -bt cubic -c 
    pass

# ############ #
# Step 3 tasks #
# ############ #

@binary(binary="$GMX_BIN", args="solvate -cp {{structure_newbox}} -cs {{configuration_solvent}} -o {{protein_solv}} -p {{topology}}" )
@task(structure_newbox=FILE_IN, protein_solv=FILE_OUT, topology=FILE_IN)
def add_solvate(structure_newbox=None, configuration_solvent='spc216.gro', protein_solv=None, topology=None):
    # Command: gmx solvate -cp structure_newbox.gro -cs spc216.gro -o protein_solv.gro -p topology.top
    pass

# ############ #
# Step 4 tasks #
# ############ #

@binary(binary="$GMX_BIN", args="grompp -f {{conf}} -c {{protein_solv}} -p {{topology}} -o {{output}}")
@task(conf=FILE_IN, protein_solv=FILE_IN, topology=FILE_IN, output=FILE_OUT)
def assemble_tpr(conf=None, protein_solv=None, topology=None, output=None):
    # Command: gmx grompp -f ions.mdp -c protein_solv.gro -p topology.top -o ions.tpr
    pass

@binary(binary="$GMX_BIN", args="genion -s {{ions}} -o {{output}} -p {{topology}} -pname {{pname}} -nname {{nname}} {{flags}}")
@task(ions=FILE_IN, output=FILE_OUT, topology=FILE_IN, group={Type:FILE_IN, StdIOStream:STDIN})
def replace_solvent_with_ions(ions=None, output=None, topology=None, pname='NA', nname='CL',flags='-neutral', group=None):
    # Command: gmx genion -s ions.tpr -o 1AKI_solv_ions.gro -p topol.top -pname NA -nname CL -neutral < ../config/genion.group
    pass

# ############ #
# Step 5 tasks #
# ############ #

@constraint(computing_units=cu)
@binary(binary="$GMX_BIN", args = "mdrun {{flags}} -s {{em}} -e {{em_energy}}" )
@task(em=FILE_IN, em_energy=FILE_OUT)
def energy_minimization(mode='mdrun',flags='-v', em=None, em_energy=None):
    # Command: gmx mdrun -v -s em.tpr
    pass

# ############ #
# Step 6 tasks #
# ############ #

@binary(binary="$GMX_BIN", args = "energy -f {{em}} -o {{output}}" )
@task(em=FILE_IN, output=FILE_OUT, selection={Type:FILE_IN, StdIOStream:STDIN})
def energy_analisis(em=None, output=None, selection=None):
    # Command: gmx energy -f em.edr -o output.xvg
    pass

@binary(binary='grace', args= "-nxy {{xvg}} -hdevice {{mode}} -printfile {{png}} {{flags}}")
@task(xvg=FILE_IN, png=FILE_OUT)
def convert_xvg_to_png(xvg=None, mode='PNG', png=None, flags='-hardcopy'):
    # Command: grace -nxy protein_potential.xvg -hdevice PNG -hardcopy -printfile protein_potential.png
    pass

# ############ #
# Final tasks #
# ############ #

@task(images=COLLECTION_FILE_IN,
      result=FILE_OUT)
def merge_results(images, result):
    from PIL import Image
    imgs = map(Image.open, images)
    widths, heights = zip(*(i.size for i in imgs))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in imgs:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save(result)


# ############# #
# MAIN FUNCTION #
# ############# #

def main(dataset_path, output_path, config_path):
    print("Starting demo")

    protein_names = []
    protein_pdbs = []

    # Look for proteins in the dataset folder
    for f in listdir(dataset_path):
        if isfile(join(dataset_path, f)):
            protein_names.append(f.split('.')[0])
            protein_pdbs.append(join(dataset_path, f))
    proteins = zip(protein_names, protein_pdbs)
    # Start counting time
    start_time = time()

    # Iterate over the proteins and process them
    result_image_paths = []
    for name, pdb in proteins:
        # 1st step - Generate topology
        structure = join(output_path, name + '.gro')
        topology = join(output_path, name + '.top')
        generate_topology(protein=pdb,
                          structure=structure,
                          topology=topology)
        # 2nd step - Define box
        structure_newbox = join(output_path, name + '_newbox.gro')
        define_box(structure=structure,
                   structure_newbox=structure_newbox)
        # 3rd step - Add solvate
        protein_solv = join(output_path, name + '_solv.gro')
        add_solvate(structure_newbox=structure_newbox,
                    protein_solv=protein_solv,
                    topology=topology)
        # 4th step - Add ions
        # Assemble with ions.mdp
        ions_conf = join(config_path, 'ions.mdp')
        ions = join(output_path, name + '_ions.tpr')
        assemble_tpr(conf=ions_conf,
                     protein_solv=protein_solv,
                     topology=topology,
                     output=ions)
        protein_solv_ions = join(output_path, name + '_solv_ions.gro')
        group = join(config_path, 'genion.group')
        replace_solvent_with_ions(ions=ions,
                                  output=protein_solv_ions,
                                  topology=topology,
                                  group=group)
        # 5th step - Minimize energy
        # Reasemble with minim.mdp
        minim_conf = join(config_path, 'minim.mdp')
        em = join(output_path, name + '_em.tpr')
        assemble_tpr(conf=minim_conf,
                     protein_solv=protein_solv_ions,
                     topology=topology,
                     output=em)
        em_energy = join(output_path, name + '_em_energy.edr')
        energy_minimization(em=em,
                            em_energy=em_energy)
        # 6th step - Energy analysis and convert the xvg to png
        energy_result = join(output_path, name + '_potential.xvg')
        energy_selection = join(config_path, 'energy.selection')  # 10 = potential
        energy_analisis(em=em_energy,
                        output=energy_result,
                        selection=energy_selection)
        energy_result_png = join(output_path, name + '_potential.png')
        result_image_paths.append(energy_result_png)
        convert_xvg_to_png(xvg=energy_result, png=energy_result_png)

    # Merge all images into a single one
    result = join(output_path, 'POTENTIAL_RESULTS.png')
    merge_results(result_image_paths, result)
    
    compss_barrier()
    elapsed_time = time() - start_time
    print("Elapsed time: %0.10f seconds." % elapsed_time)

if __name__=='__main__':
    config_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]

    main(dataset_path, output_path, config_path)
    # from pwd ~/gromacs/src/
    # execute the next command
    # runcompss -m -d --python_interpreter=python3 ./lysozyme_in_water.py /home/compss/gromacs/config /home/compss/gromacs/dataset /home/compss/gromacs/output
   
