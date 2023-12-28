This is the documentation for the eFlows4HPC ESM Workflow:
https://github.com/eflows4hpc/workflow-registry/tree/main/Pillar_II/esm

Even though there are a `requirements.txt` and a `src` directory,
note that this is not a Python package. We use Python scripts to
launch the ESM simulation.

The execution is fully parametrized. The parameters influence how
the scripts will choose to run different configurations like using
FESOM2 or AWICM3, choose the number of ensembles, start years, and
also where to write the output file.

Each model contains template namelists included in this repository.
Refer to those templates and to the generated final files, as well
as the execution log files to verify the execution of the workflow.

## Installation

For PyCOMPSs you first must export this variable:

```bash
export EXTRAE_MPI_HEADERS=/usr/include/x86_64-linux-gnu/mpi
```

Then you just need to run `pip install -r requirements.txt` to
get all the dependencies needed for the ESM workflow installed.

## Build

To check the Shell scripts:

```bash
$ find . -name "*.sh" -exec shellcheck --external-sources {} \;
```

To check the type hints:

```bash
$ mypy src
```

## Running

## FESOM2

To run the FESOM2 ESM simulation, you need to:

1. Modify `src/config/esm_ensemble.conf`,
   changing the `top_working_dir` and `output_dir` to a valid
   location for your user.
2. Modify `src/esm_simulation.py`
   changing the output dir (`path`) to a valid location for
   your user.

Then you can use the `src/launch_mn4.sh` to launch the simulation.
It takes three parameters, for example:

```bash
#               cores queue n/ensembles
./launch_mn4.sh 288 debug 2
```

The simulation Python script randomly creates an `expid` with
`8` digits (e.g. `12345678`).

The output of the shell execution will contain a Slurm Job ID.
You can check the COMPSs directory with the `expid` name. And
you can also find useful information in a folder created under
your user home directory, e.g. `~/.COMPSs/12345678/jobs/`.
In this directory you will find the outputs of your Job execution.

#### Inspecting the Cassandra snapshots

TODO: explain how to activate the generation of snapshots

The snapshots will be available in a folder created in your
home directory: `~/.c4s/`. In this folder you will have another
subfolder with the `expid` that contains `cassandra.output`.
This file contains the execution log for Cassandra.

In the `~/.c4s/` directory, you will also have several files
like `cassandra-snapshot-file-${expid}.txt`.

You can use Python `numpy`, for instance, to access the snapshot
data and validate that your simulation output was written correctly.

#### Running from Alien4Cloud

There is an environment initialization script:
/gpfs/projects/dese28/eflows4hpc/esm/fesom2/env.sh

This does the same initializations as in launch_mn4.sh but also specifies cores, queue type and n/ensembles.

Then in the A4C topology, in the extra_compss_opts=

we specify the --env_script:

```bash
--qos=debug --exec_time=120 --keep_workingdir --worker_working_dir=/gpfs/projects/dese28/eflows4hpc/esm/fesom2/src --worker_in_master_cpus=48 --num_nodes=3 --pythonpath=/gpfs/projects/dese28/eflows4hpc/esm/fesom2/src:/apps/HECUBA/2.1_intel/compss --env_script=/gpfs/projects/dese28/eflows4hpc/esm/fesom2/env.sh --storage_props=/gpfs/projects/dese28/eflows4hpc/esm/fesom2/src/hecuba_lib/storage_props.cfg --storage_home=/apps/HECUBA/2.1_intel/compss
```


