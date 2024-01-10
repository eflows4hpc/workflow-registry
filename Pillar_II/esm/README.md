# eFlows4HPC ESM workflow

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

> NOTE: Extrae is a BSC tool used to trace applications providing
> post-mortem analysis data: https://tools.bsc.es/extrae

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

To run the tests, have a look at the `.github/workflows/pillar_ii_esm.yml`.
You should be able to run the tests after installing the required
dependencies, using the following command:

```bash
pytest src tests/
```

It will print the result of the tests, and will also include the test
coverage report in the terminal.

### Compiling FESOM2

[comment]: <> (This information was copied from BSC B2DROP file:
               `compiling-fesom-with-suvi-20230926.txt`)

For eFlows4HPC ESM, FESOM2 was modified to use Hecuba to store the
model run outputs. At the time of writing, these changes have not
yet been integrated into the main branch of their Git repository.
They are in the [eflows_hecuba_templates_update](https://github.com/FESOM/fesom2/tree/eflows_hecuba_templates_update)
branch.

To compile it you will need to check out that branch, and then
run `./configure.sh bsc` (replacing `bsc` by your site name).
If you compile multiple times, it is recommended to delete the
generated `./build` directory before each run, to avoid
contamination of previous builds.

> NOTE: In some sites you may be able to compile FESOM2 with
>       Hecuba without loading the Hecuba module, exporting
>       `HECUBA_ROOT=/apps/HECUBA/$version`. But on MN4 that
>       worked up to 100%, but then after the message about
>       linking the executable it failed by not being able
>       to locate some compression libraries.

The integration with Hecuba can be turned on or off in CMake. That
is achieved through the `USE_HECUBAIO` flag in `CMakeLists.txt`.
If you want to test that branch of FESOM2 without Hecuba, just set
that to `OFF`, delete the `./build` directory, and run
`./configure.sh $env` again.

## Running

### FESOM2

To run the FESOM2 ESM simulation, you need to:

1. Modify `src/fesom2/esm_ensemble.conf`,
   changing the `top_working_dir` and `output_dir` to a valid
   location for your user.
2. Modify the FESOM2 settings in the same file.
3. Either modify the start dates in the same file, or pass it
   via command-line arguments.

Then you can use the `src/launch_fesom2.sh` to launch the simulation.
For example:

```bash
$ ./launch_fesom2.sh \
        --hpc mn4 \
        --qos debug \
        --start_dates "1948,1958" \
        --cores_per_node 48 \
        --cores 48
```

The simulation Python script randomly creates an `expid` with
`8` digits (e.g. `12345678`). But you can also pass a fixed `expid`
if you prefer.

The output of the shell execution will contain a Slurm Job ID.
You can check the COMPSs directory with the `expid` name. And
you can also find useful information in a folder created under
your user home directory, e.g. `~/.COMPSs/12345678/jobs/`.
In this directory you will find the outputs of your Job execution.

If you prefer, you can skip the Shell script and use the Python
script directly, without PyCOMPSs. This is useful for testing:

```bash
$ python esm_simulation.py \
        --expid 123456 \
        --model fesom2 \
        --config ~/esm_ensemble.conf \
        --start_dates="1948,1958,1968" \
        --debug
```

### Troubleshooting

After you launch your experiment, you can start by confirming that the
command used by COMPSs looks correct. That command should be printed in
the file `.enqueue_compss`.

```bash
$ cat .enqueue_compss | sed -e 's/ -/ \\\n        -/g'
enqueue_compss \
        --tracing \
        --graph=true \
        --debug \
        --sc_cfg=mn.cfg \
        ...\
        --model fesom2 \
        --start_dates "/"1948,1958/"" \
        --expid 006279 \
        --debug
```

[comment]: <> (These Slurm commands were extracted from the document
               `alien4cloud_notes.md` from BSC B2DROP.)

Then inspect your Slurm batch jobs outputs. At the end of the launcher
script output you should have a Slurm job ID. The COMPSs logs will use
that job ID in its names.

```bash
# get the job ID from the launcher script output, or from `squeue`
$ squeue --me
# find the job locations
$ squeue show job $jobid
$ head compss-31278175.out
STARTING UP CASSANDRA...
Launching Cassandra in the following hosts: s10r1b47
Checking...
0/1 nodes UP. Retry #0
Checking...
1/1 nodes UP. Cassandra Cluster started successfully.
[STATS] Cluster launching process took:  18s.  985ms.
head compss-31278175.err
Picked up JAVA_TOOL_OPTIONS: -Xss1280k
Picked up JAVA_TOOL_OPTIONS: -Xss1280k
Picked up JAVA_TOOL_OPTIONS: -Xss1280k
Connection error: ('Unable to connect to any servers', {'10.1.9.47:9042': ConnectionRefusedError(111, "Tried connecting to [('10.1.9.47', 9042)]. Last error: Connection refused")})
Connection error: ('Unable to connect to any servers', {'10.1.9.47:9042': ConnectionRefusedError(111, "Tried connecting to [('10.1.9.47', 9042)]. Last error: Connection refused")})
Warning: Permanently added 's10r1b47,10.2.9.47' (ECDSA) to the list of known hosts.
remove java/8u131 (PATH, MANPATH, JAVA_HOME, JAVA_ROOT, JAVA_BINDIR, SDK_HOME,
JDK_HOME, JRE_HOME)
remove papi/5.5.1 (PATH, LD_LIBRARY_PATH, C_INCLUDE_PATH)
unload PYTHON/3-intel-2021.3 (PATH, MANPATH, LIBRARY_PATH, PKG_CONFIG_PATH,
```

#### Inspecting the Cassandra snapshots

[comment]: <> (Information taken from BSC B2DROP file `cassandra_notes.md`.)

When the eFlows4HPC ESM application runs with Hecuba, it starts
a Cassandra cluster to store the data (using a BSC utility called
`cassandra4slurm`, or `c4s`). Once the application is over
the cluster is shut down. Snapshots are data files that contain data
collected during the execution and that are persisted after the
ESM application run. This is useful for confirming that data has
been written, and to inspect this data.

You can enable the snapshots by modifying `~/.c4s/conf/cassandra4slurm.cfg`:

```ini
CASS_HOME="$HECUBA_ROOT/cassandra-d8tree"
CASSANDRA_LOG_DIR=/.../eflows4hpc/top_working_dir/011213/1948/logs
DATA_PATH="/scratch/tmp"
LOG_PATH=/.../eflows4hpc/top_working_dir/011213/1948/logs
SNAP_PATH="/.../eflows4hpc/top_working_dir/011213/1948/logs"
```

If you have an instance of Cassandra still running, you can inspect
it with `cqlsh %NODE_NAME%`, and try a query like
`SELECT name FROM hecuba.istorage;`.

The snapshots will be available in a folder created in your
specified directory. In this folder you will have another
subfolder with the `expid` that contains `cassandra.output`.
This file contains the execution log for Cassandra.

In the `~/.c4s/` directory, you may also have several files
like `cassandra-snapshot-file-${expid}.txt`.

You can use Python `numpy`, for instance, to access the snapshot
data and validate that your simulation output was written correctly.

In a Cassandra instance the Numpy information is available too
with `SELECT * FROM my_app.hecuba_storagenumpy`. That will bring
blob objects as the Numpy arrays are stored in binary. Using Python
you can try:

```py
from hecuba import StorageNumpy
sn = StorageNumpy(
    None, # new storage numpy?
   "exp24/sst/4/84")
print(sn)
```

You can also recover old snapshots:

```bash
$ c4 RECOVER --qos=debug -t=00:10:00
```

Choose the last snapshot ID (most recent), if appropriate.
Cassandra will start from that snapshot. Not necessarily
fast. Then you can use `cqlsh` to run queries against it.

#### Running from Alien4Cloud

There is an environment initialization script:
`/gpfs/projects/dese28/eflows4hpc/esm/fesom2/env.sh`

This does the same initializations as in `launch_fesom2.sh`
but also specifies cores, queue type and n/ensembles.

Then in the A4C topology, in the `extra_compss_opts=`
we specify the `--env_script`:

```bash
$ compss... \
        --qos=debug \
        --exec_time=120 \
        --keep_workingdir \
        --worker_working_dir=/gpfs/projects/dese28/eflows4hpc/esm/fesom2/src \
        --worker_in_master_cpus=48 \
        --num_nodes=3 \
        --pythonpath=/gpfs/projects/dese28/eflows4hpc/esm/fesom2/src:/apps/HECUBA/2.1_intel/compss \
        --env_script=/gpfs/projects/dese28/eflows4hpc/esm/fesom2/env.sh \
        --storage_props=/gpfs/projects/dese28/eflows4hpc/esm/fesom2/src/hecuba_lib/storage_props.cfg --storage_home=/apps/HECUBA/2.1_intel/compss
```
