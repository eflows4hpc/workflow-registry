

## Running

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

The output of the shell execution  will contain a Slurm Job ID.
You can check the COMPSs directory with the `expid` name. And
you can also find useful information in a folder created under
your user home directory, e.g. `~/.COMPSs/12345678/jobs/`.
In this directory you will find the outputs of your Job execution.

### Inspecting the Cassandra snapshots

TODO: explain how to activate the generation of snapshots

The snapshots will be available in a folder created in your
home directory: `~/.c4s/`. In this folder you will have another
subfolder with the `expid` that contains `cassandra.output`.
This file contains the execution log for Cassandra.

In the `~/.c4s/` directory, you will also have several files
like `cassandra-snapshot-file-${expid}.txt`.

You can use Python `numpy`, for instance, to access the snapshot
data and validate that your simulation output was written correctly.
