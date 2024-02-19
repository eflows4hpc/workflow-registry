This is the Readme for:
CMCC-CM3 model

[Name]: CMCC-CM3 model
[Access Level]: public
[Platform]: COMPSs

[Body]
== Description ==
CMCC-CM3 model is an application that runs the CMCC-CM3 simulation.


== Execution instructions ==
Usage:
runcompss --graph=true --wall_clock_limit=14400 --streaming=FILES src/CMCC-CM3_model.py <inputPath> <outputPath> 

where:
        * - inputPath: Absolute path of the input files (e.g. /home/CMCC-CM3_model/input/)
        * - outputPath: Absolute path of the output files (e.g. /home/CMCC-CM3_model/output/)

