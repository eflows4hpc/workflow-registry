This is the Readme for:
Feature Extraction workflow

[Name]: Feature Extraction
[Access Level]: public
[Platform]: COMPSs

[Body]
== Description ==
Feature Extraction is an application that integrates different components of the ESM workflow: CMCC-CM3 simulation, post-processing, HPDA and machine learning.



== Execution instructions ==
Usage:
runcompss --graph=true --wall_clock_limit=14400 --streaming=FILES src/feature_extraction.py <inputPath> <outputPath> 

where:
        * - inputPath: Absolute path of the input files (e.g. /home/feature_extraction/input/)
        * - outputPath: Absolute path of the output files (e.g. /home/feature_extraction/output/)

