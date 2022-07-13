#Workflow Registry

This is a repository to store the Workflow descriptions using the eFlows4HPC methodology. This description consist of at least the TOSCA description of the worklfow, the code of the their different steps and their required software per step.


## Repository structure

Workflow descriptions have to be included inside this repository according to the following structure.

```
workflow-registry
  |- workflow_1
  |    |- tosca
  |    |    |- types.yml               TOSCA description of the different components involved in the workflow
  |    |       ... 
  |    |- step_1
  |    |    |- spack.yml               Sofware requirements for this workflow step as a Spack environment specification 
  |    |    |- src                     PyCOMPSs code of the workflow step
  |    |       ...
  |    |- step_2
  |         ....
  |- workflow_2                                
  |	...

```

## Including new Workflows

To include new workflows in the repository, first create a new fork of the repository and  include a new folder for the workflow with a subfolder for the TOSCA description and the different workflow steps. Finally, create a pull request with the new workflow description. This pull request will be reviewed and included in the repository.

