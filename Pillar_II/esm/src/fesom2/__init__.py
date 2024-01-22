import logging
import os
import shutil
from configparser import ConfigParser
from itertools import chain
from pathlib import Path
from typing import List, NamedTuple, Optional

from pycompss.api.api import TaskGroup  # type: ignore
from pycompss.api.api import TaskGroup  # type: ignore
from pycompss.api.exceptions import COMPSsException  # type: ignore
from pycompss.api.mpi import mpi  # type: ignore
from pycompss.api.on_failure import on_failure  # type: ignore
from pycompss.api.parameter import IN, Type, FILE_OUT, StdIOStream, STDOUT, INOUT, Prefix  # type: ignore
from pycompss.api.task import task  # type: ignore

logger = logging.getLogger(__name__)

"""This module contains the required code to run FESOM2 in eFlows4HPC."""


class Config(NamedTuple):
    """The model configuration object."""
    start_date: str
    mesh_path: Path
    climatology_path: Path
    output_path: Path
    forcing_files_path: Path


def _get_model_config(start_date: str, config_parser: ConfigParser) -> Config:
    """Get the configuration for this specific model.

    It uses the eFlows4HPC configuration to build the configuration for
    the model.
    Args:
        start_date: Start date.
        config_parser: eFlows4HPC configuration.

    Returns:
        Model configuration.
    """
    mesh_path = Path(config_parser['fesom2']['mesh_file_path'])
    climatology_path = Path(config_parser['fesom2']['climatology_path'])
    forcing_files_path = Path(config_parser['fesom2']['forcing_files_path'])
    output_path = Path(config_parser['common']['output_dir'], start_date)
    return Config(
        start_date=start_date,
        mesh_path=mesh_path,
        climatology_path=climatology_path,
        forcing_files_path=forcing_files_path,
        output_path=output_path
    )


def _namelists(start_date: str, config: ConfigParser):
    """Get the namelists used for FESOM2.

    N.B.: Python's pathlib strips trailing slashes from paths. This causes issues
          in the templates when the variables like {CLIMATOLOGY_PATH}{START_DATE} are
          created. This means we must always add the {os.sep} at the end.
          https://stackoverflow.com/a/47572715

    Args:
        start_date: Start date.
        config: The eFlows4HPC configuration.

    Returns:
        A dictionary with the name of the namelist, and an optional dictionary of values to process the namelist.
    """
    model_config = _get_model_config(start_date, config)
    return {
        'namelist.config.tmpl': {
            'START_YEAR': model_config.start_date,
            'MESH_PATH': f'{model_config.mesh_path}{os.sep}',
            'CLIMATOLOGY_PATH': f'{model_config.climatology_path}{os.sep}',
            'OUTPUT_PATH': f'{model_config.output_path}{os.sep}'
        },
        'namelist.cvmix.tmpl': None,
        'namelist.forcing.tmpl': {
            'FORCING_SET_PATH': f'{model_config.forcing_files_path}{os.sep}'
        },
        'namelist.ice.tmpl': None,
        'namelist.icepack.tmpl': None,
        'namelist.io.tmpl': None,
        'namelist.oce.tmpl': None,
    }


def _esm_ensemble_generate_namelists(start_date: str, config: ConfigParser, member_top_working_dir: Path) -> None:
    namelists = _namelists(start_date, config)
    for namelist_file_name, values in namelists.items():
        namelist_file = Path(__file__).parent.resolve() / 'namelist_templates' / namelist_file_name
        destination_file = member_top_working_dir / namelist_file.stem
        if not namelist_file.exists() or not namelist_file.is_file():
            raise ValueError(f"Invalid namelist template file (not a file, or does not exist): {namelist_file}")

        if values is None:
            # No values to replace, let's just copy this...
            shutil.copy(namelist_file, destination_file)
        else:
            # Replace values ``{var}`` in the template and write it to the target file...
            with open(namelist_file, "r") as fin, open(destination_file, "w+") as file:
                logger.debug(f"Processing namelist {namelist_file}, values {str(values)}")
                fin_data = fin.read()
                data = fin_data.format(**values)
                logger.debug(f"Old namelist contents:\n{fin_data}\n\nNew namelist contents:\n{data}")
                file.write(data)


def init_top_working_dir(
        top_working_dir: Path,
        access_rights: int,
        start_dates: List[str],
        config: ConfigParser) -> None:
    """Initialize the FESOM2 model top working directory.

    This directory exists somewhere like /data/top-working-dir/expid/start-dates...

    For each start date, we will produce a symlink there to ``fesom.x``.
    We will place each processed Fortran namelist, and Hecuba YAML file
    there.

    Args:
        top_working_dir: The model top working directory, where files needed to run the model reside for each member.
        access_rights: Default file mode to create new file system entries (e.g. 0x755)
        start_dates: The list of start dates for the model run.
        config: eFlows4HPC configuration.
    """
    fesom_exe = Path(config['fesom2']['fesom_binary_path'])
    if not fesom_exe.exists() or fesom_exe.is_dir():
        raise ValueError(f"Invalid FESOM executable (does not exist or not a file): {fesom_exe}")

    if top_working_dir.exists() and not top_working_dir.is_dir():
        raise ValueError(f"Invalid configuration directory (expected existing directory): {dir}")

    if not top_working_dir.exists():
        logger.debug(f"Creating new top working directory: {top_working_dir}")
        top_working_dir.mkdir(mode=access_rights, parents=True)

    logger.info(f"Initializing top working directory: {top_working_dir}")

    for start_date in start_dates:
        logger.info(f"=== Start date {start_date}")
        member_top_working_dir = top_working_dir / start_date
        if not member_top_working_dir.exists():
            logger.debug(f"Creating member directory: {member_top_working_dir}")
            member_top_working_dir.mkdir(mode=access_rights)

        fesom_x_link = member_top_working_dir / "fesom.x"
        if fesom_x_link.is_file():
            fesom_x_link.unlink()
        logger.info(f"Creating symlink for FESOM2 executable, from [{fesom_exe}] to [{fesom_x_link}]")
        fesom_x_link.symlink_to(fesom_exe)

        source_directory_datamodels = Path(config['fesom2']['fesom_hecuba_datamodel'])
        logger.info(f"Copying Hecuba YAML datamodel files from [{source_directory_datamodels}] "
                    f"to the top working directory (works with .yml. YAml, .yaml, etc.)")
        # Find any .yaml, .yml, .YAML, etc., files.
        for source_filename in chain(
                source_directory_datamodels.rglob('*.[yY][mM][lL]'),
                source_directory_datamodels.rglob('*.[yY][aA][mM][lL]')):
            shutil.copy(source_filename, member_top_working_dir)

        logger.info("Generating Fortran namelists...")
        _esm_ensemble_generate_namelists(start_date, config, member_top_working_dir)


def init_output_dir(
        output_dir: Path,
        access_rights: int,
        start_dates: List[str]) -> None:
    """Initialize the FESOM2 model output directory.

    This directory exists somewhere like /data/output-dir/expid/start-dates...

    The output directory will be created if it does not yet exist. We will place
    the ``fesom.clock`` file there with the respective date written inside.
    https://fesom2.readthedocs.io/en/latest/getting_started/getting_started.html#preparing-the-run

    Args:
        output_dir: The model output directory, where files produced by each ensemble model run reside.
        access_rights: Default file mode to create new file system entries (e.g. 0x755)
        start_dates: The list of start dates for the model run.
    """
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Invalid configuration directory (expected existing directory): {str(output_dir)}")

    if not output_dir.exists():
        logger.debug(f"Creating new output directory: {output_dir}")
        output_dir.mkdir(mode=access_rights, parents=True)

    logger.info(f"Initializing output directory: {output_dir}")

    for start_date in start_dates:
        logger.info(f"=== Start date {start_date}")
        member_output_dir = output_dir / start_date
        if not member_output_dir.exists():
            logger.debug(f"Creating member directory: {member_output_dir}")
            member_output_dir.mkdir(mode=access_rights)

        logger.info("Creating clock file...")
        with open(member_output_dir / "fesom.clock", "w") as fclock:
            fclock.write(f"0 1 {start_date}\r\n")
            fclock.write(f"0 1 {start_date}")


# TODO: not used?
# @task(returns=bool)
# def esm_dynamic_analysis(exp_id: str, start_years: List[str]) -> str:
#     # NOTE: This import triggers a connection to Hecuba! Leaving it here
#     #       makes it easier to test this scripts without COMPSs or Hecuba.
#     from hecuba_lib.esm_dynamic_analysis_results import esm_dynamic_analysis_results  # type: ignore
#     """Dummy method to test data exchange with Hecuba.
#
#     The ``ds.make_persistent`` creates the analysis, which is then used
#     in another task.
#
#     Args:
#         exp_id: Experiment ID.
#         start_years: List of start years.
#     Returns:
#         The persisted task ID.
#     """
#     logger.info(f"Performing dynamic analysis for experiment {exp_id}")
#     ds = esm_dynamic_analysis_results()
#     for start_year in start_years:
#         ds.results[start_year] = True
#     # TODO: Check with others why only one year was being set to True here. And why
#     #       the years were hard-coded, even though we had two sets of start years?
#     # ds.results["1948"] = False
#     # ds.results["1958"] = True
#     # ds.results["1968"] = False
#     analysis_id = f"{exp_id}_esm_dynamic_analysis"
#     logger.info(f"Dynamic analysis ID {analysis_id}")
#     ds.make_persistent(analysis_id)
#     return analysis_id
#
#
# TODO: Not used?
# @on_failure(management='IGNORE')
# @task(returns=bool)
# def esm_member_disposal(start_date: str, config_parser: ConfigParser) -> bool:
#     """Abort a member, deleting its data."""
#     output_path = Path(config_parser['common']['output_dir'], start_date)
#     if output_path.exists() and output_path.is_dir():
#         logger.debug(f"Deleting ESM member aborted [{start_date}]: {output_path}")
#         shutil.rmtree(output_path)
#     # TODO: remove hecuba data of the concerned aborted member as well
#     return True

@on_failure(management='IGNORE')
@mpi(binary="{{fesom_binary_path}}",
     runner="{{runner}}",
     processes="{{processes}}",
     working_dir="{{working_dir_exe}}",
     processes_per_node="{{processes_per_node}}",
     fail_by_exit_value=True,
     )
@task(
    log_file={
        Type: FILE_OUT,
        StdIOStream: STDOUT
    },
    working_dir_exe={
        Type: INOUT,
        Prefix: "#"
    },
    runner={
        Type: IN,
    },
    processes={
        Type: IN,
    },
    fesom_binary_path={
        Type: IN,
    },
    processes_per_node={
        Type: IN,
    },
    returns=int)
def esm_simulation(
        log_file: str,
        working_dir_exe: str,
        runner: str,
        processes: int,
        fesom_binary_path: str,
        processes_per_node: int,
) -> Optional[int]:  # type: ignore
    """PyCOMPSs task that executes the ``FESOM_EXE`` binary."""
    pass


@on_failure(management='IGNORE')
# Jorge: Prefix is only needed in @mpi or @binary to avoid to pass the parameter to the binary execution, res={Type:IN, Prefix:"#"})
@task
def esm_member_checkpoint(start_date: str, config_parser: ConfigParser, res: int) -> None:
    # NOTE: This import triggers a connection to Hecuba! Leaving it here
    #       makes it easier to test this scripts without COMPSs or Hecuba.
    from hecuba_lib.esm_dynamic_analysis_results import esm_dynamic_analysis_results  # type: ignore
    # retrieve from Hecuba the last status of the ensemble members produced by the analysis (running in parallel)
    logger.info(f"Checking status of ESM member: {start_date}")
    logger.info(f"The returned value is {str(res)}")
    ensemble_status = esm_dynamic_analysis_results.get_by_alias(
        config_parser['runtime']['expid'] + "_esm_dynamic_analysis")
    logger.info(f"Status for member [{start_date}] is: {str(ensemble_status)}")
    if not bool(ensemble_status.results[start_date]):
        raise COMPSsException(f"Member diverged - Aborting member: {start_date}")


__all__ = [
    'init_top_working_dir',
    'init_output_dir',
    'esm_simulation',
    'esm_member_checkpoint'
]
