import logging
import random
from argparse import ArgumentParser
from configparser import ConfigParser
from enum import Enum, unique
from importlib import import_module
from pathlib import Path
from shutil import rmtree
from string import ascii_letters
from typing import Any, Callable, List, Optional

from pycompss.api.api import TaskGroup  # type: ignore
from pycompss.api.api import compss_barrier_group  # type: ignore
from pycompss.api.api import compss_cancel_group  # type: ignore
from pycompss.api.api import compss_wait_on  # type: ignore
from pycompss.api.exceptions import COMPSsException  # type: ignore
from pycompss.api.on_failure import on_failure  # type: ignore
from pycompss.api.parameter import IN  # type: ignore
from pycompss.api.task import task  # type: ignore
from pycompss.runtime.management.classes import Future  # type: ignore

from dummy_prune_analysis import esm_analysis_prune

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.NOTSET)

logger = logging.getLogger(__name__)

# Every model must define some basic operations to be used in the ESM. We load
# these dynamically. Their type definitions are below:

init_top_working_dir_fn = Callable[[Path, int, List[str], int, ConfigParser], None]
init_output_dir_fn = Callable[[Path, int, List[str], int, str], None]
esm_simulation_fn = Callable[[str, str, str, str, str, str], Future]  # int
esm_member_checkpoint_fn = Callable[[str, ConfigParser, Any], Future]  # bool


@unique
class Model(str, Enum):
    """eFlows4HPC available models. Must match imported Python modules."""
    FESOM2 = 'fesom2'
    AWICM3 = 'awicm3'


def _get_config(*, model_config: Path, model: str, start_dates: Optional[str], members: Optional[int],
                processes: Optional[str], processes_per_node: Optional[str]) -> ConfigParser:
    """Get the eFlows4HPC configuration object.

    Args:
        model_config: The path to the model configuration file.
        model: The model name (see ``Model`` enum).
        start_dates: Start dates.
        members: Number of ensemble members.
        processes: Number of cores.
        processes_per_node: Number of processes per node.
    Returns:
        A ``ConfigParser`` instance.
    """
    if not model_config.exists() or not model_config.is_file():
        raise ValueError(f"Model {model} configuration file not located at {model_config}")

    config_parser = ConfigParser(inline_comment_prefixes="#")
    config_parser.read(model_config)

    # Users can specify a different list of start dates (from Alien4Cloud, for instance).
    # So we override the value from the configuration file.
    if start_dates is not None and start_dates.strip() != '':
        config_parser['common']['ensemble_start_dates'] = start_dates
    logger.info(f"List of start dates: {config_parser['common']['ensemble_start_dates']}")

    if members is not None:
        if members <= 0:
            raise ValueError(f"Invalid number of ensemble members: [{members}]")
        config_parser['common']['ensemble_members'] = str(members)
    logger.info(f"Number of members: {config_parser['common']['ensemble_members']}")

    if processes is not None:
        # ``str()`` as ``ConfigParser`` can only store strings.
        config_parser['pycompss']['processes'] = str(processes)

    if processes_per_node is not None:
        config_parser['pycompss']['processes_per_node'] = str(processes_per_node)

    # TODO: A possible fix for the warning in PyCOMPSs?
    # if processes is not None and processes_per_node is not None:
    #     if int(processes) < int(processes_per_node):
    #         logger.warning("Processes is smaller than processes per node. Fixing it so PyCOMPSs does not fail/warn.")
    #         config_parser['pycompss']['processes'] = config_parser['pycompss']['processes_per_node']

    return config_parser


def _init_top_working_directory(
        top_working_dir: Path,
        access_rights: int,
        start_dates: List[str],
        members: int,
        model: str,
        config: ConfigParser) -> None:
    """This function does most of the heavy-work to initialize the top working directory.

    In this directory we store files needed to run the model.

    Args:
        top_working_dir: The model top working directory, where files needed to run the model reside for each member.
        access_rights: Default file mode to create new file system entries (e.g. 0x755)
        start_dates: The list of start dates for the model run.
    """
    model_module = import_module(model)
    fn: init_top_working_dir_fn = model_module.init_top_working_dir
    fn(top_working_dir,
       access_rights,
       start_dates,
       members,
       config)


def _init_output_directory(
        output_dir: Path,
        access_rights: int,
        start_dates: List[str],
        members: int,
        config: ConfigParser) -> None:
    """Prepare the output directories for the model run.

    Args:
        output_dir: The model output directory, where the model may write data like NetCDF files, or images.
        access_rights: Default file mode to create new file system entries (e.g. 0x755)
        start_dates: The list of start dates for the model run.
        config: eFlows4HPC configuration.
    """
    model_module = import_module(f"{config['runtime']['model']}")
    expid = config['runtime']['expid']
    fn: init_output_dir_fn = model_module.init_output_dir
    fn(output_dir,
       access_rights,
       start_dates,
       members,
       expid)


@task(expid=IN, model=IN, config_parser=IN)
def esm_ensemble_init(*, expid, model, config_parser: ConfigParser) -> ConfigParser:
    logger.info(f"Initializing experiment {expid}")

    # We populate a few settings that exist only during runtime.

    config_parser['runtime']["expid"] = expid
    config_parser['runtime']['model'] = model

    access_rights = int(config_parser['common']['new_dir_mode'], 8)
    logger.info(f"New directories will be created with the file mode: {oct(access_rights)}")

    top_working_dir = Path(config_parser['common']['top_working_dir'], expid)
    output_dir = Path(config_parser['common']['output_dir'], expid)
    start_dates = config_parser['common']['ensemble_start_dates'].split(",")
    members = int(config_parser['common']['ensemble_members'])
    model_name = config_parser['runtime']['model']

    _init_top_working_directory(top_working_dir, access_rights, start_dates, members, model_name, config_parser)
    _init_output_directory(output_dir, access_rights, start_dates, members, config_parser)

    return config_parser


@on_failure(management='IGNORE')
@task(start_date=IN, top_working_dir=IN, output_dir=IN, returns=bool)
def esm_member_disposal(start_date: str, member: int, top_working_dir: Path, output_dir: Path) -> None:
    # TODO: remove hecuba data as well of the concerned aborted member
    for path in [top_working_dir, output_dir]:
        start_date_path = Path(path) / start_date / str(member)
        rmtree(start_date_path)


def _run_esm(*, expid: str, model: str, config_parser: ConfigParser) -> None:
    # This is the only step that is common to the models. But we can move
    # that to a common module and call the model's code directly instead
    # if needed too. For now this is good enough.
    runtime_config_parser: ConfigParser = compss_wait_on(
        esm_ensemble_init(
            expid=expid,
            model=model,
            config_parser=config_parser
        ))
    logger.info("ESM Initialization complete!")

    # Create environment variables that we need for COMPSs as
    # it cannot read certain variables during runtime (i.e.
    # ``{{processes}}`` in a ``@mpi`` task).
    runner = runtime_config_parser['pycompss']['runner']
    fesom_binary_path = runtime_config_parser['fesom2']['fesom_binary_path']
    processes = runtime_config_parser['pycompss']['processes']
    processes_per_node = runtime_config_parser['pycompss']['processes_per_node']

    # This is where we delegate the ESM execution to a model's module code
    # (e.g. FESOM2, AWICM3, etc.).
    ensemble_start_dates = runtime_config_parser['common']['ensemble_start_dates'].split(",")
    ensemble_members = int(runtime_config_parser['common']['ensemble_members'])
    top_working_dir = Path(runtime_config_parser['common']['top_working_dir'], expid)
    output_dir = Path(config_parser['common']['output_dir'], expid)
    model_module = import_module(f"{runtime_config_parser['runtime']['model']}")

    for start_date in ensemble_start_dates:
        for member_idx in range(1, ensemble_members + 1):
            member = str(member_idx)
            task_group = f"{expid}_{start_date}_{member}"
            with TaskGroup(task_group, implicit_barrier=False):
                # Launch each SIM, create an implicit dependence by passing the result to the next task (checkpoint).
                number_simulations = int(runtime_config_parser['common']['chunks'])
                logger.info(f"Total of chunks configured: {number_simulations}")

                for sim in range(1, number_simulations + 1):
                    working_dir_exe = top_working_dir / start_date / member
                    log_file = str(working_dir_exe / f"fesom2_{expid}_{start_date}_{member}_{str(sim)}.out")
                    logger.info(f"Launching simulation {start_date}.{member}.{str(sim)} in {working_dir_exe}")
                    logger.info(f"Processes [{processes}], per node [{processes_per_node}], runner [{runner}]")
                    simulation_fn: esm_simulation_fn = model_module.esm_simulation
                    res: Future = simulation_fn(
                        log_file,
                        str(working_dir_exe),
                        fesom_binary_path,
                        runner,
                        processes,
                        processes_per_node
                    )
                    logger.debug(f"Simulation binary execution return: {res}")
                    # check the state of the member, for the first one there is nothing to check
                    if sim > 1:
                        logger.info(f"Checkpoint member: {str(sim)}")
                        # Note: The barrier was added to check if it solves the pruning issue.
                        #       Adding a barrier causes all tasks to run in serial even if
                        #       these are in different task groups.
                        # compss_wait_on(model_module.esm_member_checkpoint(exp_id, sdate, res))
                        checkpoint_en: esm_member_checkpoint_fn = model_module.esm_member_checkpoint
                        try:
                            checkpoint_en(expid, runtime_config_parser, res)
                        except COMPSsException:
                            logger.exception(f"Member [{sim}] checkpoint failed!")

    for start_date in ensemble_start_dates:
        for member_idx in range(1, ensemble_members + 1):
            member = str(member_idx)
            task_group = f"{expid}_{start_date}_{member}"
            try:
                compss_barrier_group(task_group)
            except COMPSsException:
                logging.exception(f"Aborting member: {start_date}_{member}")
                # Cancel the whole COMPSs group.
                compss_cancel_group(task_group)
                # clean generated data
                esm_member_disposal(start_date=start_date, member=member, top_working_dir=top_working_dir,
                                    output_dir=output_dir)


def _get_parser():
    parser = ArgumentParser(
        prog='esm_simulation',
        description='eFlows4HPC ESM simulation'
    )
    parser.add_argument('-e', '--expid', dest='expid', help='Optional experiment ID (random used if none).',
                        type=str, required=False)
    parser.add_argument('-m', '--model', dest='model', help='ESM model selected.',
                        choices=[m.value for m in Model.__members__.values()],
                        type=str, required=True)
    parser.add_argument('-c', '--config', dest='config', help='ESM configuration.',
                        type=Path, required=True)
    parser.add_argument('--start_dates', dest='start_dates', help='Start dates.',
                        type=str, required=True)
    parser.add_argument('--members', dest='members', help='Number of ensemble members.',
                        default=1, type=int, required=False)
    parser.add_argument('--processes', dest='processes', help='Number of cores for the simulation.',
                        type=int, required=False)
    parser.add_argument('--processes_per_node', dest='processes_per_node', help='Number of cores per node.',
                        type=int, required=False)
    parser.add_argument('--prune', dest='prune', help='Enable pruning.', action='store_true')
    parser.add_argument('--debug', dest='debug', help='Enable debug (more verbose) information.', action='store_true')
    return parser


def _create_expid() -> str:
    expid_alpha = random.choice(ascii_letters).lower()
    expid_numeric = str(random.randint(10000, 99999))
    return f"{expid_alpha}{expid_numeric}"


def main() -> None:
    args = _get_parser().parse_args()

    if args.debug:
        logger.root.setLevel(logging.DEBUG)

    # If no expid provided, we generate a random 6-digit ID.
    if args.expid is None or args.expid.strip() == '':
        args.expid = _create_expid()

    logger.info(f"Running simulation for model: {args.model}")

    # Users can specify a different ``esm_ensemble.conf``.
    model_config = args.config.expanduser()

    config_parser = _get_config(
        model_config=model_config,
        model=args.model,
        start_dates=args.start_dates,
        members=args.members,
        processes=args.processes,
        processes_per_node=args.processes_per_node)

    logger.info(f"Using eFlows4HPC configuration: {model_config}")

    if args.prune:
        logger.info(f"Starting the ESM analysis prune for {args.expid}...")
        esm_analysis_prune(args.expid)

    _run_esm(
        expid=args.expid,
        model=args.model,
        config_parser=config_parser)

    logger.info("All done. Bye!")


if __name__ == "__main__":
    main()
