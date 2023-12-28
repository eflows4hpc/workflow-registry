import logging
import random
from argparse import ArgumentParser, Namespace
from configparser import ConfigParser
from enum import Enum, unique
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, List

from pycompss.api.api import TaskGroup  # type: ignore
from pycompss.api.api import compss_wait_on  # type: ignore
from pycompss.api.exceptions import COMPSsException  # type: ignore
from pycompss.api.parameter import IN  # type: ignore
from pycompss.api.task import task  # type: ignore

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.NOTSET)

logger = logging.getLogger(__name__)

# Every model must define some basic operations to be used in the ESM. We load
# these dynamically. Their type definitions are below:

init_top_working_dir_fn = Callable[[Path, int, List[str], ConfigParser], None]
init_output_dir_fn = Callable[[Path, int, List[str]], None]
esm_simulation_fn = Callable[[str, str], Any]
esm_member_checkpoint_fn = Callable[[str, ConfigParser, Any], bool]


@unique
class Model(str, Enum):
    """eFlows4HPC available models. Must match imported Python modules."""
    FESOM2 = 'fesom2'
    AWICM3 = 'awicm3'


def _get_config(model: str, model_config: Path) -> ConfigParser:
    """Get the eFlows4HPC configuration object.

    Args:
        model: The model name (see ``Model`` enum).
        model_config: The path to the model configuration file.
    """
    if not model_config.exists() or not model_config.is_file():
        raise ValueError(f"Model {model} configuration file not located at {model_config}")

    config = ConfigParser(inline_comment_prefixes="#")
    config.read(model_config)
    return config


def _init_top_working_directory(
        top_working_dir: Path,
        access_rights: int,
        start_dates: List[str],
        config: ConfigParser) -> None:
    """This function does most of the heavy-work to initialize the top working directory.

    In this directory we store files needed to run the model.

    Args:
        top_working_dir: The model top working directory, where files needed to run the model reside for each member.
        access_rights: Default file mode to create new file system entries (e.g. 0x755)
        start_dates: The list of start dates for the model run.
        config: eFlows4HPC configuration.
    """
    model_module = import_module(f"{config['runtime']['model']}")
    fn: init_top_working_dir_fn = model_module.init_top_working_dir
    fn(top_working_dir,
       access_rights,
       start_dates,
       config)


def _init_output_directory(
        top_working_dir: Path,
        access_rights: int,
        start_dates: List[str],
        config: ConfigParser) -> None:
    """Prepare the output directories for the model run.

    Args:
        top_working_dir: The model top working directory, where files needed to run the model reside for each member.
        access_rights: Default file mode to create new file system entries (e.g. 0x755)
        start_dates: The list of start dates for the model run.
        config: eFlows4HPC configuration.
    """
    model_module = import_module(f"{config['runtime']['model']}")
    fn: init_output_dir_fn = model_module.init_output_dir
    fn(top_working_dir,
       access_rights,
       start_dates)


def esm_ensemble_init(args: Namespace) -> ConfigParser:
    logger.info(f"Initializing experiment {args.expid}")

    if args.config:
        model_config = args.config.expanduser()
    else:
        cwd_path = Path(__file__).parent.resolve()
        model_config = cwd_path / args.model / 'esm_ensemble.conf'

    config = _get_config(args.model, model_config)
    config['runtime']["expid"] = args.expid
    config['runtime']['model'] = args.model
    logger.info(f"Using eFlows4HPC configuration: {model_config}")

    access_rights = int(config['common']['new_dir_mode'], 8)
    logger.info(f"New directories will be created with the file mode: {oct(access_rights)}")

    top_working_dir = Path(config['common']['top_working_dir'], args.expid)
    output_dir = Path(config['common']['output_dir'], args.expid)
    start_dates = config['common']['ensemble_start_dates'].split(" ")

    _init_top_working_directory(top_working_dir, access_rights, start_dates, config)
    _init_output_directory(output_dir, access_rights, start_dates, config)

    return config


def _run_esm(args: Namespace) -> None:
    # This is the only step that is common to the models. But we can move
    # that to a common module and call the model's code directly instead
    # if needed too. For now this is good enough.
    config_parser: ConfigParser = compss_wait_on(esm_ensemble_init(args))
    logger.info("ESM Initialization complete!")

    # This is where we delegate the ESM execution to a model's module code
    # (e.g. FESOM2, AWICM3, etc.).
    start_dates = config_parser['common']['ensemble_start_dates'].split(" ")
    top_working_dir = Path(config_parser['common']['top_working_dir'], args.expid)
    model_module = import_module(f"{config_parser['runtime']['model']}")

    for start_date in start_dates:
        with TaskGroup(f"{args.expid}_{start_date}", implicit_barrier=False):
            # Launch each SIM, create an implicit dependence by passing the result to the next task (checkpoint)
            number_simulations = int(config_parser['common']['chunks'])
            logger.info(f"Total of chunks configured: {number_simulations}")

            for sim in range(1, number_simulations + 1):
                working_dir_exe = top_working_dir / start_date
                log_file = working_dir_exe / f"fesom2_{args.expid}_{start_date}_{str(sim)}.out"
                logger.info(f"Launching simulation {start_date}.{str(sim)} in {working_dir_exe}")
                simulation_fn: esm_simulation_fn = model_module.esm_simulation
                res: int = simulation_fn(str(log_file), str(working_dir_exe))
                logger.info(f"Simulation binary execution return code: {res}")
                # check the state of the member, for the first one there is nothing to check
                if sim > 1:
                    logger.info(f"Checkpoint member: {str(sim)}")
                    # Note: The barrier was added to check if it solves the pruning issue.
                    #       Adding a barrier causes all tasks to run in serial even if
                    #       these are in different task groups.
                    # compss_wait_on(model_module.esm_member_checkpoint(exp_id, sdate, res))
                    checkpoint_en: esm_member_checkpoint_fn = model_module.esm_member_checkpoint
                    try:
                        checkpoint_en(args.expid, config_parser, res)
                    except COMPSsException:
                        logger.exception(f"Member [{sim}] checkpoint failed!")


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
                        type=Path, required=False)
    parser.add_argument('--debug', dest='debug', help='Enable debug (more verbose) information.', action='store_true')
    return parser


def main() -> None:
    args = _get_parser().parse_args()

    if args.debug:
        logger.root.setLevel(logging.DEBUG)

    # If no expid provided, we generate a random 6-digit ID.
    if args.expid is None or args.expid.strip() == '':
        args.expid = str(random.randint(100000, 999999))

    logger.info(f"Running simulation for model: {args.model}")

    _run_esm(args)

    logger.info("All done. Bye!")


if __name__ == "__main__":
    main()
