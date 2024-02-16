import logging
from random import randint
from time import sleep

from pycompss.api.api import TaskGroup  # type: ignore
from pycompss.api.api import compss_barrier_group  # type: ignore
from pycompss.api.api import compss_cancel_group  # type: ignore
from pycompss.api.api import compss_wait_on  # type: ignore
from pycompss.api.exceptions import COMPSsException  # type: ignore
from pycompss.api.on_failure import on_failure  # type: ignore
from pycompss.api.parameter import IN  # type: ignore
from pycompss.api.task import task  # type: ignore
from pycompss.runtime.management.classes import Future  # type: ignore

"""eFlows4HPC ensemble members pruning."""

CHECK_FOR_PRUNING_SLEEP_TIME_SECS = 5
"""How long should we sleep for"""

logger = logging.getLogger(__name__)


@task(expid=IN)
def esm_analysis_prune(expid: str):
    """This has to be called by PYCOMPSs as analysis.

    Args:
        expid: The experiment ID.
    """
    # N.B.: importing this results in a network query to Hecuba servers,
    #       which fails if the servers are not available.
    from hecuba import StorageDict  # type: ignore
    # This is an infinite-loop, with a sleep time. The execution
    # must be wrapped in an existing COMPSs or Slurm job, with a
    # walltime or some limit to control the maximum execution
    # time, and kill this task.
    while True:
        mdc = StorageDict.get_by_alias(expid)
        if mdc:
            break
        logging.info(f"Simulation {expid} not found sleeping +{CHECK_FOR_PRUNING_SLEEP_TIME_SECS} seconds...")
        sleep(CHECK_FOR_PRUNING_SLEEP_TIME_SECS)

    prune_sec = randint(10, 20)
    sleep(prune_sec)
    mdc['prune'] = "true"
    logging.info(f"Pruned {expid} after {prune_sec} seconds")


__all__ = ['esm_analysis_prune']
