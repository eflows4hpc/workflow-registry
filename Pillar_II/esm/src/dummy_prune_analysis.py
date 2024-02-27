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

# TODO: remove me later
MEMBERS_PRUNED = [2]

logger = logging.getLogger(__name__)


@on_failure(management='IGNORE', returns=0)
@task(expid=IN, member=IN, returns=bool, time_out=480)
def esm_analysis_prune(expid: str, member: int) -> bool:
    """This has to be called by PYCOMPSs as analysis.

    Args:
        expid: The experiment ID.
        member: The ensemble member.
    Returns:
        bool: True if the ensemble member was pruned.
    """
    # N.B.: importing this results in a network query to Hecuba servers,
    #       which fails if the servers are not available.

    # TODO: Suvi: call the AI code here
    if member not in MEMBERS_PRUNED:
        logging.info(f"We only prune the ensemble members #{str(MEMBERS_PRUNED)}!")
        return False

    logging.info(f"Pruning ensemble member #{str(member)}!")

    logging.info("Importing Hecuba")
    from hecuba import StorageDict  # type: ignore

    class MetaDictClass(StorageDict):
        """
        @TypeSpec dict <<keyname0:str>,valuename0:str>
        """
        # N.B. This is used by Hecuba. The ``get_by_alias`` method will
        #      look for the type specification of both keys and values
        #      in the docstring of the class. It needs this information
        #      to instantiate the object and link it with the table
        #      corresponding in Cassandra. So this method should be
        #      invoked on a class defined with the ``TypeSpec`` clause,
        #      not on the generic base class ``StorageDict``. Calling
        #      it on the generic base class will result in an error
        #      similar to:
        #
        #      RuntimeError: StorageDict: missed specification.
        #      Type of Primary Key or Column undefined

    # This is an infinite-loop, with a sleep time. The execution
    # must be wrapped in an existing COMPSs or Slurm job, with a
    # walltime or some limit to control the maximum execution
    # time, and kill this task.
    expid_member = f'{expid}_{str(member)}'
    while True:
        logging.info(f"MetaDictClass.get_by_alias('{expid_member}')")
        try:
            mdc = MetaDictClass.get_by_alias(expid_member)
            break
        except RuntimeError:
            logging.info(
                f"Simulation member {expid_member} not found sleeping +{CHECK_FOR_PRUNING_SLEEP_TIME_SECS} seconds...")
            sleep(CHECK_FOR_PRUNING_SLEEP_TIME_SECS)

    prune_sec = randint(10, 20)
    sleep(prune_sec)
    logging.info("Setting prune to TRUE!")
    mdc['prune'] = "true"
    logging.info(f"Pruned member [{str(member)}] from experiment [{expid}] after {prune_sec} seconds")

    return True


__all__ = ['esm_analysis_prune']
