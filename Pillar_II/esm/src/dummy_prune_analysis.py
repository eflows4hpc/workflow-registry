from hecuba import StorageDict
import time
import random

class MetaDictClass(StorageDict):
   '''
   @TypeSpec dict <<keyname0:str>,valuename0:str>
   '''

def simulation_started(expid):
    state = MetaDictClass.get_by_alias(expid) 
    if len([k for k in s.keys()]) > 1:
        return True
    else:
        return False

def esm_analysis_prune(expid):
    
    while True:
        started = simulation_started(expid)
        if started is True:
            break
        print("Simulation {expid} not found sleeping +5 seconds")
        time.sleep(5)
    
        mdc = MetaDictClass.get_by_alias(expid)
        prune_sec = random.randint(10,20) 
        time.sleep(prune_sec) 
        mdc['prune'] = "true"
        return f"Pruned {expid} after {prune_sec} seconds"

# esm_analysis_prune(expid) has to be called by pycompss as analysis         
