from hecuba import StorageDict  # type: ignore


class fesom_outputs(StorageDict):
    '''
    @TypeSpec dict <<lat:float,ts:int>,metrics:numpy.ndarray>
    '''
