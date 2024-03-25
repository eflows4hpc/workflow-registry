from modules.data_target import DataTarget
import yaml
import numpy as np
import pandas as pd
import os
from dislib.data.array import Array


class Disk(DataTarget):
    """
    Object that manages the writings and readings to a disk

    This class should be instantiated directly and passed as argument to a data_manager
    or other objects.

    Parameters
    ----------
    route : string
        Path where the read and write files are
    read_file : string
        Name of the file to read data from
    write_file : string
        Name of the file to write data into
    """

    def __init__(self, route, read_file=None, write_file=None):
        if os.path.isfile(route):
            res = np.char.rpartition(route, '/')
            super().__init__(res[0])
            self.read_file = res[-1]
            self.write_file = res[-1]
        else:
            super().__init__(route)
            if not os.path.isdir(route):
                os.makedir(route)
            if read_file is not None and os.path.isfile(route + "/" + read_file):
                self.read_file = read_file
            else:
                self.read_file = None
            if write_file is not None:
                self.write_file = write_file
            else:
                self.write_file = None

    def get_data(self):
        """
        Reads data from the specified file and path
        :return: data
        """
        if self.read_file is None:
            raise ValueError("There is no specified file to read data from.")
        file_format = np.char.rpartition(self.read_file, '.')
        file_format = file_format[-1]
        if file_format == "npy":
            data = np.load(self.route + "/" + self.read_file)
        elif file_format == "yaml":
            with open(self.route + "/" + self.read_file, 'r') as file:
                data = yaml.safe_load(file)
        elif file_format == "csv":
            data = pd.read_csv(self.route + "/" + self.read_file)
        elif file_format == "txt":
            file = open(self.route + "/" + self.read_file, 'r')
            data = file.read()
        return data

    def set_data(self, data):
        """
        Writes data in the specified data for writing
        data: data to write
        """
        if data is None:
            raise ValueError("Data to write should not be none.")
        if self.write_file is None:
            raise ValueError("There is no specified file for writing data.")
        file_format = np.char.rpartition(self.write_file, '.')
        file_format = file_format[-1]
        if isinstance(data, Array):
            pass
        elif file_format == "npy":
            if isinstance(data, np.ndarray):
                with open(self.route + "/" + self.write_file, 'wb') as f:
                    np.save(f, data)
            else:
                raise ValueError("Wrong data format.")
        elif file_format == "csv":
            data.to_csv(self.route + "/" + self.write_file)
        elif file_format == "txt":
            f = open(self.route + "/" + self.write_file, "a")
            f.write(data)
            f.close()

    def read(self, source):
        """
        Set file to read data from, and reads the data
        source: string, name of the file to read the data from.
        returns: data
        """
        self.set_file_to_read(source)
        return self.get_data()

    def write(self, source, data):
        """
         Sets file to write data int and writes the data
         source: string, name of the file to write the data into.
         """
        self.set_file_to_write(source)
        self.set_data(data)

    def clear_data(self):
        """
        Deletes the data from the file to write
        """
        if self.write_file is None:
            raise ValueError("There is no specified file for writing data.")
        file_format = np.char.rpartition(self.write_file, '.')
        file_format = file_format[-1]
        if file_format == "npy":
            pass
        elif file_format == "csv":
            pass
        elif file_format == "txt":
            f = open(self.route+self.read_file, "a")
            f.seek(0)
            f.truncate()
            f.close()

    def set_file_to_read(self, read_file):
        """
        Sets file to read data from
        write_file: string, name of the file to read the data from.
        """
        res = np.char.rpartition(read_file, '/')
        if os.path.isfile(self.route + "/" + res[-1]):
            self.read_file = read_file
        elif os.path.isfile(read_file):
            self.read_file = res[-1]
            self.route = res[0]
        else:
            raise ValueError("The file specified does not exist in the "
                             "working directory neither in the "
                             "specified route")

    def set_file_to_write(self, write_file):
        """
        Sets file to write data into
        write_file: string, name of the file to write the data into.
        """
        if os.path.isfile(write_file):
            res = np.char.rpartition(write_file, '/')
            self.route = res[0]
            self.write_file = res[-1]
        elif os.path.isfile(self.route + "/" + write_file):
            self.write_file = write_file
        elif os.path.exists(self.route):
            self.write_file = write_file
        else:
            raise ValueError("The file specified does not exist in the "
                             "working directory neither in the specified route")
