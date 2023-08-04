# -*- coding: utf-8 -*-

"""
PyCOMPSs API - dummy - decorator.

This file contains the dummy class task used as decorator.

Taken from PyCOMPSs binding (https://github.com/bsc-wdc/compss).
* Removed typing.
"""


class _Dummy:
    """Dummy task class (decorator style)."""

    def __init__(self, *args, **kwargs):
        """Construct a dummy Task decorator.

        :param args: Task decorator arguments.
        :param kwargs: Task decorator keyword arguments.
        :returns: None
        """
        self.args = args
        self.kwargs = kwargs

    def __call__(self, function):
        """Invoke the dummy decorator.

        :param function: Decorated function.
        :returns: Result of executing the given function.
        """

        def wrapped_f(*args, **kwargs):
            # returns may appear in @task decorator
            if "returns" in kwargs:
                kwargs.pop("returns")
            return function(*args, **kwargs)

        return wrapped_f

    def __repr__(self):
        attributes = f"(args: {repr(self.args)}, kwargs: {repr(self.kwargs)})"
        return f"Dummy {self.__class__.__name__} decorator {attributes}"
