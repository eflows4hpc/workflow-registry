# -*- coding: utf-8 -*-
"""
PyCOMPSs API - dummy - task.

This file contains the dummy class task used as decorator.

Taken from PyCOMPSs binding (https://github.com/bsc-wdc/compss).
* Removed typing.
"""

from backtrackbb.dummy._decorator import _Dummy as Dummy

Task = Dummy
task = Dummy  # pylint: disable=invalid-name
