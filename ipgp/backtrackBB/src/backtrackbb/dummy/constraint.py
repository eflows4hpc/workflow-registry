# -*- coding: utf-8 -*-
"""
PyCOMPSs API - dummy - constraint.

This file contains the dummy class constraint used as decorator.

Taken from PyCOMPSs binding (https://github.com/bsc-wdc/compss).
* Removed typing.
"""

from backtrackbb.dummy._decorator import _Dummy as Dummy

Constraint = Dummy
constraint = Dummy  # pylint: disable=invalid-name
