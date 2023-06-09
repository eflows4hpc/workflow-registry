# -*- coding: utf-8 -*-
"""setup.py: setuptools control."""
from setuptools import setup
from setuptools import Extension

import inspect
import os
import sys

# Import the version string.
from backtrackbb.external.version.version import get_git_version

with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")


setup(
    name="backtrackbb",
    packages=[
        "backtrackbb",
        "backtrackbb.external.configobj",
        "backtrackbb.external.nllgrid",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "btbb = backtrackbb.btbb:main",
            "btbb_continuous = backtrackbb.btbb_continuous:main",
            "btbb_distrostream = backtrackbb.btbb_distrostream:main",
            "mbf_plot = backtrackbb.mbf_plot:main",
            "bt2eventdata = backtrackbb.bt2eventdata:main",
            "group_triggers = backtrackbb.group_triggers:main",
        ]
    },
    version=get_git_version(),
    ext_package="backtrackbb.libs",
    ext_modules=[
        Extension(
            name="lib_rec_filter", sources=["backtrackbb/c_libs/lib_rec_filter.c"]
        ),
        Extension(name="lib_rec_rms", sources=["backtrackbb/c_libs/lib_rec_rms.c"]),
        Extension(name="lib_rec_hos", sources=["backtrackbb/c_libs/lib_rec_hos.c"]),
        Extension(name="lib_rec_cc", sources=["backtrackbb/c_libs/lib_rec_cc.c"]),
        Extension(
            name="lib_map_project",
            sources=[
                "backtrackbb/c_libs/map_project/util.c",
                "backtrackbb/c_libs/map_project/map_project.c",
                "backtrackbb/c_libs/map_project/coord_convert.c",
            ],
        ),
        Extension(
            name="lib_rosenberger",
            sources=[
                "backtrackbb/c_libs/rosenberger/IA_Kdiag.c",
                "backtrackbb/c_libs/rosenberger/IA_Ealloc.c",
                "backtrackbb/c_libs/rosenberger/IA_R2upd.c",
                "backtrackbb/c_libs/rosenberger/rosenberger.c",
            ],
        ),
    ],
    description="Multi-band array detection and location of seismic sources",
    long_description=long_descr,
    author="Natalia Poiata",
    author_email="poiata@ipgp.fr",
    url="http://backtrackbb.github.io",
    license="CeCILL Free Software License Agreement, Version 2.1",
    platforms="OS Independent",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: CeCILL Free Software License "
        "Agreement, Version 2.1",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=["obspy>=1.0.0", "scipy>=0.17", "pyproj"],
)
