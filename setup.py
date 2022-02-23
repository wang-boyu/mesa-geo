#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

from setuptools import setup, find_packages
from codecs import open

requires = ["mesa >= 0.8.6", "geopandas", "libpysal", "rtree"]

version = ""
with open("mesa_geo/__init__.py", "r") as fd:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', fd.read(), re.MULTILINE
    ).group(1)

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="mesa-geo",
    version=version,
    description="Agent-based modeling (ABM) in Python 3+",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Project GeoMesa Team",
    author_email="",
    url="https://github.com/corvince/mesa-geo",
    packages=find_packages(),
    package_data={
        "mesa_geo": [
            "visualization/templates/*.html",
            "visualization/templates/css/*",
            "visualization/templates/fonts/*",
            "visualization/templates/js/*",
        ]
    },
    include_package_data=True,
    install_requires=requires,
    keywords="agent based modeling model ABM simulation multi-agent",
    license="Apache 2.0",
    zip_safe=False,
    classifiers=(
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Life",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
    ),
)
