#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pkgutil
import re
import shutil
import sys
from distutils.command.build import build

from setuptools import setup
from setuptools.command.develop import develop


def get_version_from_package() -> str:
    with open("mesa_geo/__init__.py", "r") as fd:
        version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', fd.read(), re.MULTILINE
        ).group(1)
    return version


class DevelopCommand(develop):
    """Installation for development mode."""

    def run(self):
        get_mesa_templates(package="mesa", template_dir="visualization/templates")
        develop.run(self)


class BuildCommand(build):
    """Command for build mode."""

    def run(self):
        get_mesa_templates(package="mesa", template_dir="visualization/templates")
        build.run(self)


def get_mesa_templates(package, template_dir):
    pkg_dir = sys.modules[package].__path__[0]
    for subdir in os.listdir(os.path.join(pkg_dir, template_dir)):
        # do not copy modular_template.html to avoid being overwritten
        if os.path.isdir(os.path.join(pkg_dir, template_dir, subdir)):
            shutil.copytree(
                os.path.join(pkg_dir, template_dir, subdir),
                os.path.join("mesa_geo", template_dir, subdir),
                dirs_exist_ok=True,
            )


if __name__ == "__main__":
    setup(
        name="Mesa-Geo",
        version=get_version_from_package(),
        cmdclass={
            "develop": DevelopCommand,
            "build": BuildCommand,
        },
    )
