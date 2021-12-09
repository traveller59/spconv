#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import shutil
import sys
from pathlib import Path
from shutil import rmtree
from typing import List

import pccm
from pccm.extension import ExtCallback, PCCMBuild, PCCMExtension
from setuptools import Command, find_packages, setup
from setuptools.extension import Extension
from ccimport import compat
import subprocess 
import re 

# Package meta-data.
NAME = 'spconv'
RELEASE_NAME = NAME
deps = ["cumm"]
cuda_ver = os.environ.get("CUMM_CUDA_VERSION", "")
# is_ci_build = cuda_ver != ""
# if not cuda_ver:
#     nvcc_version = subprocess.check_output(["nvcc", "--version"
#                                             ]).decode("utf-8").strip()
#     nvcc_version_str = nvcc_version.split("\n")[3]
#     version_str: str = re.findall(r"release (\d+.\d+)",
#                                     nvcc_version_str)[0]
#     cuda_ver = version_str

if cuda_ver:
    cuda_ver = cuda_ver.replace(".", "") # 10.2 to 102

    RELEASE_NAME += "-cu{}".format(cuda_ver)
    deps = ["cumm-cu{}>=0.2.8".format(cuda_ver)]
else:
    deps = ["cumm>=0.2.8"]



DESCRIPTION = 'spatial sparse convolution'
URL = 'https://github.com/traveller59/spconv'
EMAIL = 'yanyan.sub@outlook.com'
AUTHOR = 'Yan Yan'
REQUIRES_PYTHON = '>=3.6'
VERSION = None

# What packages are required for this module to be executed?
REQUIRED = ["pccm>=0.2.21", "pybind11>=2.6.0", "fire", "numpy", *deps]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(str(Path(__file__).parent))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open('version.txt', 'r') as f:
        version = f.read().strip()
else:
    version = VERSION
cwd = os.path.dirname(os.path.abspath(__file__))


def _convert_build_number(build_number):
    parts = build_number.split(".")
    if len(parts) == 2:
        return "{}{:03d}".format(int(parts[0]), int(parts[1]))
    elif len(parts) == 1:
        return build_number
    else:
        raise NotImplementedError


env_suffix = os.environ.get("SPCONV_VERSION_SUFFIX", "")
if env_suffix != "":
    version += ".dev{}".format(_convert_build_number(env_suffix))
version_path = os.path.join(cwd, NAME, '__version__.py')
about['__version__'] = version

with open(version_path, 'w') as f:
    f.write("__version__ = '{}'\n".format(version))

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(
            sys.executable))

        self.status('Uploading the package to PyPI via Twine...')
        os.system('twine upload dist/*')

        self.status('Pushing git tags...')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()



disable_jit = os.getenv("SPCONV_DISABLE_JIT", None)

if disable_jit is not None and disable_jit == "1":
    cmdclass = {
        'upload': UploadCommand,
        'build_ext': PCCMBuild,
    }
    from cumm.gemm.main import GemmMainUnitTest
    from spconv.core import SHUFFLE_SIMT_PARAMS, SHUFFLE_VOLTA_PARAMS, SHUFFLE_TURING_PARAMS
    from spconv.core import IMPLGEMM_SIMT_PARAMS, IMPLGEMM_VOLTA_PARAMS, IMPLGEMM_TURING_PARAMS
    from cumm.conv.main import ConvMainUnitTest
    from cumm.constants import CUMM_CPU_ONLY_BUILD
    from spconv.csrc.sparse.all import SpconvOps
    from spconv.csrc.utils import BoxOps
    from spconv.csrc.hash.core import HashTable
    from cumm.common import CompileInfo

    cu = GemmMainUnitTest(SHUFFLE_SIMT_PARAMS + SHUFFLE_VOLTA_PARAMS + SHUFFLE_TURING_PARAMS)
    convcu = ConvMainUnitTest(IMPLGEMM_SIMT_PARAMS + IMPLGEMM_VOLTA_PARAMS + IMPLGEMM_TURING_PARAMS)
    convcu.namespace = "cumm.conv.main"

    cu.namespace = "cumm.gemm.main"
    std = "c++17"
    if cuda_ver:
        cuda_ver_number = int(cuda_ver)
        if cuda_ver_number < 110:
            std = "c++14" 
        else:
            std = "c++17"
    cus = [cu, convcu, SpconvOps(), BoxOps(), HashTable(), CompileInfo()]
    if CUMM_CPU_ONLY_BUILD:
        cus = [SpconvOps(), BoxOps(), HashTable(), CompileInfo()]
    ext_modules: List[Extension] = [
        PCCMExtension(cus,
                      "spconv/core_cc",
                      Path(__file__).resolve().parent / "spconv",
                      objects_folder="objects",
                      std=std,
                      disable_pch=True,
                      verbose=True)
    ]
else:
    cmdclass = {
        'upload': UploadCommand,
    }
    ext_modules = []

# Where the magic happens:
setup(
    name=RELEASE_NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests', )),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    entry_points={
        'console_scripts': [],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='Apache License 2.0',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    # $ setup.py publish support.
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
