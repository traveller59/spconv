import os
import re
import sys
import platform
import subprocess
import torch
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

from pathlib import Path

# if 'LIBTORCH_ROOT' not in os.environ:
#     raise ValueError("You must set LIBTORCH_ROOT to your torch c++ library.")

LIBTORCH_ROOT = str(Path(torch.__file__).parent)

PYTHON_VERSION = "{}.{}".format(sys.version_info.major, sys.version_info.minor)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', library_dirs=[]):
        Extension.__init__(self, name, sources=[], library_dirs=library_dirs)
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            raise NotImplementedError

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        print(extdir)
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir + "/spconv",
                      '-DCMAKE_PREFIX_PATH=' + LIBTORCH_ROOT,
                      '-DPYBIND11_PYTHON_VERSION={}'.format(PYTHON_VERSION),
                      '-DSPCONV_BuildTests=OFF',
                      '-DCMAKE_CUDA_FLAGS="--expt-relaxed-constexpr"']

        cfg = 'Debug' if self.debug else 'Release'
        # cfg = 'Debug'
        build_args = ['--config', cfg]
        print(cfg)
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


packages = find_packages(exclude=('tools', 'tools.*'))
setup(
    name='spconv',
    version='1.0',
    author='Yan Yan',
    author_email='scrin@foxmail.com',
    description='spatial sparse convolution for pytorch',
    long_description='',
    setup_requires = ['torch>=1.0.0'],
    packages=packages,
    package_dir = {'spconv': 'spconv'},
    ext_modules=[CMakeExtension('spconv', library_dirs=[])],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)

