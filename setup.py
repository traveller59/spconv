import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion
from pathlib import Path

import torch
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# if 'LIBTORCH_ROOT' not in os.environ:
#     raise ValueError("You must set LIBTORCH_ROOT to your torch c++ library.")

LIBTORCH_ROOT = str(Path(torch.__file__).parent)

SPCONV_FORCE_BUILD_CUDA = os.getenv("SPCONV_FORCE_BUILD_CUDA")

PYTHON_VERSION = "{}.{}".format(sys.version_info.major, sys.version_info.minor)

remove_device = re.search(r"(\+|\.)(dev|cu|cpu)", torch.__version__)
PYTORCH_VERSION = torch.__version__
if remove_device is not None:
    PYTORCH_VERSION = torch.__version__[:remove_device.start()]
PYTORCH_VERSION = list(map(int, PYTORCH_VERSION.split(".")))
PYTORCH_VERSION_NUMBER = PYTORCH_VERSION[0] * 10000 + PYTORCH_VERSION[1] * 100 + PYTORCH_VERSION[2]
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
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.13.0':
                raise RuntimeError("CMake >= 3.13.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [# '-G "Visual Studio 15 2017 Win64"',
                      '-DCMAKE_PREFIX_PATH={}'.format(LIBTORCH_ROOT),
                      '-DPYBIND11_PYTHON_VERSION={}'.format(PYTHON_VERSION),
                      '-DSPCONV_BuildTests=OFF',
                      '-DPYTORCH_VERSION={}'.format(PYTORCH_VERSION_NUMBER)
                      ] #  -arch=sm_61
        if not torch.cuda.is_available() and SPCONV_FORCE_BUILD_CUDA is None:
            cmake_args += ['-DSPCONV_BuildCUDA=OFF']
        else:
            cuda_flags = ["\"--expt-relaxed-constexpr\""]
            # must add following flags to use at::Half
            # but will remove raw half operators.
            cuda_flags += ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__"]
            # cuda_flags += ["-D__CUDA_NO_HALF2_OPERATORS__"] 
            cmake_args += ['-DCMAKE_CUDA_FLAGS=' + " ".join(cuda_flags)]
        cfg = 'Debug' if self.debug else 'Release'
        assert cfg == "Release", "pytorch ops don't support debug build."
        build_args = ['--config', cfg]
        print(cfg)
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), str(Path(extdir) / "spconv"))]
            # cmake_args += ['-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), str(Path(extdir) / "spconv"))]
            cmake_args += ['-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), str(Path(extdir) / "spconv"))]
            cmake_args += ["-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE"]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(str(Path(extdir) / "spconv"))]
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print("|||||CMAKE ARGS|||||", cmake_args)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


packages = find_packages(exclude=('tools', 'tools.*'))
setup(
    name='spconv',
    version='1.2.1',
    author='Yan Yan',
    author_email='scrin@foxmail.com',
    description='spatial sparse convolution for pytorch',
    long_description='',
    setup_requires = ['torch>=1.3.0'],
    packages=packages,
    package_dir = {'spconv': 'spconv'},
    ext_modules=[CMakeExtension('spconv', library_dirs=[])],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
