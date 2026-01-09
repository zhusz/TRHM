#

import os
import unittest

from distutils.extension import Extension

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def test_all():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")
    return test_suite


# installationsDir = "/.../local/Installations/"
ext_modules = [
    CUDAExtension(
        "nerfacc_v053.csrc.nerfacc",
        [
            "nerfacc_v053/csrc/nerfacc.cpp",
            "nerfacc_v053/csrc/camera.cu",
            "nerfacc_v053/csrc/grid.cu",
            "nerfacc_v053/csrc/pdf.cu",
            "nerfacc_v053/csrc/scan.cu",
        ],
        # include_dirs=[installationsDir + "/" + "eigen/eigen3/"],
    ),
]

INSTALL_REQUIREMENTS = ["numpy", "torch", "torchvision"]

setup(
    description="nerfacc v0.5.3",
    author="Ruilong Li",
    author_email="RuilongLi'sEmail",
    license="",
    version="0.5.3",
    name="nerfacc_v053",
    # test_suite="setup.test_all",
    packages=["nerfacc_v053.csrc"],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
