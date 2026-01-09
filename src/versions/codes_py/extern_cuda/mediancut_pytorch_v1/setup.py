# (local)
import os
import unittest

from distutils.extension import Extension

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def test_all():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")
    return test_suite


# installationsDir = "/Users/zhusz/local/r/willow_zhusz/local/Installations/"
installationsDir = "/home/zhusz/local/Installations/"
ext_modules = [
    CppExtension(
        "mediancut_v1.csrc.mediancut_main",
        [
            "mediancut_v1/csrc/mediancut_main.cc",
            # "mediancut_v1/csrc/spherical_harmonics.cc",
            # "mediancut_v1/csrc/default_image.cc",
        ],
        include_dirs=[installationsDir + "/" + "eigen/eigen3/"],
    ),
]

INSTALL_REQUIREMENTS = ["numpy", "torch", "torchvision"]

setup(
    description="MedianCutV1",
    author="Shizhan Zhu",
    author_email="zhshzhutah2@gmail.com",
    license="",
    version="1.0.0",
    name="mediancut_v1",
    test_suite="setup.test_all",
    packages=["mediancut_v1.csrc"],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
