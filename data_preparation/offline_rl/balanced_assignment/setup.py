from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

torch_lib_dir = str(Path(torch.__file__).resolve().parent / "lib")

setup(
    name='balanced_assignment',
    ext_modules=[
        CppExtension(
            'balanced_assignment',
            ['ba.cpp'],
            runtime_library_dirs=[torch_lib_dir],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
