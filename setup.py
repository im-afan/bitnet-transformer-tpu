from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='transformer_cuda',
    ext_modules=[
        CUDAExtension('transformer_cuda', [
            'src/cuda/bindings.cpp',
            'src/cuda/bindings_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
