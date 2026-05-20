from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# os.environ["CC"] = "usr/bin/gcc-12"
# os.environ["CXX"] = "usr/bin/g++-12"

setup(
    name='transformer_cuda',
    ext_modules=[
        CUDAExtension(
            'transformer_cuda', 
            [
                'src/cuda/bindings.cpp',
                'src/cuda/bindings_kernel.cu',
            ],
            extra_compile_args={
                "cxx": ["std=c++17"],
                "nvcc": [
                    # "-ccbin=/usr/bin/g++-12"
                    "std=c++17"
                    "--expt-relaxed-constexpr"
                ]
            }
            ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    }
)
