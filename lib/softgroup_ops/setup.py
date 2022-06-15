from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='SG_OP',
    ext_modules=[
        CUDAExtension(
            'SG_OP', ['src/softgroup_api.cpp', 'src/softgroup_ops.cpp', 'src/cuda.cu'],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']
            })
    ],
    cmdclass={'build_ext': BuildExtension})
