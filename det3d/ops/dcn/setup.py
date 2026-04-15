import os
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _ensure_torch_cuda_arch_list():
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        arch_list = sorted({line.strip() for line in output.splitlines() if line.strip()})
        if arch_list:
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)
            return
    except Exception:
        pass

    # Fall back to Turing so builds still work in environments where torch
    # cannot query a visible GPU during extension compilation.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"


_ensure_torch_cuda_arch_list()

setup(
    name='masked_conv',
    ext_modules=[
        CUDAExtension('deform_conv_cuda', [
            'src/deform_conv_cuda.cpp',
            'src/deform_conv_cuda_kernel.cu',
        ],
        define_macros=[('WITH_CUDA', None)],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
        ]})],
        cmdclass={'build_ext': BuildExtension})
