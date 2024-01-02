"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import pathlib
import re

import setuptools
import torch.utils.cpp_extension as torch_cpp_ext

root = pathlib.Path(__name__).parent


def get_version(path):
    with open(path) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=", maxsplit=1)[1].replace('"', "").strip()
    raise ValueError("Version not found")


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


enable_bf16 = True
for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
    arch = int(re.search("compute_\d+", cuda_arch_flags).group()[-2:])
    if arch < 75:
        raise RuntimeError("FlashInfer requires sm75+")
    elif arch == 75:
        # disable bf16 for sm75
        enable_bf16 = False

if enable_bf16:
    torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_BF16")

remove_unwanted_pytorch_nvcc_flags()
ext_modules = []
ext_modules.append(
    torch_cpp_ext.CUDAExtension(
        name="flashinfer.ops._kernels",
        sources=[
            "csrc/single_decode.cu",
            "csrc/single_prefill.cu",
            "csrc/cascade.cu",
            "csrc/batch_decode.cu",
            "csrc/flashinfer_ops.cu",
            "csrc/batch_prefill.cu",
        ],
        include_dirs=[
            str(root.resolve().parent / "include"),
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3"],
        },
    )
)

setuptools.setup(
    name="flashinfer",
    version=get_version(root / "flashinfer/__init__.py"),
    packages=setuptools.find_packages(),
    author="FlashInfer team",
    license="Apache License 2.0",
    description="FlashInfer: Kernel Library for LLM Serving",
    url="https://github.com/flashinfer-ai/flashinfer",
    python_requires=">=3.9",
    ext_modules=ext_modules,
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
)
