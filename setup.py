
import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "janus_nlp"


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = [
    #    "-L/usr/local/lib",   # Link to the directory where the libraries are located
    #    "-L/home/panos/Applications/libtorch/lib",  # Link to the directory where the libtorch libraries are located
    #    "-lradaute",          # Link to libradaute.so
    #    "-lradauted"          # Link to libradauted.so
    ]

    
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-g0",
            "-march=native",
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            "-lineinfo" if debug_mode else "",  # Include CUDA line info in debug builds
        ],
    }

    # Remove any empty strings from the compiler arguments
    extra_compile_args["cxx"] = [arg for arg in extra_compile_args["cxx"] if arg]
    extra_compile_args["nvcc"] = [arg for arg in extra_compile_args["nvcc"] if arg]

    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.abspath(os.path.dirname(__file__))
    print(f"This directory: {this_dir}")
    extensions_dir = os.path.join(this_dir, "src", "cpp")
    print(f"Extensions directory: {extensions_dir}")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    print(f"CPP sources: {sources}")


    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Janus NLP",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    cmdclass={"build_ext": BuildExtension},
)