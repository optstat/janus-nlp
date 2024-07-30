#!/bin/bash

# Create the virtual environment if it doesn't exist
if [ ! -d "build/venv" ]; then
    python3 -m venv build/venv
fi

# Activate the virtual environment
source build/venv/bin/activate

# Install necessary packages
pip3 install setuptools wheel numpy scipy torch smac

# Download and extract LibTorch
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip"
LIBTORCH_ZIP="build/libtorch.zip"
LIBTORCH_DIR="build/libtorch"

if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Downloading LibTorch..."
    wget $LIBTORCH_URL -O $LIBTORCH_ZIP
    echo "Extracting LibTorch..."
    unzip $LIBTORCH_ZIP -d build
fi

# Set environment variables for CMake
export Torch_DIR=$(pwd)/build/libtorch/share/cmake/Torch
export Torch_LIBRARY_DIR=$(pwd)/build/libtorch/lib
export Torch_INCLUDE_DIR=$(pwd)/build/libtorch/include



# Clean previous builds
rm -rf build/lib

# Run the setup script
python3 setup.py build_ext --verbose
pip install .


# Deactivate the virtual environment
deactivate
