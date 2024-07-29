#!/bin/bash

# Create the virtual environment if it doesn't exist
if [ ! -d "build/venv" ]; then
    python3 -m venv build/venv
fi

# Activate the virtual environment
source build/venv/bin/activate

# Install necessary packages
pip install setuptools wheel numpy scipy torch

# Run the setup script
python setup.py build_ext --inplace

# Deactivate the virtual environment
deactivate
