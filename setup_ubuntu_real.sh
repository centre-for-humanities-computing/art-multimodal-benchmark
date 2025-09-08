#!/bin/bash

sudo apt-get update

sudo apt-get install python3-venv

# Create virtual environment in ./env
python3 -m venv env

# Activate the environment
source env/bin/activate

# Upgrade pip inside the env
python -m pip install --upgrade pip

# Install dependencies
python -m pip install -r requirements.txt
python -m pip install ipykernel

# Install Jupyter kernel so this env can be selected in notebooks
python -m ipykernel install --user --name=benchmark_env

# Deactivate the environment (optional)
deactivate
