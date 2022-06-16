#!/bin/bash
#
# Script Name: install_conmo_conda
#
# Version: 1.2   
#
# Author: mym.inv.uniovi@gmail.com
# Date : 15/06/2022
#
# Description: Creates a clean Conda environment with the necessary depedences to use Conmo.

CMDNAME=`basename ${BASH_SOURCE:-$0}`
if [ $# -ne 1 ]; then
    echo "Please set the name of the environment."
    echo "Usage: source $CMDNAME env_name" 
    exit 1
fi

# Exit immediately if a command exits with a non-zero status (fails)
set -e

# Configure conda options
eval "$(conda shell.bash hook)"
conda config --set always_yes yes

# Create enviroment with desired libraries installed
conda create --name $1 python=3.7.11 pandas=1.3.4 scikit-learn==0.24.2 tensorflow=2.4.1 pytest=6.2.5 numpy=1.21.2 scipy=1.7.1 joblib=1.1.0 pyarrow=3.0.0 isort h5py autopep8 sphinx sphinx_rtd_theme
