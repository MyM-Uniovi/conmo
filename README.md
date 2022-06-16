# Conmo
Conmo is a framework developed in Python whose main objective is to facilitate the execution and comparison of different experiments mainly related to Anomaly Detection and Condition Monitoring problems.
These experiments consist of a series of concatenated stages forming a pipeline architecture, i.e. the output of one stage is the input of the next one.
This framework aims to provide a way to standarize machine learning experiments, thus being able to reconstruct result tables of scientific papers.

## Requirements
Conmo works properly in Python versions 3.7 and 3.8. It has not yet been tested for operation with newer or deprecated versions.
To be able to start working with Conmo you need to have installed this list of libraries:

* Numpy 
* Pandas
* Tensorflow
* Scikit-Learn
* Scipy
* Requests
* Pyarrow

If you want to make a contribution by modifying code and documentation you need to include these libraries as well:

* Sphinx 
* Sphinx-rtd-theme
* Isort
* Autopep8

## Installation
There are currently two ways to install Conmo:

### Package manager Pip
The easiest way is to use the pip command so that it's installed together with all its dependencies.

```bash
  pip install conmo
```

### From source code
You can also download this repository and then create a virtual environment to install the dependencies in.
We recommend this option if you plan to contribute to Conmo.

```bash
git clone https://github.com/MyM-Uniovi/conmo.git
cd conmo
```

In /scripts folder we provide a bash script to prepare a Conda environment ready for running Conmo:

```bash
cd scripts
./install_conmo_conda.sh
```

In case you are not using a Linux distribution and your OS is Windows 10/11 you can use Windows Subsytem for Linux (WSL) tool or create the virtual environment manually.
To check if the Conda enviroment is activated you should see a (conda_env_name) in your command line. If it is not activated, then you can activated it using:

```bash
conda activate conda_env_name
```

## Overview
The experiments in Conmo have a pipeline-based architecture. A pipeline consists of a chain of processes connected in such a way that the output of each element of the chain is the input of the next, thus creating a data flow. Each of these processes represents one of the typical generic steps in Machine Learning experiments. These steps are:

1. Datasets
2. Splitters
3. Preprocesses
4. Algorithms
5. Metrics

In "/examples" folder there are a small set of Conmo experiments with source code explained to try to help you understand how this framework works.

## Authors
Conmo has been developed by [Metrology and Models Group](https://mym.grupos.uniovi.es/en/inicio) in University of Oviedo (Principality of Asturias, Spain).
However, as this is a collaborative project it is intended that anyone can include their own experiments, so please feel free to collaborate with us!

## Issues & Bugs
As the project is currently in an early stage of development, it is easy for different bugs and issues to appear, so please, if you detect one post a ticket in the issues tab or contact the developers directly via email: mym.inv.uniovi@gmail.com