# Higher-order LOD method

## Description
This code contains the examples run in the paper [^KruM25]. This codebase acts as a proof of concept for the higher-order LOD method and is not optimized in any way.  

## Installation
Set up your virtual environment in the parent folder `plodwave/` using

    sudo apt update  
    sudo apt install python3.12-dev python3.12-venv  
    sudo apt install libsuitesparse-dev swig  
    
    python3.12 -m venv .venv  

and activate the environment using

    source .venv/bin/activate  

Install the package by  

    pip install pip wheel setuptools --upgrade  
    pip install -e .  

## Usage
Run the examples in the following way inside the virtual environment providing a path where the data is saved.  

    python examples/example_1/main.py [save_path]  
    python examples/example_2/main.py [save_path]  
    python examples/example_2_1/main.py [save_path]  
    python examples/example_2_2/main.py [save_path]  
    python examples/example_3/main.py [save_path]  

[^KruM25]: F. Krumbiegel and R. Maier, A higher order multiscale method for the wave equation, IMA J. Numer. Anal. {\bf 45} (2025), no.~4, 2248--2273, https://doi.org/10.1093/imanum/drae059
