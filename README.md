# Direct stellarator coil design using global optimization: application to a comprehensive exploration of quasi-axisymmetric devices
The examples directory contain scripts for executing the globalization then phase I of the workflow in

*Direct stellarator coil design using global optimization: application to a comprehensive exploration of quasi-axisymmetric devices*, A. Giuliani, Arxiv

## Background
There are two options for globalization: a naive approach where an ensemble of initial guesses obtained by perturbing initially flat coils, or a less ad-hoc approach based on TuRBO.  We search for coils with near-axis quasisymmetry using the optimization problem described in:

*Single-stage gradient-based stellarator coil design: Optimization for near-axis quasi-symmetry*, A Giuliani, F Wechsung, A Cerfon, G Stadler, M Landreman, Journal of Computational Physics 459, 111147

The goal of the scripts in this work is now to properly globalize the direct coil design algorithm.


## Installation
To use this code, first clone the repository including all its submodules, via

    git clone --recursive 

Next, best practice is to generate a virtual environment and install PyPlasmaOpt there:

    cd PyPlasmaOpt
    python -m venv venv
    source venv/bin/activate
    cd LinkingNumber; mkdir build; cd build; cmake ..; make; cd ../../
    cd TuRBO; pip install -e .; cd ..
    pip install -e .

## Running the scripts

To run the near-axis optimization with TuRBO globalization:

    ./ex_TuRBO.py arguments.txt

with naive globalization (perturbing the initial guess with Gaussian noise):

    ./ex_naive.py arguments.txt

Typically, the practitioner will have to run the optimization multiple times for a fair comparison of the the two techniques.
