# dp_inference

This directory contains code for performing differential privacy inference using the dp_inference package.

## Installation

To set up and install the `dp_inference` conda environment, follow these steps:

```shell
conda create -n dp_inference python=3.10
conda install numpy scipy scikit-learn matplotlib cvxpy pandas -c conda-forge
pip install folktables
pip install -e .
```


The code uses multiprocessing to correctly run on a linux environment, To run it without multiprocessing, change the corresponding script files. The dp_inference folder contains the main functions implementing the algorithms and the scripts folder containts scripts that call those functions to run the experiments in the paper. An example command to run the code is:

```shell
python scripts/one_dim_methods.py 
```
