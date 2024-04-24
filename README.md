# dp_inference

This directory contains code for performing differential privacy inference using the dp_inference package.

## Installation

To set up and install the `dp_inference` conda environment, follow these steps:

1. Create the conda environment from the `environment.yml` file:

    ```shell
    conda env create -f environment.yml
    ```

    This will create a new conda environment named `dp_inference` with all the necessary dependencies.

2. Activate the `dp_inference` environment:

    ```shell
    conda activate dp_inference
    ```

    You are now ready to use the `dp_inference` package for differential privacy inference.

3. Next, use the following command to the dp_inference package locally:
    
    ```shell
    pip install -e .
    ```


The code uses multiprocessing to correctly run on a linux environment, To run it without multiprocessing, change the corresponding script files. The dp_inference folder contains the main functions implementing the algorithms and the scripts folder containts scripts that call those functions to run the experiments in the paper. An example command to run the code is:

    ```shell
    python scripts/one_dim_methods.py 
    ```
