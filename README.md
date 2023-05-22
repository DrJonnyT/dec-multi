# DEC-Keras
Deep Embedding Clustering in Keras [https://arxiv.org/abs/1511.06335]

Based on a fork from [the original by fferroni](https://github.com/fferroni/DEC-Keras)


Scripts to run kmeans and deep embedded clustering many times on different sized samples from the mnist dataset and output the results to CSV:
* run_kmeans_many_times.py
* run_dec_many_times.py

Scripts to process the results from clustering
* process_results.py
* process_full_mnist_results_nruns.py
* process_multi_labels_csv.py


Folder containing the forked (and modified) backend
* keras_dec/


## System requirements
For full functionality, a GPU with >=6GB VRAM is required. Testing on a system with 4GB VRAM shows the system is unable to run with more than around 30000 images.

## Environment setup
1. Use conda to create and activate new environment with most packages:
```
conda env create -f environment.yml
conda activate dec-keras
```

2. Install tensorflow from pip:
```
python -m pip install "tensorflow<2.11"
```
These instructions have been tested on Windows 10 native.

