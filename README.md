# DEC-Keras
Deep Embedding Clustering in Keras [https://arxiv.org/abs/1511.06335]

Based on a fork from [the original by fferroni](https://github.com/fferroni/DEC-Keras)


Scripts to run kmeans and deep embedded clustering many times on different sized samples from the mnist dataset:
* run_kmeans_many_times.py
* run_dec_many_times.py

Scripts to process the results from clustering
* process_results.py


Folder containing the forked (and modified) backend
* keras_dec/


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

