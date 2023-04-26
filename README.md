# DEC-Keras
Deep Embedding Clustering in Keras [https://arxiv.org/abs/1511.06335]

Based on a fork from [the original by fferroni](https://github.com/fferroni/DEC-Keras)


Scripts to run kmeans and deep embedded clustering many times on different sized samples from the mnist dataset:
* run_kmeans_many_times.py
* run_dec_many_times.py

Scripts to process the results from clustering
* process_results.py

Folder of parameter files
* params/

Folder containing the forked (and modified) backend
* keras_dec/


## Environment setup
1. **Install the following packages from conda:**
   - tensorflow-gpu (2.6.0)
   * pandas (1.5.2)
   * pytest (7.1.2)

2. **Install the following packages from pip:**
   * matplotlib (3.7.1)

These instructions have been tested on Windows 10 native. If you install matplotlib from anaconda it might install ok but fail to run.

