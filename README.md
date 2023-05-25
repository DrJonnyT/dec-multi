# dec-multi
Exploring Deep Embedding Clustering (DEC) using different sized datasets of MNIST digits. DEC is based on [the original paper by Xie et al.](https://arxiv.org/abs/1511.06335)

This repo is based on a fork from [a Keras implementation of DEC by fferroni](https://github.com/fferroni/DEC-Keras).

## System requirements
For full functionality, a GPU with >=6GB VRAM is required. Testing on a system with 4GB VRAM shows the system is unable to run with more than around 30000 images. Even with 6GB VRAM, running DEC with the full MNIST dataset often runs out of VRAM and has to restart.

## Environment setup
1. Use conda to create and activate new environment with most packages:
```
conda env create -f environment.yml
conda activate dec-multi
```

2. Install tensorflow from pip:
```
python -m pip install "tensorflow<2.11"
```
These instructions have been tested on Windows 10 native.


## Typical workflow

| | File | Estimated time | Function |
| --- | --- | --- | --- |
| 1. | run_kmeans_many_times.py | 30 minutes | Run kmeans many times on the MNIST digits and save to CSV. |
| 2. | run_dec_many_times.py | A few hours to a few days depending on how many iterations | Run DEC many times on the MNIST digits and save to CSV. |
| 3. | process_multi_labels_csv.py | 10 hours | Load the results from the above, calculate accuracy and aggregate labels using mode and probabalistic label aggregation. Save to CSV. |
| 4. | plot_multi_labels_from_csv.py | A few seconds | Load aggegated labels accuracy and plot. |

Estimated times are provided running on a mid-range laptop CPU and low-end laptop GPU (in the year 2023)
