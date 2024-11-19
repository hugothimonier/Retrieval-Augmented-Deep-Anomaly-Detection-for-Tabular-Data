# Making Parametric Anomaly Detection on Tabular Data Non-Parametric Again

[![arXiv](https://img.shields.io/badge/arXiv-2401.17052-b31b1b.svg)](https://arxiv.org/abs/2401.17052)

  **[Overview](#overview)**
| **[Installation](#installation)**
| **[Examples](#examples)**

## Overview

This repo contains the code to run the experiments in "Making Parametric Anomaly Detection on Tabular Data Non-Parametric Again".

## Installation

Set up and activate the Python environment by executing

```
conda env create -f environment.yml
```
Make sure to have the **latest version of condas**.

## Datasets

To download all datasets at once, with `wget`:
```
bash get_dataset_wget.sh
```
with `curl`:
```
bash get_dataset_curl.sh
```

## Examples

To run the experiments for each dataset, without retrieval, for cpu or mono-gpu:
```
source ./scripts/cpu/no_retrieval/abalone.sh
```
where ``abalone`` can be replaced by any dataset in the paper. 

For distributed training, change the number of GPUs accordingly in ``./scripts/distributed/no_retrieval/abalone.sh`` and run:

```
--nnodes=$NUMBER_OF_NODE --nproc_per_node=$NUMBER_OF_GPUS_PER_NODE
``` 
```
--mp_nodes $NUMBER_OF_NODE        #number of computing nodes
--mp_gpus $TOTAL_NUMBER_OF_GPUS   #total number of gpus
``` 

Similarly, for retrieval-augmented methods, replace ``no_retrieval`` in the previous path by the chosen retrieval method in ``['knn', 'v-attention', 'attention_bsim', 'attention_bsim_bval']``. For ``abalone`` and ``knn``retrieval, run the following:
```
source ./scripts/cpu/knn/abalone.sh
```
or 
```
source ./scripts/distributed/knn/abalone.sh
```

## Citation

If you use this code for your work, please cite our paper
[Paper](https://arxiv.org/abs/2401.17052) as

```bibtex
@inproceedings{thimonier2024making,
author = {Thimonier, Hugo and Popineau, Fabrice and Rimmel, Arpad and Doan, Bich-Li\^{e}n},
title = {Retrieval Augmented Deep Anomaly Detection for Tabular Data},
year = {2024},
isbn = {9798400704369},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3627673.3679559},
doi = {10.1145/3627673.3679559},
booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
pages = {2250–2259},
numpages = {10},
keywords = {anomaly detection, deep learning, tabular data},
location = {Boise, ID, USA},
series = {CIKM '24}
}
```
