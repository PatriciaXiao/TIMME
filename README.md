# TIMME

## Data Updates on 7 March 2021: Text provided

We proide the tweet IDs the **P_all** accounts posted on [Box](https://ucla.box.com/s/47i6p7mkaer1f4ec8c0qcyxj1u0g3z8m).
Besides, considering that it was collected long time ago, thus some tweets are no longer available on Twitter, we provide the pure text content (grouped by account IDs) as a [zipped file](https://ucla.box.com/s/nk27vfb26jfhqrvyfv9e8m3dvbz11sq1) as well. 

## Introduction

This repository contains the code implementation and the data of *TIMME: Twitter Ideology-detection via Multi-task Multi-relational Embedding*. The paper is [in proceedings of KDD'20, Applied Data Science Track](https://dl.acm.org/doi/10.1145/3394486.3403275).

Please cite [our paper](https://arxiv.org/abs/2006.01321) to use the [dataset](./data/) or the [code](./code/).

Published version available at [ACM Digital Library](https://dl.acm.org/doi/10.1145/3394486.3403275). Shown on their website you can cite us as:

```
@inproceedings{10.1145/3394486.3403275,
author = {Xiao, Zhiping and Song, Weiping and Xu, Haoyan and Ren, Zhicheng and Sun, Yizhou},
title = {TIMME: Twitter Ideology-Detection via Multi-Task Multi-Relational Embedding},
year = {2020},
isbn = {9781450379984},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394486.3403275},
doi = {10.1145/3394486.3403275},
abstract = {We aim at solving the problem of predicting people's ideology, or political tendency. We estimate it by using Twitter data, and formalize it as a classification problem. Ideology-detection has long been a challenging yet important problem. Certain groups, such as the policy makers, rely on it to make wise decisions. Back in the old days when labor-intensive survey-studies were needed to collect public opinions, analyzing ordinary citizens' political tendencies was uneasy. The rise of social medias, such as Twitter, has enabled us to gather ordinary citizen's data easily. However, the incompleteness of the labels and the features in social network datasets is tricky, not to mention the enormous data size and the heterogeneousity. The data differ dramatically from many commonly-used datasets, thus brings unique challenges. In our work, first we built our own datasets from Twitter. Next, we proposed TIMME, a multi-task multi-relational embedding model, that works efficiently on sparsely-labeled heterogeneous real-world dataset. It could also handle the incompleteness of the input features. Experimental results showed that TIMME is overall better than the state-of-the-art models for ideology detection on Twitter. Our findings include: links can lead to good classification outcomes without text; conservative voice is under-represented on Twitter; follow is the most important relation to predict ideology; retweet and mention enhance a higher chance of like, etc. Last but not least, TIMME could be extended to other datasets and tasks in theory.},
booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
pages = {2258–2268},
numpages = {11},
keywords = {social network analysis, ideology detection, graph convolutional networks, heterogeneous information network, multi-task learning},
location = {Virtual Event, CA, USA},
series = {KDD '20}
}
```

Or, if you prefer citing ArXiv Preprint:

```
@article{xiao2020timme,
  title={TIMME: Twitter Ideology-detection via Multi-task Multi-relational Embedding},
  author={Xiao, Zhiping and Song, Weiping and Xu, Haoyan and Ren, Zhicheng and Sun, Yizhou},
  journal={arXiv preprint arXiv:2006.01321},
  year={2020}
}
```

Presentation videos are on my [personal website](http://web.cs.ucla.edu/~patricia.xiao/timme.html).
Note that I was doing live presentation during the conference session, so you might find the pre-recorded video's content slightly different from the live version.

## Typos in the Paper

### Figure 9 typo

Figure 9 (one of the ablation studies): instead of being F1, it should be ROC-AUC score shown in the graph.
It is link-prediction task performance, for link-prediction we only use ROC-AUC and PR-AUC.

Sorry for all the inconvenience it brought you.

### section 4.1 typo

There is an "𝛼 == ", where it should be "=" instead of "==".

## Environment

Language: Tested on Python 3.6, 3.7 and 3.8. All worked well.

Pre-requisites (other versions might apply as well, these are the developing environment we've used):

| Python | torch | pandas | numpy  | scipy | scikit-learn |
|:------:| :---: | :----: | :----: | :---: | :----------: |
| 3.8    | 1.4.0 |  1.0.3 | 1.18.2 | 1.4.1 |  0.23.1      |
| 3.7    | 1.4.0 | 0.6.3  | 1.17.2 | 1.3.1 |  0.20.2      |
| 3.6    | 1.3.1 | 0.23.4 | 1.15.4 | 1.1.0 |  0.20.2      |

All other dependencies should be automatically installed once you get these packages installed. Otherwise please follow the instruction to install missing packages, and welcome to give us feedback if we gave the wrong version.

Where there's CUDA available, it should automatically use the GPU #0.
Sorry we didn't specify the GPU id, if you need to do so, you can do it by replacing:
```
model_or_tensor.cuda()
```
with
```
model_or_tensor.cuda(gpu_id_you_specify)
```
Enjoy.

## Related Reporitories

Baseline models, and the geography visualization code, are given in separate repositories.

### Baseline Models

Implemented and tuned by [Haoyan Xu](https://github.com/uphoupho).
* GCN baseline we use comes from [Kpif](https://github.com/tkipf/pygcn)
* rGCN baseline we use comes from [Kpif](https://github.com/tkipf/relational-gcn)
* HAN baseline we use comes from [DGL](https://github.com/dmlc/dgl/tree/master/examples/pytorch/han)

For link prediction tasks, we use the [same NTN component](https://github.com/PatriciaXiao/TIMME/blob/master/code/model/model.py#L91-L107) as used in TIMME.

### Geography Visualization

Geography visualization tasks envolve two parts: first, interprete the location from the chaotic Twitter user profiles; second, plot the maps according to our results.

* [TIMME-formatted-location](https://github.com/franklinnwren/TIMME-formatted-location)
* [TIMME-data-visualization](https://github.com/franklinnwren/TIMME-data-visualization)

by Zhicheng Ren.

## Usage

### Sample Usage
```shell
python main.py -e 20 --fixed_random_seed
```

### Hyperparameters

For more options please refer to command line arguments specified in [main.py](./code/main.py).
There are some other details that aren't listed here, for which we use the default settings set in our code.

#### Number of epochs
```shell
python main.py --epochs 20
```
or
```shell
python main.py -e 20
```
means running for *20* epochs.

#### Random Seed
```shell
python main.py -e 20 --fixed_random_seed
```
or
```shell
python main.py -e 20 -frd
```
fix the random seeds.

#### Dataset
```shell
python main.py -e 20 --data P50
```
or
```shell
python main.py -e 20 -d P50
```
will run the dataset **P50**, instead of the default **PureP**. For the dataset options please check the [data](./data) we have.

#### Feature
```shell
python main.py -e 20 --feature one_hot
```
or
```shell
python main.py -e 20 -f one_hot
```
will get our model run with the fixed one_hot features. Options include: "tweets_average", "description", "status", "one_hot", "random". Among which, "tweets_average", "description", "status" are the features that are partly-trainable. In practice, we found one_hot works the best in our case. We suspect that it is caused by the quality of our features.

#### The variations of the models

Command line argument ```--task``` or ```-t```.

Sample usage:
- TIMME-single
    * single classification task: ```python main.py -e 20``` (it is the default option) or ```python main.py -e 20 -t Classification```
    * TIMME-single: ```python main.py -e 600 -t TIMME_SingleLink --single_relation 0``` single link-prediction task of single relation ```0```; relations are labeled ```0, 1, 2, 3, 4``` in our case.
- TIMME
    * TIMME (basic): ```python main.py -e 20 -t TIMME```
- TIMME-hierarchical
    * TIMME-hierarchical: ```python main.py -e 20 -t TIMME_hierarchical```

The link-prediction component is a simplified version of NTN model, refered to as **TIMME-NTN** for convenience in our paper, implemented in LinkPrediction model at [here](https://github.com/PatriciaXiao/TIMME/blob/master/code/model/model.py#L91-L107).

| Model Name (in [paper](https://arxiv.org/abs/2006.01321)) | Command Line Flag (in [main](./code/main.py)) | Task Manager (in [task](./code/task.py)) | Model Name (in [model](./code/model/model.py)) |
|:---------------------:| :-------------------------: | :------------------: | :------------------: |
| N/A                   | Classification              | ClassificationTask   | Classification       |
| N/A                   | LinkPrediction              | LinkPredictionTask   | LinkPrediction       |
| TIMME-single          | TIMME_SingleLink            | TIMMEManager         | TIMMEsingle          |
| TIMME                 | TIMME                       | TIMMEManager         | TIMME                |
| TIMME-hierarchical    | TIMME_hierarchical          | TIMMEManager         | TIMMEhierarchical    |




