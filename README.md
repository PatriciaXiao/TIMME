# TIMME

This repository contains the code implementation and the data of *TIMME: Twitter Ideology-detection via Multi-task Multi-relational Embedding*.

Please cite [our paper](https://arxiv.org/abs/2006.01321) to use the [dataset](./data/).

## Environment

Language: Tested on Python 3.6, 3.7 or 3.8. All worked well.

Pre-requisites (other versions might apply as well, these are the developing environment we've used):

| Python | torch | pandas | numpy  | scipy | scikit-learn |
|:------:| :---: | :----: | :----: | :---: | :----------: |
| 3.8    | 1.4.0 |  1.0.3 | 1.18.2 | 1.4.1 |  0.23.1      |
| 3.7    | 1.4.0 | 0.6.3  | 1.17.2 | 1.3.1 |  0.20.2      |
| 3.6    | 1.3.1 | 0.23.4 | 1.15.4 | 1.1.0 |  0.20.2      |

All other dependencies should be automatically installed once you get these packages installed. Otherwise please follow the instruction to install missing packages, and welcome to give us feedback if we gave the wrong version.

## Related Reporitories

Baseline models, and the geography visualization code, are given in separate repositories.

### Baseline Models

Pending: repository to be released by Haoyan.

by Haoyan Xu

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
    * single classification task: ```python main.py -e 20```
    * TIMME-single: ```python main.py -e 600 -t TIMME_SingleLink --single_relation 0``` single link-prediction task of single relation 0; relations are labeled 0, 1, 2, 3, 4 in our case.
- TIMME
    * TIMME (basic): ```python main.py -e 20 -t TIMME```
- TIMME-hierarchical
    * TIMME-hierarchical: ```python main.py -e 20 -t TIMME_hierarchical```

| Model Name (in [paper](https://arxiv.org/abs/2006.01321)) | Command Line Flag (in [main](./code/main.py)) | Task Manager (in [task](./code/task.py)) | Model Name (in [model](./code/model/model.py)) |
|:---------------------:| :-------------------------: | :------------------: | :------------------: |
| N/A                   | Classification              | ClassificationTask   | Classification       |
| N/A                   | LinkPrediction              | LinkPredictionTask   | LinkPrediction       |
| TIMME-single          | TIMME_SingleLink            | TIMMEManager         | TIMMEsingle          |
| TIMME                 | TIMME                       | TIMMEManager         | TIMME                |
| TIMME-hierarchical    | TIMME_hierarchical          | TIMMEManager         | TIMMEhierarchical    |

